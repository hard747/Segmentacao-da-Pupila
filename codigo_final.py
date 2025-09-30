import cv2
import numpy as np
import time
import random
import os
import serial

# Parámetros generales
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
GRID_ROWS_CALIB = 3
GRID_COLS_CALIB = 3
GRID_ROWS_PRED = 5
GRID_COLS_PRED = 7
ALVO_SHOW_TIME = 2.0
CAMERA_INDEX = 0
FPS = 60

GRID_COLOR = (255, 255, 255)
CROSS_COLOR = (0, 255, 0)
CROSS_SIZE = 40

MIN_BRIGHT_PIXEL_VALUE = 200
MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL = 40
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL = 0.85
GLINT_MIN_AREA = 3
GLINT_MAX_AREA = 120
GLINT_CIRCULARITY_THRESHOLD = 0.65

ARDUINO_PORT = '/dev/ttyACM0'
ARDUINO_BAUD = 115200

def check_arduino_connection(port=ARDUINO_PORT, baud=ARDUINO_BAUD):
    try:
        ser = serial.Serial(port, baud, timeout=1)
        time.sleep(2)
        ser.write(b'ping\n')
        time.sleep(0.1)
        response = ser.readline()
        ser.close()
        print("Arduino conectado correctamente.")
        return True
    except Exception as e:
        print(f"Error al conectar con Arduino: {e}")
        return False

def get_grid_centers(cols, rows, width, height):
    cell_w = width / cols
    cell_h = height / rows
    centers = []
    for row in range(rows):
        for col in range(cols):
            x = col * cell_w + cell_w / 2
            y = row * cell_h + cell_h / 2
            centers.append((x, y))
    return centers

def draw_grid(image, cols, rows, width, height):
    cell_w = width / cols
    cell_h = height / rows
    for i in range(1, cols):
        x = int(i * cell_w)
        cv2.line(image, (x, 0), (x, height), GRID_COLOR, 1)
    for i in range(1, rows):
        y = int(i * cell_h)
        cv2.line(image, (0, y), (width, y), GRID_COLOR, 1)

def draw_cross(image, center, color, size):
    x, y = int(center[0]), int(center[1])
    cv2.line(image, (x - size//2, y), (x + size//2, y), color, 2)
    cv2.line(image, (x, y - size//2), (x, y + size//2), color, 2)

def detect_glints(img_gray, roi_mask=None):
    img_to_process = img_gray.copy()
    if roi_mask is not None:
        img_to_process = cv2.bitwise_and(img_gray, img_gray, mask=roi_mask)
    min_val, max_val, _, _ = cv2.minMaxLoc(img_to_process)
    if max_val < 50:
        return []
    glint_thresh = int(max_val * GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL)
    glint_thresh = max(100, min(240, glint_thresh))
    _, thresh = cv2.threshold(img_to_process, glint_thresh, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    glints = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if GLINT_MIN_AREA < area < GLINT_MAX_AREA:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                if circularity > GLINT_CIRCULARITY_THRESHOLD:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        glints.append({'center': (cx, cy), 'area': area, 'circularity': circularity})
    return glints

def detect_pupil_center(img_gray):
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return None
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def fit_polynomial_2nd_order(pupil_points, screen_points):
    N = len(pupil_points)
    A = np.zeros((N, 6))
    Bx = np.zeros(N)
    By = np.zeros(N)
    for i, ((x, y), (X, Y)) in enumerate(zip(pupil_points, screen_points)):
        A[i] = [x**2, x*y, y**2, x, y, 1]
        Bx[i] = X
        By[i] = Y
    coeffs_x, _, _, _ = np.linalg.lstsq(A, Bx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, By, rcond=None)
    return coeffs_x, coeffs_y

def map_pupil_to_screen(x, y, coeffs_x, coeffs_y):
    args = np.array([x**2, x*y, y**2, x, y, 1])
    X = np.dot(coeffs_x, args)
    Y = np.dot(coeffs_y, args)
    return (X, Y)

def main():
    # --- Verifica conexión Arduino ---
    print("Verificando conexión con Arduino...")
    arduino_ok = check_arduino_connection()
    if not arduino_ok:
        print("Atención: Arduino NO conectado. Verifica la conexión y reinicia el programa si es necesario.")

    # --- Prepara salida de datos ---
    os.makedirs("output_gaze", exist_ok=True)
    raw_video_path = os.path.join("output_gaze", "video_ojos.avi")
    gaze_data_path = os.path.join("output_gaze", "gaze_data.txt")
    alvo_data_path = os.path.join("output_gaze", "alvo_data.txt")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SCREEN_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SCREEN_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(raw_video_path, fourcc, FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
    if not cap.isOpened() or not out_video.isOpened():
        print("No se pudo abrir la cámara o el VideoWriter.")
        return

    # --- Calibración: grilla 3x3 ---
    print("Fase de calibración: mira cada alvo cuando aparece en pantalla.")
    grid_centers = get_grid_centers(GRID_COLS_CALIB, GRID_ROWS_CALIB, SCREEN_WIDTH, SCREEN_HEIGHT)
    pupil_calib_points = []
    screen_calib_points = []
    timestamps_calib = []
    frame_gaze_data = []
    frame_alvo_data = []

    alvo_idx = 0
    alvo_start_time = time.time()
    calib_phase = True
    calib_done = False

    cv2.namedWindow("Calibracion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while calib_phase:
        ret, frame = cap.read()
        if not ret:
            continue
        ts = time.time()
        out_video.write(frame)
        img_show = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_grid(img_show, GRID_COLS_CALIB, GRID_ROWS_CALIB, SCREEN_WIDTH, SCREEN_HEIGHT)
        draw_cross(img_show, grid_centers[alvo_idx], CROSS_COLOR, CROSS_SIZE)
        cv2.putText(img_show, f"Calibracion: Mira el alvo {alvo_idx+1}/{len(grid_centers)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.imshow("Calibracion", img_show)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_eye = gray[0:h, w//2:w]
        center_pupil = detect_pupil_center(roi_eye)
        glints = detect_glints(roi_eye)

        frame_gaze_data.append({
            'timestamp': ts,
            'center_pupil': (center_pupil[0]+w//2, center_pupil[1]) if center_pupil else None,
            'glints': [(g['center'][0]+w//2, g['center'][1]) for g in glints]
        })
        frame_alvo_data.append({
            'timestamp': ts,
            'alvo_idx': alvo_idx,
            'alvo_coords': grid_centers[alvo_idx]
        })

        if time.time() - alvo_start_time >= ALVO_SHOW_TIME:
            if center_pupil:
                pupil_calib_points.append((center_pupil[0]+w//2, center_pupil[1]))
                screen_calib_points.append(grid_centers[alvo_idx])
                timestamps_calib.append(ts)
            alvo_idx += 1
            alvo_start_time = time.time()
            if alvo_idx >= len(grid_centers):
                calib_phase = False
                calib_done = True

        key = cv2.waitKey(1)
        if key == 27:
            break

    cv2.destroyAllWindows()
    print("Calibracion terminada.")

    # --- Ajusta polinomios de mapeo ---
    print("Ajustando mapeo polinomial de 2do grado...")
    coeffs_x, coeffs_y = fit_polynomial_2nd_order(pupil_calib_points, screen_calib_points)
    print("Coeficientes X:", coeffs_x)
    print("Coeficientes Y:", coeffs_y)

    # --- Descansa usuario y espera Enter ---
    input("Descanso antes de predicción. Presiona ENTER para continuar...")

    # --- Predicción: grilla 5x7, alvos aleatorios y sin repetición ---
    print("Fase de predicción: grilla 5x7, alvos aleatorios sin repetir.")
    grid_pred_centers = get_grid_centers(GRID_COLS_PRED, GRID_ROWS_PRED, SCREEN_WIDTH, SCREEN_HEIGHT)
    random_pred_order = list(range(len(grid_pred_centers)))
    random.shuffle(random_pred_order)
    pred_idx = 0
    alvo_pred_start_time = time.time()
    pred_phase = True

    cv2.namedWindow("Prediccion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Prediccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pred_frame_gaze_data = []
    frame_pred_alvo_data = []

    while pred_phase:
        ret, frame = cap.read()
        if not ret:
            break
        ts = time.time()
        out_video.write(frame)
        img_show = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_grid(img_show, GRID_COLS_PRED, GRID_ROWS_PRED, SCREEN_WIDTH, SCREEN_HEIGHT)
        alvo_pos = grid_pred_centers[random_pred_order[pred_idx]]
        draw_cross(img_show, alvo_pos, CROSS_COLOR, CROSS_SIZE)
        cv2.putText(img_show, f"Prediccion: Mira el alvo {pred_idx+1}/{len(grid_pred_centers)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.imshow("Prediccion", img_show)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_eye = gray[0:h, w//2:w]
        center_pupil = detect_pupil_center(roi_eye)
        glints = detect_glints(roi_eye)
        if center_pupil:
            cx_global = center_pupil[0] + w//2
            cy_global = center_pupil[1]
            pred_X, pred_Y = map_pupil_to_screen(cx_global, cy_global, coeffs_x, coeffs_y)
            draw_cross(img_show, (pred_X, pred_Y), (0,0,255), CROSS_SIZE//2)
            cv2.putText(img_show, f"Prediccion: ({pred_X:.0f},{pred_Y:.0f})", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            pred_frame_gaze_data.append({
                'timestamp': ts,
                'center_pupil': (cx_global, cy_global),
                'pred_screen': (pred_X, pred_Y),
                'glints': [(g['center'][0]+w//2, g['center'][1]) for g in glints]
            })
        frame_pred_alvo_data.append({
            'timestamp': ts,
            'alvo_idx': random_pred_order[pred_idx],
            'alvo_coords': alvo_pos
        })

        if time.time() - alvo_pred_start_time >= ALVO_SHOW_TIME:
            pred_idx += 1
            alvo_pred_start_time = time.time()
            if pred_idx >= len(grid_pred_centers):
                pred_phase = False

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()

    # --- Guarda datos en archivos ---
    with open(gaze_data_path, 'w') as f:
        for d in frame_gaze_data:
            f.write(f"{d}\n")
    with open(alvo_data_path, 'w') as f:
        for d in frame_alvo_data:
            f.write(f"{d}\n")
    with open(os.path.join("output_gaze", "pred_gaze_data.txt"), 'w') as f:
        for d in pred_frame_gaze_data:
            f.write(f"{d}\n")
    with open(os.path.join("output_gaze", "pred_alvo_data.txt"), 'w') as f:
        for d in frame_pred_alvo_data:
            f.write(f"{d}\n")
    print("Datos guardados en carpeta output_gaze")

if __name__ == "__main__":
    main()