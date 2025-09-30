import cv2
import numpy as np

# --- CONFIGURACIÓN ---
IMAGE_PATH = "input.png"  # Cambia si tu imagen tiene otro nombre
OUTPUT_PATH = "output_with_right_eye_glints.png"

# --- LEE LA IMAGEN ---
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise Exception("No se pudo leer la imagen. Verifica el nombre y la ubicación.")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --- DETECCIÓN DE GLINTS ---
# Usamos un threshold alto para encontrar los puntos más brillantes
min_val, max_val, _, _ = cv2.minMaxLoc(gray)
glint_thresh = max(100, int(max_val * 0.85))
_, thresh = cv2.threshold(gray, glint_thresh, 255, cv2.THRESH_BINARY)

# Un poco de morfología para limpiar
kernel = np.ones((3, 3), np.uint8)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

# Encuentra los contornos de los glints
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
glints = []
for cnt in contours:
    area = cv2.contourArea(cnt)
    if 3 < area < 120:  # Filtra glints muy pequeños o grandes
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            glints.append({'center': (cx, cy), 'area': area})

# --- SELECCIONA GLINTS DEL OJO DERECHO DE LA IMAGEN ---
h, w = gray.shape
center_x = w // 2
# Ojo derecho de la imagen está a la derecha del centro
right_eye_glints = [g for g in glints if g['center'][0] > center_x]

# Ordena por área y toma los dos más grandes (puedes ajustar si quieres más o menos)
right_eye_glints = sorted(right_eye_glints, key=lambda g: g['area'], reverse=True)[:2]

# --- DIBUJA GLINTS EN AMARILLO ---
for glint in right_eye_glints:
    cv2.circle(img, glint['center'], 4, (0, 255, 255), -1)  # Amarillo

# --- GUARDA Y/O MUESTRA EL RESULTADO ---
cv2.imwrite(OUTPUT_PATH, img)
cv2.imshow("Glints Ojo Derecho", img)
cv2.waitKey(0)
cv2.destroyAllWindows()