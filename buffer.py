# Importaciones de bibliotecas necesarias
import cv2
import numpy as np
import time
import random
import os
import serial
import pyudev
import atexit
from scipy.ndimage import convolve1d

# ==============================================================================
# CONSTANTES GLOBALES
# ==============================================================================
CROSS_COLOR = (0, 255, 0)
CROSS_SIZE = 40
GRID_COLOR = (255, 255, 255)
ALVO_SHOW_TIME = 2.0  # Tiempo que se muestra cada objetivo en segundos
FPS = 60
SCREEN_WIDTH = 1920  # Ancho de pantalla deseado para estímulos
SCREEN_HEIGHT = 1080 # Alto de pantalla deseado para estímulos
CAMERA_WIDTH = 640   # Resolución forzada para el feed de la cámara
CAMERA_HEIGHT = 480  # Resolución forzada para el feed de la cámara

# ==============================================================================
# GERENCIADOR DE DISPOSITIVOS (HARDWARE SETUP)
# ==============================================================================
class GerenciadorDispositivos:
    """Clase para inicializar y gestionar la cámara PS3 Eye y la comunicación serial con Arduino."""
    PS3_EYE_VENDOR_ID = '1415'
    PS3_EYE_MODEL_ID = '2000'
    PORTA_ARDUINO = '/dev/ttyACM0'
    TAXA_BAUD = 112500
    TEMPO_ESPERA_ARDUINO_SEG = 0.1

    def __init__(self, fps_alvo, duracao_estrobe, pre_atraso):
        self.captura = None
        self.serial = None
        self.largura_frame = CAMERA_WIDTH  # Usar las constantes forzadas
        self.altura_frame = CAMERA_HEIGHT # Usar las constantes forzadas
        self.FPS_ALVO = fps_alvo
        self.DURACAO_ESTROBE_INICIAL_US = duracao_estrobe
        self.PRE_ATRASO_ESTROBE_INICIAL_US = pre_atraso

    def obter_id_camera_ps3_eye(self):
        """Busca el ID de la cámara PS3 Eye usando pyudev."""
        print("Buscando cámara PS3 Eye usando pyudev...")
        try:
            contexto = pyudev.Context()
            for dispositivo in contexto.list_devices(subsystem='video4linux'):
                propriedades = dispositivo.properties
                if 'ID_VENDOR_ID' in propriedades and 'ID_MODEL_ID' in propriedades:
                    if propriedades['ID_VENDOR_ID'] == self.PS3_EYE_VENDOR_ID and propriedades['ID_MODEL_ID'] == self.PS3_EYE_MODEL_ID:
                        try:
                            indice_camera = int(dispositivo.device_node.replace('/dev/video', ''))
                            print(f"Câmera PS3 Eye encontrada en: {dispositivo.device_node} (ID OpenCV: {indice_camera})")
                            return indice_camera
                        except ValueError:
                            print(f"Aviso: No fue posible analizar el ID numérico de {dispositivo.device_node}.")
                            return None
            print(f"Dispositivo PS3 Eye (Proveedor:{self.PS3_EYE_VENDOR_ID}, Modelo:{self.PS3_EYE_MODEL_ID}) no encontrado usando pyudev.")
            return None
        except ImportError:
            print("La biblioteca 'pyudev' no está instalada o no se puede acceder. Intentando ID 0...")
            return 0
        except Exception as e:
            print(f"Error al buscar dispositivos udev: {e}. Intentando ID 0...")
            return 0

    def iniciar_camera_e_serial(self):
        """Inicializa la cámara y la conexión serial con Arduino, forzando la resolución."""
        print("--- Inicializando Câmera PS3 Eye ---")
        id_camera = self.obter_id_camera_ps3_eye()
        if id_camera is None:
            print("Error: Câmera PS3 Eye no encontrada o ID no disponible.")
            return False

        # Configuración de OpenCV para la cámara
        self.captura = cv2.VideoCapture(id_camera)
        if not self.captura.isOpened():
            print(f"Error: No fue posible abrir la cámara con OpenCV (ID: {id_camera}).")
            return False
            
        # FORZAR RESOLUCIÓN 640x480
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.captura.set(cv2.CAP_PROP_FPS, self.FPS_ALVO)
        time.sleep(1)
        
        # Verificar resolución real establecida (aunque forzamos, puede variar)
        self.largura_frame = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.altura_frame = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Câmera abierta con OpenCV (ID: {id_camera}).")
        fps_real = self.captura.get(cv2.CAP_PROP_FPS)
        print(f"FPS real de la cámara medido: {fps_real:.2f} FPS.")
        print(f"Resolución de captura de la cámara: {self.largura_frame}x{self.altura_frame}")

        # Configuración de comunicación Serial
        print("\n--- Inicializando conexión Serial con Arduino ---")
        try:
            self.serial = serial.Serial(self.PORTA_ARDUINO, self.TAXA_BAUD, timeout=1)
            time.sleep(2) # Espera a que Arduino se reinicie
            print(f"Conexión serial establecida con Arduino en {self.PORTA_ARDUINO}.")
            self.enviar_comando(f'S{self.DURACAO_ESTROBE_INICIAL_US}')
            self.enviar_comando(f'P{self.PRE_ATRASO_ESTROBE_INICIAL_US}')
        except serial.SerialException as e:
            print(f"Error: No fue posible establecer conexión serial con Arduino en {self.PORTA_ARDUINO}. {e}")
            if self.captura:
                self.captura.release()
            return False
        return True

    def liberar(self):
        """Libera la cámara, cierra la conexión serial y destruye las ventanas de OpenCV."""
        print("\n--- Realizando limpieza al salir ---")
        if self.captura and self.captura.isOpened():
            self.captura.release()
            print("Cámara liberada (OpenCV).")
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Puerto serial del Arduino cerrado.")
        cv2.destroyAllWindows()
        print("Ventanas del OpenCV cerradas.")
        print("--- Limpieza completa ---")

    def enviar_comando(self, comando):
        """Envia un comando al Arduino via serial y espera la respuesta."""
        if self.serial and self.serial.is_open:
            try:
                # El comando serial debe ser codificado en bytes
                comando_bytes = (comando + '\n').encode('utf-8')
                self.serial.write(comando_bytes)
                # Opcional: imprimir el comando enviado para depuración
                # print(f"Serial: Comando enviado: {comando}") 
                time.sleep(self.TEMPO_ESPERA_ARDUINO_SEG)
            except Exception as e:
                print(f"Error al enviar comando serial '{comando}': {e}")


# ==============================================================================
# DETECTOR FAIXA (BLACK BAND DETECTOR)
# ==============================================================================
class DetectorFaixa:
    """Clase para detectar la banda negra y clasificar el tipo de pupila."""
    def __init__(self, tamanho_kernel=5, limiar_contraste=25, min_bright_pixel_val=200, min_pixel_count_for_bright_pupil=500):
        self.tamanho_kernel = tamanho_kernel if tamanho_kernel % 2 == 1 else tamanho_kernel + 1
        self.limiar_contraste = limiar_contraste
        # Kernel para convolución (detección de bordes)
        self.kernel = np.array([-1] * (self.tamanho_kernel // 2) + [0] + [1] * (self.tamanho_kernel // 2), dtype=float)
        self.min_bright_pixel_val = min_bright_pixel_val
        self.min_pixel_count_for_bright_pupil = min_pixel_count_for_bright_pupil

    def detectar_faixa_preta(self, frame_imagem):
        """Detecta la posición vertical de la banda negra en el frame."""
        if frame_imagem.ndim != 2:
            frame_imagem = cv2.cvtColor(frame_imagem, cv2.COLOR_BGR2GRAY)
        
        # Ic: Intensidad promedio de cada fila
        Ic = np.mean(frame_imagem.astype(float), axis=1)
        # Convolución para detectar el cambio de contraste (la banda)
        Iband = convolve1d(Ic, self.kernel, mode='constant', cval=255.0)
        
        if len(Iband) < 2 or np.ptp(Iband) < self.limiar_contraste:
            return None, None, None
            
        Dmax_idx = np.argmax(Iband)
        Dmin_idx = np.argmin(Iband)
        
        if Dmin_idx > Dmax_idx:
            Dmin_idx, Dmax_idx = Dmax_idx, Dmin_idx
            
        centro_faixa_D = (Dmax_idx + Dmin_idx) * 0.5
        
        # Agregamos una verificación de bordes para evitar detecciones falsas
        margem_borda = max(10, self.tamanho_kernel // 2 + 5)
        intensidade_media_imagem = np.mean(frame_imagem)
        LIMIAR_IMAGEM_BRILHANTE = 180
        esta_nas_bordas_extremas = (Dmin_idx <= margem_borda or Dmax_idx >= len(Ic) - 1 - margem_borda)
        if esta_nas_bordas_extremas and intensidade_media_imagem > LIMIAR_IMAGEM_BRILHANTE:
            return None, None, None
            
        return centro_faixa_D, Dmax_idx, Dmin_idx

    def identificar_tipo_pupila(self, img_gray):
        """Clasifica la pupila como brillante o oscura basándose en el histograma."""
        if img_gray is None:
            return 'desconocido', {}
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        bright_pixels_in_range_count = np.sum(hist[self.min_bright_pixel_val : 256])
        metricas = {'bright_pixel_count': int(bright_pixels_in_range_count)}
        if bright_pixels_in_range_count >= self.min_pixel_count_for_bright_pupil:
            return 'brilhante', metricas
        else:
            return 'escura', metricas

# ==============================================================================
# FUNCIONES AUXILIARES (DRAWING & MATH)
# ==============================================================================

def get_grid_centers(cols, rows, width, height):
    """Calcula las coordenadas centrales de una cuadrícula de NxM."""
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
    """Dibuja una cuadrícula en la imagen."""
    cell_w = width / cols
    cell_h = height / rows
    for i in range(1, cols):
        x = int(i * cell_w)
        cv2.line(image, (x, 0), (x, height), GRID_COLOR, 1)
    for i in range(1, rows):
        y = int(i * cell_h)
        cv2.line(image, (0, y), (width, y), GRID_COLOR, 1)

def draw_cross(image, center, color, size):
    """Dibuja una cruz en la posición central especificada."""
    x, y = int(center[0]), int(center[1])
    cv2.line(image, (x - size//2, y), (x + size//2, y), color, 2)
    cv2.line(image, (x, y - size//2), (x, y + size//2), color, 2)

def detect_pupil_center(img_gray):
    """Detecta el centro de la pupila en la imagen en escala de grises usando contornos."""
    # Umbralización para aislar la pupila (oscura)
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
    # Operación de apertura para eliminar ruido
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
        
    # Encuentra el contorno más grande (asumiendo que es la pupila)
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50: # Evita detecciones de ruido muy pequeño
        return None
        
    # Calcula los momentos para encontrar el centroide (centro de masa)
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

def fit_polynomial_2nd_order(pupil_points, screen_points):
    """Ajusta un mapeo polinomial de 2do orden (pupila -> pantalla)."""
    N = len(pupil_points)
    if N < 6:
        print(f"Advertencia: Se necesitan al menos 6 puntos para el ajuste de 2do orden. Solo hay {N}.")
        # Retorna coeficientes cero si no hay suficientes puntos
        return np.zeros(6), np.zeros(6)

    # La matriz A representa los términos del polinomio: [x^2, x*y, y^2, x, y, 1]
    A = np.zeros((N, 6))
    Bx = np.zeros(N) # Coordenadas X de la pantalla (Target)
    By = np.zeros(N) # Coordenadas Y de la pantalla (Target)
    
    for i, ((x, y), (X, Y)) in enumerate(zip(pupil_points, screen_points)):
        A[i] = [x**2, x*y, y**2, x, y, 1]
        Bx[i] = X
        By[i] = Y
        
    # Resuelve el sistema de ecuaciones usando mínimos cuadrados
    coeffs_x, _, _, _ = np.linalg.lstsq(A, Bx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, By, rcond=None)
    return coeffs_x, coeffs_y

def map_pupil_to_screen(x, y, coeffs_x, coeffs_y):
    """Aplica los coeficientes del polinomio para predecir la posición en pantalla."""
    args = np.array([x**2, x*y, y**2, x, y, 1])
    # Producto punto entre los coeficientes y los términos polinomiales
    X = np.dot(coeffs_x, args)
    Y = np.dot(coeffs_y, args)
    return (X, Y)

def wait_for_enter(window_name, message, width, height):
    """Muestra un mensaje en pantalla completa y espera la tecla ENTER."""
    # Asegura que la ventana sea FullScreen para los estímulos
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        img_wait = np.zeros((height, width, 3), dtype=np.uint8)
        text_size = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height // 2
        
        cv2.putText(img_wait, message, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(img_wait, "Presiona ENTER para continuar.", (text_x-150, text_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow(window_name, img_wait)

        # Usar waitKey(10) para un manejo más robusto de la tecla ENTER
        key = cv2.waitKey(10) 
        # La tecla ENTER es ASCII 13
        if key == 13: 
            cv2.destroyWindow(window_name)
            break
        if key == 27: # ESC
            exit()
            
# ==============================================================================
# FLUJO PRINCIPAL (MAIN EXECUTION)
# ==============================================================================
def main():
    
    # === Inicialización ===
    gerenciador = GerenciadorDispositivos(fps_alvo=FPS, duracao_estrobe=8000, pre_atraso=4000)
    if not gerenciador.iniciar_camera_e_serial():
        print("Error al iniciar dispositivos. Saliendo.")
        return
    atexit.register(gerenciador.liberar)
    detector_faixa = DetectorFaixa()

    os.makedirs("output_gaze", exist_ok=True)
    raw_video_path = os.path.join("output_gaze", "video_ojos.avi")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # El grabador de video usa la resolución de la cámara (640x480)
    out_video = cv2.VideoWriter(raw_video_path, fourcc, FPS, (gerenciador.largura_frame, gerenciador.altura_frame))

    # ======= SINCRONIZACION DE FAIXA PRETA (Estabilidad) =======
    print("--- Fase de sincronización de banda negra (Estabilidad) ---")
    periodo = 16666  # Período inicial para 60fps
    gerenciador.enviar_comando(f'M{periodo}')
    banda_estavel = False
    historico = []
    skip_movement_sync = False 

    # --- INICIO: CONFIGURACIÓN DE VENTANA MINIMALISTA SIN DECORACIONES ---
    window_sync_name = "Sincronizacion Faixa"
    # Usamos WINDOW_AUTOSIZE para la ventana minimalista
    cv2.namedWindow(window_sync_name, cv2.WINDOW_AUTOSIZE)
    # --- FIN: CONFIGURACIÓN DE VENTANA MINIMALISTA SIN DECORACIONES ---

    while not banda_estavel:
        ret, frame = gerenciador.captura.read()
        if not ret:
            # Intentar reabrir si falla la lectura (manejo de errores de cámara)
            if not gerenciador.captura.isOpened():
                gerenciador.iniciar_camera_e_serial() # Reintentar inicializar
            continue
            
        centro_faixa, Dmax, Dmin = detector_faixa.detectar_faixa_preta(frame)
        
        if centro_faixa is not None:
            historico.append(centro_faixa)
            if len(historico) > 20:
                historico.pop(0)
            
            # Condición de estabilidad
            if len(historico) == 20 and np.std(historico) < 1.5:
                banda_estavel = True
                print("Banda negra estabilizada automáticamente.")
        
        # Muestra el frame de la cámara (con info de debug y la instrucción de ENTER)
        frame_sync = frame.copy()
        
        # Agrega las instrucciones en el feed de la cámara
        std_dev = np.std(historico) if len(historico) > 1 else 0
        cv2.putText(frame_sync, f"Estabilidad Banda: {std_dev:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame_sync, "Presiona ENTER para pasar a Calibracion.", (20, gerenciador.altura_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
        
        cv2.imshow(window_sync_name, frame_sync)
        
        # USO DE TECLAS (CORRECCIÓN IMPORTANTE)
        key = cv2.waitKey(10)
        
        if key == 13: # ASCII de la tecla ENTER
            banda_estavel = True
            skip_movement_sync = True # <-- Se salta la fase de Movimiento
            print("Paso forzado por usuario (ENTER). Saltando Sincronizacion de Movimiento.")
            # Cierre la ventana de forma explícita ANTES de salir del bucle
            cv2.destroyAllWindows() 
            
        if key == 27: # Tecla ESC
            gerenciador.liberar()
            return
            
    # Si la estabilidad es automática, la ventana se cerrará aquí
    if not skip_movement_sync:
        cv2.destroyAllWindows() 

    # ======= SINCRONIZACION DE FAIXA PRETA (Movimiento) =======
    # Este bloque se ejecuta solo si la estabilidad automática fue detectada (skip_movement_sync es False)
    if not skip_movement_sync: 
        print("Moviendo banda negra fuera de la región de interés para sincronizar...")
        gerenciador.enviar_comando(f'M{periodo-500}')  # Aplica sesgo (bias)
        banda_desaparecida = False
        desaparecida_count = 0
        
        # --- INICIO: CONFIGURACIÓN DE VENTANA MINIMALISTA SIN DECORACIONES ---
        window_final_name = "Sincronizacion Final"
        # Usamos WINDOW_AUTOSIZE para la ventana final también
        cv2.namedWindow(window_final_name, cv2.WINDOW_AUTOSIZE)
        # --- FIN: CONFIGURACIÓN DE VENTANA MINIMALISTA SIN DECORACIONES ---

        while not banda_desaparecida:
            ret, frame = gerenciador.captura.read()
            if not ret:
                continue
                
            centro_faixa, _, _ = detector_faixa.detectar_faixa_preta(frame)
            
            if centro_faixa is None:
                desaparecida_count += 1
            else:
                desaparecida_count = 0
                
            # Muestra el frame de la cámara para que el usuario monitoree
            frame_sync_final = frame.copy()
            cv2.putText(frame_sync_final, f"Cuenta de Desaparicion: {desaparecida_count}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            cv2.putText(frame_sync_final, "Esperando que la banda desaparezca...", (20, gerenciador.altura_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.imshow(window_final_name, frame_sync_final)
            
            if desaparecida_count >= 10:
                banda_desaparecida = True
                print("Banda negra desaparecida. Sistema sincronizado.")
                
            if cv2.waitKey(10) & 0xFF == 27:
                gerenciador.liberar()
                return
                
        cv2.destroyAllWindows() # Cierra la ventana de la cámara antes de la calibración

    # ======= CALIBRACIÓN (3x3) - INICIA EN PANTALLA COMPLETA =======
    print("Fase de calibración: mira cada objetivo cuando aparece en pantalla.")
    GRID_COLS_CALIB = 3
    GRID_ROWS_CALIB = 3
    # Usamos la resolución de pantalla deseada para calcular los centros de la cuadrícula
    grid_centers = get_grid_centers(GRID_COLS_CALIB, GRID_ROWS_CALIB, SCREEN_WIDTH, SCREEN_HEIGHT)
    pupil_calib_points = []
    screen_calib_points = []
    frame_gaze_data = [] # Datos de pupila para guardar
    frame_alvo_data = [] # Datos de objetivo para guardar

    alvo_idx = 0
    alvo_start_time = time.time()
    calib_phase = True

    # Ventana de Calibración: FULLSCREEN
    cv2.namedWindow("Calibracion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while calib_phase:
        ret, frame = gerenciador.captura.read()
        if not ret:
            continue
            
        ts = time.time()
        out_video.write(frame)
        
        # Prepara la imagen para mostrar (1920x1080) - FONDO NEGRO PURO
        img_show = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_grid(img_show, GRID_COLS_CALIB, GRID_ROWS_CALIB, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        current_alvo_center = grid_centers[alvo_idx]
        draw_cross(img_show, current_alvo_center, CROSS_COLOR, CROSS_SIZE)
        
        cv2.putText(img_show, f"Calibracion: Mira el alvo {alvo_idx+1}/{len(grid_centers)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.imshow("Calibracion", img_show)

        # Procesa el frame de la cámara
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_eye = gray[0:h, w//2:w] # ROI: Mitad derecha del frame para el ojo
        
        # Manejo de detección de pupila. Solo procedemos si el centro no es None
        center_pupil = detect_pupil_center(roi_eye)
        tipo_pupila, _ = detector_faixa.identificar_tipo_pupila(roi_eye)
        
        pupil_coords_global = None
        if center_pupil:
             # Coordenadas globales (en todo el frame)
            pupil_coords_global = (center_pupil[0]+w//2, center_pupil[1]) 

        # Guarda los datos de pupila (coordenadas relativas a la cámara)
        frame_gaze_data.append({
            'timestamp': ts,
            'center_pupil': pupil_coords_global,
            'tipo_pupila': tipo_pupila
        })
        frame_alvo_data.append({
            'timestamp': ts,
            'alvo_idx': alvo_idx,
            'alvo_coords': current_alvo_center
        })

        if time.time() - alvo_start_time >= ALVO_SHOW_TIME:
            if pupil_coords_global:
                pupil_calib_points.append(pupil_coords_global)
                screen_calib_points.append(current_alvo_center)
            
            alvo_idx += 1
            alvo_start_time = time.time()
            if alvo_idx >= len(grid_centers):
                calib_phase = False

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

    # --- Espera manual antes de la predicción ---
    # Usa la función wait_for_enter, que maneja su propia ventana FULLSCREEN
    wait_for_enter(
        "Pausa",
        "Calibracion Completa. Presiona ENTER para iniciar la Prediccion.",
        SCREEN_WIDTH,
        SCREEN_HEIGHT
    )

    # ==== FASE DE PREDICCIÓN (7x5) - INICIA EN PANTALLA COMPLETA ====
    print("Fase de predicción iniciada.")
    GRID_COLS_PRED = 7
    GRID_ROWS_PRED = 5
    grid_pred_centers = get_grid_centers(GRID_COLS_PRED, GRID_ROWS_PRED, SCREEN_WIDTH, SCREEN_HEIGHT)
    random_pred_order = list(range(len(grid_pred_centers)))
    random.shuffle(random_pred_order)
    pred_idx = 0
    alvo_pred_start_time = time.time()
    pred_phase = True

    # Ventana de Predicción: FULLSCREEN
    cv2.namedWindow("Prediccion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Prediccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    pred_frame_gaze_data = [] # Datos de pupila y predicción para guardar
    frame_pred_alvo_data = []  # Datos de objetivo de predicción para guardar

    while pred_phase:
        ret, frame = gerenciador.captura.read()
        if not ret:
            break
            
        ts = time.time()
        out_video.write(frame)
        
        # Prepara la imagen para mostrar (1920x1080) - FONDO NEGRO PURO
        img_show = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_grid(img_show, GRID_COLS_PRED, GRID_ROWS_PRED, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        # Dibuja el objetivo actual
        alvo_pos = grid_pred_centers[random_pred_order[pred_idx]]
        draw_cross(img_show, alvo_pos, CROSS_COLOR, CROSS_SIZE)
        cv2.putText(img_show, f"Prediccion: Mira el alvo {pred_idx+1}/{len(grid_pred_centers)}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        
        # Procesa el frame de la cámara
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        roi_eye = gray[0:h, w//2:w]
        center_pupil = detect_pupil_center(roi_eye)
        tipo_pupila, _ = detector_faixa.identificar_tipo_pupila(roi_eye)
        
        if center_pupil:
            cx_global = center_pupil[0] + w//2
            cy_global = center_pupil[1]
            
            # Mapea las coordenadas de la pupila a la pantalla
            pred_X, pred_Y = map_pupil_to_screen(cx_global, cy_global, coeffs_x, coeffs_y)
            
            # Dibuja la predicción en pantalla (cruz roja)
            draw_cross(img_show, (pred_X, pred_Y), (0,0,255), CROSS_SIZE//2)
            cv2.putText(img_show, f"Prediccion: ({pred_X:.0f},{pred_Y:.0f})", (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            
            pred_frame_gaze_data.append({
                'timestamp': ts,
                'center_pupil': (cx_global, cy_global),
                'pred_screen': (pred_X, pred_Y),
                'tipo_pupila': tipo_pupila
            })
            
        cv2.imshow("Prediccion", img_show)
        
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

    # ==== Finalización y Guardado ====
    gerenciador.liberar()
    out_video.release()
    cv2.destroyAllWindows()

    # --- Guarda datos en archivos ---
    gaze_data_path = os.path.join("output_gaze", "calib_gaze_data.txt")
    alvo_data_path = os.path.join("output_gaze", "calib_alvo_data.txt")
    pred_gaze_data_path = os.path.join("output_gaze", "pred_gaze_data.txt")
    pred_alvo_data_path = os.path.join("output_gaze", "pred_alvo_data.txt")

    with open(gaze_data_path, 'w') as f:
        for d in frame_gaze_data:
            f.write(f"{d}\n")
    with open(alvo_data_path, 'w') as f:
        for d in frame_alvo_data:
            f.write(f"{d}\n")
    with open(pred_gaze_data_path, 'w') as f:
        for d in pred_frame_gaze_data:
            f.write(f"{d}\n")
    with open(pred_alvo_data_path, 'w') as f:
        for d in frame_pred_alvo_data:
            f.write(f"{d}\n")
            
    print("Datos guardados en carpeta output_gaze")

if __name__ == "__main__":
    main()
