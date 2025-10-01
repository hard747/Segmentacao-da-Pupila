# ==============================================================================
# IMPORTACIONES DE BIBLIOTECAS NECESARIAS
# ==============================================================================
import cv2
import numpy as np
import time
import random
import os
import serial
import pyudev
import atexit
import csv
import sys # Añadido para manejo seguro de excepciones en liberacion
from scipy.ndimage import convolve1d
from scipy.optimize import least_squares
from sklearn.cluster import KMeans

# ==============================================================================
# CONSTANTES GLOBALES Y FISICAS
# ==============================================================================
CROSS_COLOR = (0, 255, 0)
CROSS_SIZE = 40
GRID_COLOR = (255, 255, 255)
ALVO_SHOW_TIME = 2.0  # Tiempo que se muestra cada objetivo en segundos
FPS = 60
SCREEN_WIDTH = 1920  # Ancho de pantalla deseado para estímulos (píxeles)
SCREEN_HEIGHT = 1080 # Alto de pantalla deseado para estímulos (píxeles)
CAMERA_WIDTH = 640   # Resolución forzada para el feed de la cámara
CAMERA_HEIGHT = 480  # Resolución forzada para el feed de la cámara

# CONSTANTES FISICAS DEL MONITOR (para cálculo de Error Angular)
MONITOR_WIDTH_CM = 51.5
MONITOR_HEIGHT_CM = 29.0
MONITOR_DISTANCE_CM = 40.0 # Distancia Ojo-Pantalla

# Conversion de Píxeles a CM
CM_PER_PIXEL_X = MONITOR_WIDTH_CM / SCREEN_WIDTH
CM_PER_PIXEL_Y = MONITOR_HEIGHT_CM / SCREEN_HEIGHT

# Estados para la sincronización de la banda
STATUS_SEEKING = 0      # Buscando estabilidad (STD > 1.5)
STATUS_STABLE = 1       # Estabilidad encontrada (STD < 1.5), esperando ENTER
STATUS_FINISHED = 2     # Fase terminada

# CONSTANTES DE PROCESAMIENTO PUPILA/GLINT
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL = 0.8
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL = 0.5
GLINT_MIN_AREA = 5
GLINT_MAX_AREA = 100
GLINT_CIRCULARITY_THRESHOLD = 0.6
MIN_BRIGHT_PIXEL_VAL = 200 # Umbral de brillo para clasificar el tipo de pupila
MIN_PIXEL_COUNT_FOR_BRIGHT_PUPIL = 500 # Conteo mínimo de píxeles brillantes
EYE_ROI_OFFSET_X = CAMERA_WIDTH // 2 # Asume que el ojo está en la mitad derecha del frame (320px)

# ==============================================================================
# GERENCIADOR DE DISPOSITIVOS (HARDWARE SETUP)
# ==============================================================================
class GerenciadorDispositivos:
    """Clase para inicializar y gestionar la cámara PS3 Eye y la comunicación serial con Arduino."""
    # IDs de hardware de la PS3 Eye
    PS3_EYE_VENDOR_ID = '1415'
    PS3_EYE_PRODUCT_ID = '2000' # Anteriormente 'MODEL_ID', renombrado a 'PRODUCT_ID' para coincidir con pyudev

    def __init__(self, fps_alvo, duracao_estrobe, pre_atraso):
        self.captura = None
        self.serial = None
        self.largura_frame = CAMERA_WIDTH
        self.altura_frame = CAMERA_HEIGHT
        self.FPS_ALVO = fps_alvo
        self.DURACAO_ESTROBE_INICIAL_US = duracao_estrobe
        self.PRE_ATRASO_ESTROBE_INICIAL_US = pre_atraso

    def obter_id_camera_ps3_eye(self):
        """Busca el ID de la cámara PS3 Eye usando pyudev o intenta IDs por defecto."""
        # 1. Intenta encontrar la cámara usando pyudev
        try:
            # Crea un contexto udev
            context = pyudev.Context()
            
            # Itera sobre dispositivos USB
            for device in context.list_devices(subsystem='video4linux'):
                if 'ID_VENDOR_ID' in device.properties and 'ID_MODEL_ID' in device.properties:
                    vendor_id = device.properties['ID_VENDOR_ID']
                    product_id = device.properties['ID_MODEL_ID']
                    
                    if vendor_id == self.PS3_EYE_VENDOR_ID and product_id == self.PS3_EYE_PRODUCT_ID:
                        # La propiedad DEVNAME contiene la ruta del dispositivo, p.ej., /dev/video0
                        dev_path = device.device_node
                        # Extrae el número del ID (0, 1, 2...)
                        if dev_path and dev_path.startswith('/dev/video'):
                            camera_id = int(dev_path.replace('/dev/video', ''))
                            print(f"PS3 Eye encontrada via pyudev en ID: {camera_id}")
                            return camera_id
        except ImportError:
            print("pyudev no está instalado o no es accesible. Intentando IDs de cámara 0 a 9.")
        except Exception as e:
            print(f"Error al usar pyudev: {e}. Intentando IDs de cámara 0 a 9.")
            
        # 2. Si falla pyudev, intenta iterar sobre IDs de cámara de 0 a 9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Puedes intentar leer un frame para confirmar que no es una cámara virtual
                ret, frame = cap.read()
                if ret:
                    cap.release()
                    print(f"Cámara abierta exitosamente en ID: {i}. Usando esta cámara.")
                    return i
                cap.release()

        print("Error: PS3 Eye no encontrada o ninguna cámara disponible.")
        return None

    def iniciar_camera_e_serial(self):
        """Inicializa la cámara y la conexión serial con Arduino, forzando la resolución."""
        print("--- Inicializando Câmera PS3 Eye ---")
        id_camera = self.obter_id_camera_ps3_eye()
        if id_camera is None:
            return False

        self.captura = cv2.VideoCapture(id_camera)
        if not self.captura.isOpened():
            print(f"Error: No fue posible abrir la cámara con OpenCV (ID: {id_camera}).")
            return False
            
        # Establece propiedades específicas para la PS3 Eye si es posible
        self.captura.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.captura.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        self.captura.set(cv2.CAP_PROP_FPS, self.FPS_ALVO)
        time.sleep(1)
        
        self.largura_frame = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.altura_frame = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Câmera abierta con OpenCV (ID: {id_camera}).")
        print(f"Resolución de captura de la cámara: {self.largura_frame}x{self.altura_frame}")

        # Configuración de comunicación Serial (Comentada para evitar errores sin Arduino físico)
        # print("\n--- Inicializando conexión Serial con Arduino ---")
        # try:
        #     self.serial = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
        #     time.sleep(2) 
        #     print(f"Conexión serial establecida.")
        #     self.enviar_comando(f'S{self.DURACAO_ESTROBE_INICIAL_US}')
        #     self.enviar_comando(f'P{self.PRE_ATRASO_ESTROBE_INICIAL_US}')
        # except serial.SerialException as e:
        #     print(f"Error: No fue posible establecer conexión serial. Corriendo sin comandos seriales. {e}")
        #     if self.captura:
        #         self.captura.release()
        #     return False
        return True

    def liberar(self):
        """
        Realiza la limpieza de recursos (cámara, ventanas, etc.).
        Se asegura de liberar la cámara de forma segura y maneja el error de puntero nulo de OpenCV.
        """
        print("\n--- Realizando limpieza al salir ---")
        if self.captura and self.captura.isOpened():
            self.captura.release()
            print("Cámara liberada (OpenCV).")
        
        # if self.serial and self.serial.is_open:
        #     print("Puerto serial del Arduino cerrado.")

        # --- Limpieza de Ventanas (Se añade try-except para el error de 'Null pointer') ---
        try:
            # Comprobación original: Verifica si alguna ventana es visible o existe.
            if cv2.getWindowProperty("Calibracion", cv2.WND_PROP_VISIBLE) >= 0 or \
               cv2.getWindowProperty("Prediccion", cv2.WND_PROP_VISIBLE) >= 0 or \
               cv2.getWindowProperty("Sincronizacion Faixa", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyAllWindows()
                print("Ventanas del OpenCV cerradas.")
            
        except cv2.error as e:
            # Captura el error de OpenCV (-27:Null pointer) que ocurre cuando se consulta una propiedad 
            # de una ventana que ya ha sido cerrada/destruida por el sistema o por otro método.
            if "Null pointer" in str(e):
                cv2.destroyAllWindows()
                print("Ventanas del OpenCV cerradas.")
            else:
                # Si es otro error de OpenCV, lo relanzamos.
                raise
        except Exception as e:
            # Captura cualquier otra excepción que pueda ocurrir durante el cierre.
            print(f"Error inesperado durante la liberación de ventanas: {e}", file=sys.stderr)

        print("--- Limpieza completa ---")

    def enviar_comando(self, comando):
        """Envia un comando al Arduino via serial y espera la respuesta."""
        if self.serial and self.serial.is_open:
            try:
                comando_bytes = (comando + '\n').encode('utf-8')
                # self.serial.write(comando_bytes)
                time.sleep(self.TEMPO_ESPERA_ARDUINO_SEG)
            except Exception as e:
                print(f"Error al enviar comando serial '{comando}': {e}")


# ==============================================================================
# DETECTOR FAIXA (BLACK BAND DETECTOR)
# ==============================================================================
class DetectorFaixa:
    """Clase para detectar la banda negra y clasificar el tipo de pupila."""
    def __init__(self, tamanho_kernel=5, limiar_contraste=25):
        self.tamanho_kernel = tamanho_kernel if tamanho_kernel % 2 == 1 else tamanho_kernel + 1
        self.limiar_contraste = limiar_contraste
        self.kernel = np.array([-1] * (self.tamanho_kernel // 2) + [0] + [1] * (self.tamanho_kernel // 2), dtype=float)
        self.min_bright_pixel_val = MIN_BRIGHT_PIXEL_VAL
        self.min_pixel_count_for_bright_pupil = MIN_PIXEL_COUNT_FOR_BRIGHT_PUPIL

    def detectar_faixa_preta(self, frame_imagem):
        """Detecta la posición vertical de la banda negra en el frame."""
        if frame_imagem.ndim != 2:
            frame_imagem = cv2.cvtColor(frame_imagem, cv2.COLOR_BGR2GRAY)
        
        Ic = np.mean(frame_imagem.astype(float), axis=1)
        Iband = convolve1d(Ic, self.kernel, mode='constant', cval=255.0)
        
        if len(Iband) < 2 or np.ptp(Iband) < self.limiar_contraste:
            return None, None, None
            
        Dmax_idx = np.argmax(Iband)
        Dmin_idx = np.argmin(Iband)
        
        if Dmin_idx > Dmax_idx:
            Dmin_idx, Dmax_idx = Dmax_idx, Dmin_idx
            
        centro_faixa_D = (Dmax_idx + Dmin_idx) * 0.5
        
        margem_borda = max(10, self.tamanho_kernel // 2 + 5)
        intensidade_media_imagem = np.mean(frame_imagem)
        LIMIAR_IMAGEM_BRILHANTE = 180
        esta_nas_bordas_extremas = (Dmin_idx <= margem_borda or Dmax_idx >= len(Ic) - 1 - margem_borda)
        
        if esta_nas_bordas_extremas and intensidade_media_imagem > LIMIAR_IMAGEM_BRILHANTE:
            return None, None, None
            
        return centro_faixa_D, Dmax_idx, Dmin_idx

    def identificar_tipo_pupila(self, img_gray):
        """
        Clasifica la pupila como brillante o oscura basándose en la contagem de píxeles
        por encima de un umbral de brillo, utilizando las constantes globales.
        """
        if img_gray is None:
            return 'desconocido', {}
        
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        bright_pixels_in_range_count = np.sum(hist[self.min_bright_pixel_val : 256])

        metricas = {
            'bright_pixel_count': int(bright_pixels_in_range_count)
        }

        if bright_pixels_in_range_count >= self.min_pixel_count_for_bright_pupil:
            return 'brilhante', metricas
        else:
            return 'escura', metricas

# ==============================================================================
# CLASES DE PROCESAMIENTO AVANZADO (PUPILA Y GLINT)
# ==============================================================================

class CircleFitter:
    """Ajusta un círculo a un conjunto de puntos de contorno utilizando mínimos cuadrados no lineales."""
    @staticmethod
    def fit_circle(points):
        """
        Ajusta un círculo utilizando el método de mínimos cuadrados no lineales.
        Retorna (cx, cy, r) o (None, None, None) si falla.
        """
        if points is None or len(points) < 3:
            return None, None, None
        
        if points.ndim == 3:
            points = points.squeeze()
        if points.ndim == 1:
            points = points.reshape(-1, 2)
            
        def residuals(params, x_data, y_data):
            cx, cy, r = params
            return np.sqrt((x_data - cx) ** 2 + (y_data - cy) ** 2) - r
        
        x = points[:, 0]
        y = points[:, 1]
        
        initial_cx = np.mean(x)
        initial_cy = np.mean(y)
        initial_r = np.mean(np.sqrt((x - initial_cx) ** 2 + (y - initial_cy) ** 2))
        
        try:
            result = least_squares(
                residuals,
                [initial_cx, initial_cy, initial_r],
                args=(x, y),
                bounds=([0, 0, 1], np.inf),
            )
            return result.x[0], result.x[1], result.x[2] # cx, cy, r
        except Exception:
            return None, None, None

class GlintDetector:
    """Detecta los reflejos (glints) en la imagen utilizando umbralización y propiedades de contorno."""
    @staticmethod
    def detect_glints(img_gray, roi_mask=None, is_dark_pupil_frame=False, max_glints=None):
        if img_gray is None:
            return []
        
        img_to_process = img_gray.copy()
        if roi_mask is not None:
            img_to_process = cv2.bitwise_and(img_gray, img_gray, mask=roi_mask)

        min_val, max_val, _, _ = cv2.minMaxLoc(img_to_process)
        if max_val < 50:
            return []
            
        threshold_perc = GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL if is_dark_pupil_frame else GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL
        glint_thresh = int(max_val * threshold_perc)
        
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
        
        if max_glints is not None and len(glints) > max_glints:
            glints = sorted(glints, key=lambda x: x['area'], reverse=True)[:max_glints]

        return glints

class PupilSegmenter:
    """Implementa la segmentación de pupila utilizando diferencia de frames y Watershed."""
    def __init__(self):
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)

    def refine_roi(self, escura_gray, brilhante_gray):
        """Calcula la diferencia entre frames y crea una máscara ROI inicial de la pupila."""
        diff = cv2.subtract(brilhante_gray, escura_gray)
        diff_blurred = cv2.GaussianBlur(diff, (9, 9), 0)
        
        _, roi_mask = cv2.threshold(diff_blurred, 40, 255, cv2.THRESH_BINARY)
        
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.kernel_medium, iterations=2)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.kernel_medium, iterations=2)
        roi_mask = cv2.erode(roi_mask, self.kernel_small, iterations=1)
        roi_mask = cv2.dilate(roi_mask, self.kernel_small, iterations=1)
        
        return roi_mask, diff

    def remove_glints_from_image(self, img_gray, glints_info):
        """Rellena el área de los glints con píxeles oscuros para evitar que interfieran con la segmentación."""
        img_no_glints = img_gray.copy()
        for glint in glints_info:
            center = glint['center']
            radius = int(max(2, np.sqrt(glint['area'] / np.pi) * 1.5))
            cv2.circle(img_no_glints, center, radius, 0, -1)
        return img_no_glints

    def segment_pupil_dark(self, escura_gray_no_glints, roi_mask, frame_original):
        """Segmentación de la pupila oscura (fondo brillante) usando Watershed."""
        result_img = frame_original.copy()
        pupil_info_list = []

        _, thresh_inv = cv2.threshold(escura_gray_no_glints, 50, 255, cv2.THRESH_BINARY_INV)
        thresh_inv_roi = cv2.bitwise_and(thresh_inv, thresh_inv, mask=roi_mask)

        opening = cv2.morphologyEx(thresh_inv_roi, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        sure_bg = cv2.dilate(opening, self.kernel_small, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        pupil_mask_escura = np.zeros_like(escura_gray_no_glints)

        if np.max(markers) > 1:
            markers_final = cv2.watershed(frame_original.copy(), markers.copy())
            pupil_mask_escura[markers_final > 1] = 255

            contours, _ = cv2.findContours(pupil_mask_escura, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 50 < area < 6000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        if circularity > 0.35:
                            cx, cy, r = CircleFitter.fit_circle(cnt)
                            if cx is not None:
                                center = (int(cx), int(cy))
                                cv2.drawContours(result_img, [cnt], -1, (255, 0, 0), 2)
                                pupil_info_list.append({
                                    'center': center,
                                    'radius': r,
                                    'fit_circle': (cx, cy, r)
                                })
        return result_img, pupil_mask_escura, pupil_info_list

    def segment_pupil_bright(self, brilhante_gray_no_glints, roi_mask, frame_original):
        """Segmentación de la pupila brillante (fondo oscuro) usando Watershed."""
        result_img = frame_original.copy()
        pupil_info_list = []

        roi_no_glints = cv2.bitwise_and(brilhante_gray_no_glints, brilhante_gray_no_glints, mask=roi_mask)
        
        min_val, max_val, _, _ = cv2.minMaxLoc(roi_no_glints, mask=roi_mask)
        threshold_val = 60
        if max_val > 0:
            threshold_val = int(max_val * 0.5)
            threshold_val = max(50, min(200, threshold_val))

        _, thresh = cv2.threshold(roi_no_glints, threshold_val, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_small, iterations=1)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, self.kernel_small, iterations=1)

        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, self.kernel_small, iterations=2)
        sure_bg = cv2.dilate(opening, self.kernel_small, iterations=3)
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

        dist_max = dist_transform.max()
        if dist_max > 0:
            _, sure_fg = cv2.threshold(dist_transform, 0.4 * dist_max, 255, 0)
        else:
            sure_fg = np.zeros_like(brilhante_gray_no_glints)

        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        pupil_mask_brilhante = np.zeros_like(brilhante_gray_no_glints)

        if np.max(markers) > 1:
            markers_final = cv2.watershed(frame_original.copy(), markers.copy())
            pupil_mask_brilhante[markers_final > 1] = 255

            contours, _ = cv2.findContours(pupil_mask_brilhante, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 40 < area < 7000:
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter ** 2)
                        if circularity > 0.30:
                            cx, cy, r = CircleFitter.fit_circle(cnt)
                            if cx is not None:
                                center = (int(cx), int(cy))
                                cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
                                pupil_info_list.append({
                                    'center': center,
                                    'radius': r,
                                    'fit_circle': (cx, cy, r)
                                })
        return result_img, pupil_mask_brilhante, pupil_info_list

    def segment_pair(self, frame_escura, frame_brilhante, offset_x):
        """Procesa un par de frames oscuro/brillante (que ya son el ROI del ojo) para obtener la segmentación final."""
        if frame_escura is None or frame_brilhante is None:
            return None

        escura_gray = cv2.cvtColor(frame_escura, cv2.COLOR_BGR2GRAY)
        brilhante_gray = cv2.cvtColor(frame_brilhante, cv2.COLOR_BGR2GRAY)
        
        roi_mask, diff = self.refine_roi(escura_gray, brilhante_gray)

        if not np.any(roi_mask):
            return None

        glints_escura = GlintDetector.detect_glints(escura_gray, roi_mask, is_dark_pupil_frame=True, max_glints=2)
        glints_brilhante = GlintDetector.detect_glints(brilhante_gray, roi_mask, is_dark_pupil_frame=False, max_glints=2) # Aumentado a 2 para manejar 2 glints

        escura_no_glints = self.remove_glints_from_image(escura_gray, glints_escura)
        brilhante_no_glints = self.remove_glints_from_image(brilhante_gray, glints_brilhante)

        escura_resultado, pupil_mask_escura, pupil_escura_info = self.segment_pupil_dark(
            escura_no_glints, roi_mask, frame_escura
        )
        brilhante_resultado, pupil_mask_brilhante, pupil_brilhante_info = self.segment_pupil_bright(
            brilhante_no_glints, roi_mask, frame_brilhante
        )
        
        # Coordenadas Globales y Dibujado de Glints
        for info in pupil_escura_info + pupil_brilhante_info:
            info['center_global'] = (info['center'][0] + offset_x, info['center'][1])
            info['fit_circle_global'] = (info['fit_circle'][0] + offset_x, info['fit_circle'][1], info['fit_circle'][2])
            
        for g in glints_escura:
            g['center_global'] = (g['center'][0] + offset_x, g['center'][1])
            cv2.circle(escura_resultado, g['center'], 5, (0, 0, 255), -1)
            
        for g in glints_brilhante:
            g['center_global'] = (g['center'][0] + offset_x, g['center'][1])
            cv2.circle(brilhante_resultado, g['center'], 3, (0, 255, 255), -1)
        
        diff_colored = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        roi_mask_colored = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
        
        # Asegúrate de que todas las imágenes tienen el mismo tamaño para hstack/vstack
        rows, cols = frame_escura.shape[:2]
        diff_colored = cv2.resize(diff_colored, (cols, rows))
        roi_mask_colored = cv2.resize(roi_mask_colored, (cols, rows))
        
        top_row = np.hstack((escura_resultado, brilhante_resultado))
        bottom_row = np.hstack((diff_colored, roi_mask_colored))
        debug_image = np.vstack((top_row, bottom_row))

        return {
            'debug_image': debug_image,
            'pupil_dark_info': pupil_escura_info,
            'pupil_bright_info': pupil_brilhante_info,
            'glints_dark': glints_escura,
            'glints_bright': glints_brilhante,
        }

def cluster_glints_by_eye(glints, n_eyes=1, max_glints_per_eye=2):
    """
    Función de utilidad para agrupar glints (mantida, aunque no se usa directamente en el flujo actual
    donde el ROI ya aísla un ojo).
    """
    if not glints:
        return []
    coords = np.array([g['center'] for g in glints])
    if len(coords) <= n_eyes:
        return glints
        
    kmeans = KMeans(n_clusters=n_eyes, n_init=10, random_state=42)
    xs = coords[:, 0].reshape(-1, 1)
    labels = kmeans.fit_predict(xs)
    
    selected_glints = []
    for i in range(n_eyes):
        cluster_glints = [g for g, l in zip(glints, labels) if l == i]
        cluster_glints = sorted(cluster_glints, key=lambda g: g['area'], reverse=True)
        selected_glints.extend(cluster_glints[:max_glints_per_eye])
        
    return selected_glints

# ==============================================================================
# CALCULO DE ERROR ANGULAR (NUEVA FUNCION)
# ==============================================================================

def calculate_angular_error(pred_screen_coords, target_coords):
    """
    Calcula el error angular (en grados) entre la mirada predicha y el objetivo real.
    
    Asume que el ojo está en el origen (0, 0, 0) y la pantalla está en Z = -MONITOR_DISTANCE_CM.
    
    Args:
        pred_screen_coords (tuple): Coordenadas (X, Y) en píxeles de la mirada predicha.
        target_coords (tuple): Coordenadas (X, Y) en píxeles del alvo real.
        
    Returns:
        float: Error angular en grados.
    """
    if pred_screen_coords is None or target_coords is None:
        return np.nan
        
    # 1. Convertir coordenadas de píxeles a CM (Origen en el centro de la pantalla)
    
    # Centro de la pantalla en píxeles
    center_x_px = SCREEN_WIDTH / 2
    center_y_px = SCREEN_HEIGHT / 2
    
    # Conversión a CM, con origen en el centro de la pantalla
    # El eje X positivo va a la derecha, Y positivo va hacia arriba (OpenCV tiene Y invertido)
    # Z positivo va hacia el ojo
    
    def px_to_cm_coords(px_x, px_y):
        # Coordenada X en CM (del centro a la derecha/izquierda)
        cm_x = (px_x - center_x_px) * CM_PER_PIXEL_X
        # Coordenada Y en CM (del centro hacia arriba/abajo)
        cm_y = (center_y_px - px_y) * CM_PER_PIXEL_Y
        # Coordenada Z es la distancia fija de la pantalla
        cm_z = -MONITOR_DISTANCE_CM 
        return np.array([cm_x, cm_y, cm_z])
        
    # 2. Definir los vectores 3D (Ojo en el origen (0,0,0))
    # Vector Objetivo (V_T)
    V_T = px_to_cm_coords(target_coords[0], target_coords[1])
    
    # Vector Predicho (V_P)
    V_P = px_to_cm_coords(pred_screen_coords[0], pred_screen_coords[1])

    # 3. Aplicar la fórmula del producto escalar para el ángulo (en radianes)
    # cos(theta) = (V_P . V_T) / (|V_P| * |V_T|)
    
    dot_product = np.dot(V_P, V_T)
    magnitude_product = np.linalg.norm(V_P) * np.linalg.norm(V_T)
    
    if magnitude_product == 0:
        return np.nan
        
    cos_theta = np.clip(dot_product / magnitude_product, -1.0, 1.0)
    
    # Ángulo en radianes
    angle_rad = np.arccos(cos_theta)
    
    # Convertir a grados
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# ==============================================================================
# FLUJO PRINCIPAL (MAIN EXECUTION)
# ==============================================================================
def main():
    
    # === Inicialización ===
    gerenciador = GerenciadorDispositivos(fps_alvo=FPS, duracao_estrobe=8000, pre_atraso=4000)
    
    # Registrar la liberación de recursos al final
    atexit.register(gerenciador.liberar)
    
    if not gerenciador.iniciar_camera_e_serial():
        print("Error al iniciar dispositivos. Saliendo.")
        return
        
    detector_faixa = DetectorFaixa()
    segmenter = PupilSegmenter()

    os.makedirs("output_gaze", exist_ok=True)
    raw_video_path = os.path.join("output_gaze", "video_ojos.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out_video = cv2.VideoWriter(raw_video_path, fourcc, FPS, (gerenciador.largura_frame, gerenciador.altura_frame))

    frame_buffer = {'escura': None, 'brilhante': None}
    
    # Variable de control de depuración
    show_debug = False 
    debug_window_name = "Debug Segmentacion"
    
    # ======= SINCRONIZACION DE FAIXA PRETA (Estabilidad) =======
    print("--- Fase de sincronización de banda negra (Estabilidad) ---")
    periodo = 16666
    
    sync_status = STATUS_SEEKING
    historico = []               
    STD_THRESHOLD = 1.5         
    MIN_BUFFER_SIZE = 20        
    window_sync_name = "Sincronizacion Faixa"
    cv2.namedWindow(window_sync_name, cv2.WINDOW_AUTOSIZE)
    
    while sync_status != STATUS_FINISHED:
        ret, frame = gerenciador.captura.read()
        if not ret: continue
        
        centro_faixa, Dmax, Dmin = detector_faixa.detectar_faixa_preta(frame)
        
        if centro_faixa is not None:
            historico.append(centro_faixa)
            if len(historico) > MIN_BUFFER_SIZE: historico.pop(0)
        
        std_dev = np.std(historico) if len(historico) == MIN_BUFFER_SIZE else float('inf')
        
        if sync_status == STATUS_SEEKING:
            if len(historico) == MIN_BUFFER_SIZE and std_dev < STD_THRESHOLD:
                sync_status = STATUS_STABLE
                print(f"Estabilidad de Banda Negra ALCANZADA (STD:{std_dev:.2f}). Esperando confirmación de usuario.")
        
        if sync_status == STATUS_STABLE:
            if std_dev >= STD_THRESHOLD:
                sync_status = STATUS_SEEKING
                print(f"Advertencia: Estabilidad perdida (STD:{std_dev:.2f}). Volviendo a la fase de búsqueda.")
        
        frame_sync = frame.copy()
        if sync_status == STATUS_SEEKING:
            color = (0, 165, 255)
            status_text = f"BUSCANDO ESTABILIDAD... STD: {std_dev:.2f} (Umbral: {STD_THRESHOLD})"
            instruction = "Mueve el LED/Cámara si el STD es alto."
        elif sync_status == STATUS_STABLE:
            color = (0, 255, 0)
            status_text = f"ESTABLE! STD: {std_dev:.2f}. Presiona ENTER para continuar."
            instruction = "SISTEMA LISTO. Presiona ENTER para pasar a CALIBRACION."
            
        cv2.putText(frame_sync, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(frame_sync, instruction, (20, gerenciador.altura_frame - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        cv2.imshow(window_sync_name, frame_sync)
        
        key = cv2.waitKey(10)
        
        if key == 13 and sync_status == STATUS_STABLE:
            sync_status = STATUS_FINISHED
            print("Paso a Calibración confirmado por usuario (ENTER).")
            
        if key == 27: 
            # Si se presiona ESC, salimos del bucle y vamos al procesamiento final
            break
            
    cv2.destroyAllWindows() 
    
    # ======= CALIBRACIÓN (3x3) - INICIA EN PANTALLA COMPLETA =======
    print("Fase de calibración: mira cada objetivo cuando aparece en pantalla.")
    GRID_COLS_CALIB = 3
    GRID_ROWS_CALIB = 3
    grid_centers = get_grid_centers(GRID_COLS_CALIB, GRID_ROWS_CALIB, SCREEN_WIDTH, SCREEN_HEIGHT)
    pupil_calib_points = []
    screen_calib_points = []
    glint_calib_points = []
    
    alvo_idx = 0
    alvo_start_time = time.time()
    calib_phase = True
    
    cv2.namedWindow("Calibracion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibracion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while calib_phase:
        ret, frame = gerenciador.captura.read()
        if not ret: break
            
        ts = time.time()
        out_video.write(frame)
        
        h, w = frame.shape[:2]
        roi_frame = frame[0:h, EYE_ROI_OFFSET_X:w].copy()
        
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        current_frame_type, _ = detector_faixa.identificar_tipo_pupila(roi_gray)
        
        if current_frame_type != 'desconocido':
            frame_buffer[current_frame_type] = roi_frame.copy()
            
        segmentation_results = None
        if frame_buffer['escura'] is not None and frame_buffer['brilhante'] is not None:
            segmentation_results = segmenter.segment_pair(
                frame_buffer['escura'], 
                frame_buffer['brilhante'], 
                offset_x=EYE_ROI_OFFSET_X
            )
            frame_buffer = {'escura': None, 'brilhante': None}
            
        pupil_center_global = None
        glint_center_global = None
        
        if segmentation_results:
            pupil_info = segmentation_results['pupil_dark_info']
            glint_info = segmentation_results['glints_bright']
            
            if pupil_info:
                pupil_center_global = pupil_info[0]['center_global']
            
            if glint_info:
                # Calcula el promedio de las coordenadas de los glints
                glint_coords = np.array([g['center_global'] for g in glint_info])
                # Explicación: Se usa el punto medio de los glints para el vector PCG
                glint_center_global = tuple(np.mean(glint_coords, axis=0).astype(int))

            # Lógica de ventana de debug Opcional
            if show_debug:
                cv2.imshow(debug_window_name, segmentation_results['debug_image'])
            
        # --- Lógica de Estímulo y Muestreo ---
        img_show = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_grid(img_show, GRID_COLS_CALIB, GRID_ROWS_CALIB, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        current_alvo_center = grid_centers[alvo_idx]
        draw_cross(img_show, current_alvo_center, CROSS_COLOR, CROSS_SIZE)
        
        if pupil_center_global:
            cv2.circle(img_show, pupil_center_global, 5, (255, 0, 0), -1)
        if glint_center_global:
            cv2.circle(img_show, glint_center_global, 5, (0, 0, 255), -1)
            
        cv2.putText(img_show, f"Calibracion: Alvo {alvo_idx+1}/{len(grid_centers)}. Presiona 'd' para Debug.", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.imshow("Calibracion", img_show)

        if time.time() - alvo_start_time >= ALVO_SHOW_TIME:
            if pupil_center_global and glint_center_global:
                pupil_calib_points.append(pupil_center_global)
                glint_calib_points.append(glint_center_global)
                screen_calib_points.append(current_alvo_center)
            
            alvo_idx += 1
            alvo_start_time = time.time()
            if alvo_idx >= len(grid_centers):
                calib_phase = False

        key = cv2.waitKey(1)
        if key == ord('d'): # Tecla 'd' para activar/desactivar el debug
            show_debug = not show_debug
            if not show_debug and cv2.getWindowProperty(debug_window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(debug_window_name)
                print("Ventana de Debug CERRADA.")
            elif show_debug:
                print("Ventana de Debug ABIERTA.")
        if key == 27: 
            # Si se presiona ESC, salimos del bucle y vamos al procesamiento final
            break

    cv2.destroyAllWindows()
    print("Calibracion terminada.")
    
    # --- Ajusta polinomios de mapeo ---
    if len(pupil_calib_points) != len(glint_calib_points):
        print("Error: Número de puntos Pupila y Glint no coinciden.")
        return

    pcg_calib_points = [
        (p[0] - g[0], p[1] - g[1]) 
        for p, g in zip(pupil_calib_points, glint_calib_points)
    ]
    
    print("Ajustando mapeo polinomial de 2do grado (usando vector PCG)...")
    coeffs_x, coeffs_y = fit_polynomial_2nd_order(pcg_calib_points, screen_calib_points)
    
    # IMPRESIÓN DE COEFICIENTES (Añadido)
    print("\n--- Coeficientes Polinomiales de Mapeo (PCG -> Pantalla) ---")
    print(f"Coeficientes X (Mapeo de la Abscisa de Pantalla): {coeffs_x}")
    print(f"Coeficientes Y (Mapeo de la Ordenada de Pantalla): {coeffs_y}")
    print("----------------------------------------------------------\n")

    wait_for_enter(
        "Pausa Pre-Prediccion",
        "Calibracion Completa. Presiona ENTER para iniciar la PREDICCION (Fullscreen).",
        SCREEN_WIDTH,
        SCREEN_HEIGHT,
        is_fullscreen=True
    )

    # ==== FASE DE PREDICCIÓN (7x5) - INICIA EN PANTALLA COMPLETA ====
    print("Fase de predicción iniciada.")
    GRID_COLS_PRED = 7
    GRID_ROWS_PRED = 5
    grid_pred_centers = get_grid_centers(GRID_COLS_PRED, GRID_ROWS_PRED, SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Crea un orden de alvos secuencial (0 a 34, de arriba a abajo, izquierda a derecha)
    target_ids = list(range(len(grid_pred_centers)))
    
    alvo_idx_sequential = 0
    alvo_start_time = time.time()
    pred_phase = True
    
    cv2.namedWindow("Prediccion", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Prediccion", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Asegúrate de que la ventana de debug esté cerrada antes de la fase de predicción
    if cv2.getWindowProperty(debug_window_name, cv2.WND_PROP_VISIBLE) >= 1:
        cv2.destroyWindow(debug_window_name)
        show_debug = False # Reinicia el flag por si acaso

    pred_frame_gaze_data = [] 

    while pred_phase:
        ret, frame = gerenciador.captura.read()
        if not ret: break
            
        ts = time.time()
        out_video.write(frame)
        
        h, w = frame.shape[:2]
        roi_frame = frame[0:h, EYE_ROI_OFFSET_X:w].copy()
        
        roi_gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        current_frame_type, _ = detector_faixa.identificar_tipo_pupila(roi_gray)
        
        if current_frame_type != 'desconocido':
            frame_buffer[current_frame_type] = roi_frame.copy()
            
        segmentation_results = None
        if frame_buffer['escura'] is not None and frame_buffer['brilhante'] is not None:
            segmentation_results = segmenter.segment_pair(
                frame_buffer['escura'], 
                frame_buffer['brilhante'], 
                offset_x=EYE_ROI_OFFSET_X
            )
            frame_buffer = {'escura': None, 'brilhante': None}
        
        pupil_center_global = None
        glint_center_global = None
        pcg_vector = None
        pred_screen = None
        angular_error = np.nan
        
        # Alvo actual
        current_alvo_id = alvo_idx_sequential 
        alvo_pos = grid_pred_centers[current_alvo_id]
        
        if segmentation_results:
            pupil_info = segmentation_results['pupil_dark_info']
            glint_info = segmentation_results['glints_bright']
            
            if pupil_info: pupil_center_global = pupil_info[0]['center_global']
            
            if glint_info:
                glint_coords = np.array([g['center_global'] for g in glint_info])
                glint_center_global = tuple(np.mean(glint_coords, axis=0).astype(int))

            if show_debug:
                cv2.imshow(debug_window_name, segmentation_results['debug_image'])

            if pupil_center_global and glint_center_global:
                pcg_vector = (pupil_center_global[0] - glint_center_global[0], 
                              pupil_center_global[1] - glint_center_global[1])
                
                pred_X, pred_Y = map_pupil_to_screen(
                    pcg_vector[0], 
                    pcg_vector[1], 
                    coeffs_x, 
                    coeffs_y
                )
                pred_screen = (int(pred_X), int(pred_Y))
                
                # Calcular el Error Angular
                angular_error = calculate_angular_error(pred_screen, alvo_pos)


        # --- Guardado de datos por frame ---
        pred_frame_gaze_data.append({
            'timestamp': ts,
            'target_id': current_alvo_id,
            'pupil_global': pupil_center_global,
            'glint_global': glint_center_global,
            'pcg_vector': pcg_vector,
            'pred_screen': pred_screen,
            'alvo_coords': alvo_pos,
            'angular_error_deg': angular_error,
            # Asegura que la clave existe, valor inicializado a nan
            'avg_angular_error_target_deg': np.nan 
        })


        # --- Lógica de Estímulo y Muestreo ---
        img_show = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        draw_grid(img_show, GRID_COLS_PRED, GRID_ROWS_PRED, SCREEN_WIDTH, SCREEN_HEIGHT)
        
        draw_cross(img_show, alvo_pos, CROSS_COLOR, CROSS_SIZE)
        
        if pred_screen:
            draw_cross(img_show, pred_screen, (0,0,255), CROSS_SIZE//2)
            error_text = f"Error Angular: {angular_error:.2f} deg" if not np.isnan(angular_error) else "Error Angular: N/A"
            cv2.putText(img_show, error_text, (30,100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            
        cv2.putText(img_show, f"Prediccion: Alvo ID {current_alvo_id} ({alvo_idx_sequential+1}/{len(grid_pred_centers)}).", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 2)
        cv2.imshow("Prediccion", img_show)
        
        if time.time() - alvo_start_time >= ALVO_SHOW_TIME:
            alvo_idx_sequential += 1
            alvo_start_time = time.time()
            if alvo_idx_sequential >= len(grid_pred_centers):
                pred_phase = False

        key = cv2.waitKey(1)
        if key == ord('d'): # Tecla 'd' para activar/desactivar el debug
            show_debug = not show_debug
            if not show_debug and cv2.getWindowProperty(debug_window_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow(debug_window_name)
                print("Ventana de Debug CERRADA.")
            elif show_debug:
                print("Ventana de Debug ABIERTA.")
        if key == 27: 
            # Si se presiona ESC, salimos del bucle y vamos al procesamiento final
            break

    # ==== Finalización y Procesamiento de Datos ====
    out_video.release()
    cv2.destroyAllWindows()

    # --- Procesamiento y Agregación de Datos ---
    
    # Estructura para almacenar promedios por alvo
    target_data_map = {} # {target_id: [error1, error2, ...]}

    for d in pred_frame_gaze_data:
        target_id = d['target_id']
        error = d['angular_error_deg']
        
        # Agrupa los errores angulares válidos por ID de alvo
        if not np.isnan(error):
            if target_id not in target_data_map:
                target_data_map[target_id] = []
            target_data_map[target_id].append(error)

    # Cálculo del promedio por alvo y asignación a CADA frame
    for d in pred_frame_gaze_data:
        target_id = d['target_id']
        errors = target_data_map.get(target_id, [])
        d['avg_angular_error_target_deg'] = np.mean(errors) if errors else np.nan
        
    # --- Guarda datos en archivo CSV (más robusto) ---
    pred_gaze_data_path = os.path.join("output_gaze", "pred_gaze_data.csv")

    header = [
        "timestamp_frame", 
        "target_id",
        "pcg_x", "pcg_y", 
        "alvo_x", "alvo_y", 
        "pred_x", "pred_y", 
        "angular_error_frame_deg", 
        "avg_angular_error_target_deg"
    ]
    
    try:
        with open(pred_gaze_data_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            
            for d in pred_frame_gaze_data:
                # Utilizamos .get para mayor seguridad
                pcg_vector = d.get('pcg_vector')
                pred_screen = d.get('pred_screen')
                alvo_coords = d.get('alvo_coords')
                
                pcg_x = f"{pcg_vector[0]:.2f}" if pcg_vector and not np.isnan(pcg_vector[0]) else 'None'
                pcg_y = f"{pcg_vector[1]:.2f}" if pcg_vector and not np.isnan(pcg_vector[1]) else 'None'
                
                pred_x = f"{pred_screen[0]:.2f}" if pred_screen and not np.isnan(pred_screen[0]) else 'None'
                pred_y = f"{pred_screen[1]:.2f}" if pred_screen and not np.isnan(pred_screen[1]) else 'None'
                
                # Las coordenadas del alvo y el ID siempre deberían estar presentes
                target_x = f"{alvo_coords[0]:.2f}"
                target_y = f"{alvo_coords[1]:.2f}"
                
                # Manejo del error angular por frame
                angular_error_frame_val = d.get('angular_error_deg', np.nan)
                angular_error_frame = f"{angular_error_frame_val:.2f}" if not np.isnan(angular_error_frame_val) else 'None'
                
                # Manejo del promedio del error angular
                avg_error_val = d.get('avg_angular_error_target_deg', np.nan)
                avg_angular_error_target = f"{avg_error_val:.2f}" if not np.isnan(avg_error_val) else 'None'

                writer.writerow([
                    f"{d['timestamp']:.6f}",
                    d['target_id'],
                    pcg_x, pcg_y,
                    target_x, target_y,
                    pred_x, pred_y,
                    angular_error_frame,
                    avg_angular_error_target
                ])
                
        print(f"Datos completos (incluyendo Error Angular y Promedios) guardados en: {pred_gaze_data_path}")
        print("El archivo CSV contiene una columna con el error angular para cada frame, y otra columna con el promedio de ese error angular por cada alvo (ID).")
        
    except Exception as e:
        print(f"¡ADVERTENCIA CRÍTICA! Fallo al escribir el archivo CSV: {e}")
        print("El archivo CSV no pudo ser generado debido al error anterior.")


# Funciones auxiliares (sin cambios relevantes en la interfaz)

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

def fit_polynomial_2nd_order(pcg_points, screen_points):
    """Ajusta un mapeo polinomial de 2do orden (PCG Vector -> pantalla)."""
    N = len(pcg_points)
    if N < 6:
        # Si no hay suficientes puntos para el ajuste polinomial de 2do orden (necesita al menos 6)
        print(f"Advertencia: Se necesitan al menos 6 puntos para el ajuste de 2do orden. Solo se obtuvieron {N}.")
        # Se rellena con 0 si no hay suficientes puntos para que el código no falle
        return np.zeros(6), np.zeros(6) 

    # La matriz A representa los términos del polinomio: [x^2, x*y, y^2, x, y, 1]
    A = np.zeros((N, 6))
    Bx = np.zeros(N) # Coordenadas X de la pantalla (Target)
    By = np.zeros(N) # Coordenadas Y de la pantalla (Target)
    
    for i, ((x, y), (X, Y)) in enumerate(zip(pcg_points, screen_points)):
        A[i] = [x**2, x*y, y**2, x, y, 1]
        Bx[i] = X
        By[i] = Y
        
    coeffs_x, _, _, _ = np.linalg.lstsq(A, Bx, rcond=None)
    coeffs_y, _, _, _ = np.linalg.lstsq(A, By, rcond=None)
    return coeffs_x, coeffs_y

def map_pupil_to_screen(x, y, coeffs_x, coeffs_y):
    """Aplica los coeficientes del polinomio al vector PCG (x, y) para predecir la posición en pantalla."""
    args = np.array([x**2, x*y, y**2, x, y, 1])
    X = np.dot(coeffs_x, args)
    Y = np.dot(coeffs_y, args)
    return (X, Y)

def wait_for_enter(window_name, message, width, height, is_fullscreen=True):
    """Muestra un mensaje en pantalla (completa o no) y espera la tecla ENTER."""
    if is_fullscreen:
        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        img_wait = np.zeros((height, width, 3), dtype=np.uint8)
        
        text_size_main = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
        text_x_main = (width - text_size_main[0]) // 2
        text_y_main = height // 2
        cv2.putText(img_wait, message, (text_x_main, text_y_main), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

        instruction = "Presiona ENTER para continuar."
        text_size_inst = cv2.getTextSize(instruction, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_x_inst = (width - text_size_inst[0]) // 2
        text_y_inst = text_y_main + 80
        cv2.putText(img_wait, instruction, (text_x_inst, text_y_inst), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        cv2.imshow(window_name, img_wait)

        key = cv2.waitKey(10) 
        if key == 13: 
            cv2.destroyWindow(window_name)
            break
        if key == 27: 
            exit()

if __name__ == "__main__":
    main()
