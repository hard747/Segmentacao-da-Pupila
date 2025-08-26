import cv2
import numpy as np
import os
import collections
import time

from scipy.optimize import least_squares

# --- Constantes globais ---
# Constantes para a classificação de pupila (AJUSTAR ESTES VALORES!)
MIN_BRIGHT_PIXEL_VALUE = 200 # Valor MÍNIMO de intensidade que um PIXEL deve ter para ser considerado "muito brilhante".
                             # Ajuste este valor (ex. 230, 240, 250) conforme seu vídeo.
MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL = 40 # NÚMERO MÍNIMO de pixels que devem atingir MIN_BRIGHT_PIXEL_VALUE
                                     # para que o frame seja classificado como "Pupila Clara".
                                     # AJUSTE ESTE VALOR! Tente com 50, 100, 200, etc.

# Constantes para a detecção de Glints (AJUSTADAS PARA MELHOR DETECÇÃO EM PUPILA ESCURA)
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL = 0.85 # Porcentaje del valor máximo de brillo para glint en pupila escura.
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL = 0.90 # Porcentaje para glint en pupila brillante.

GLINT_MIN_AREA = 3        # Área mínima do contorno de um glint. (Reducido ligeramente)
GLINT_MAX_AREA = 120      # Área máxima do contorno de um glint. (Aumentado ligeramente)
GLINT_CIRCULARITY_THRESHOLD = 0.65 # Circularidade mínima para considerar um glint (são redondos). (Reducido ligeramente)

# --- Funções auxiliares ---

def fit_circle_to_points(points):
    """
    Ajusta un círculo a un conjunto de puntos 2D usando el método de mínimos cuadrados.
    """
    if points is None or points.shape[0] < 3:
        return None, None, None

    def residuals(params, x_data, y_data):
        center_x, center_y, radius = params
        return np.sqrt((x_data - center_x)**2 + (y_data - center_y)**2) - radius

    initial_center_x = np.mean(points[:, 0])
    initial_center_y = np.mean(points[:, 1])
    initial_radius = np.mean(np.sqrt((points[:, 0] - initial_center_x)**2 + (points[:, 1] - initial_center_y)**2))
    
    initial_params = [initial_center_x, initial_center_y, initial_radius]

    try:
        result = least_squares(residuals, initial_params, args=(points[:, 0], points[:, 1]), bounds=([0, 0, 1], np.inf))
        center_x, center_y, radius = result.x
        return center_x, center_y, radius
    except Exception as e:
        # print(f"Erro ao ajustar círculo: {e}") # Descomentar para depurar
        return None, None, None

def identificar_tipo_pupila_por_conteo_pixeles(img_gray, min_bright_pixel_val, min_pixel_count_for_clear_pupil):
    """
    Identifica se uma imagem contém pupila escura ou brilhante
    com base na contagem de pixels acima de um limiar de brilho.
    """
    if img_gray is None:
        return 'desconhecido', {}

    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    bright_pixels_in_range_count = np.sum(hist[min_bright_pixel_val : 256])

    metricas = {
        'bright_pixel_count': int(bright_pixels_in_range_count)
    }

    # CORREÇÃO AQUI: 'min_pixel_count_for_clear_pupiL' mudado para 'min_pixel_count_for_clear_pupil'
    if bright_pixels_in_range_count >= min_pixel_count_for_clear_pupil:
        return 'brilhante', metricas
    else:
        return 'escura', metricas

def detectar_glints(img_gray, roi_mask=None, is_dark_pupil_frame=False):
    """
    Detecta os glints (pontos de luz) em uma imagem.
    Adicionado um parâmetro 'is_dark_pupil_frame' para ajustar o limiar de brilho.
    """
    glints = []
    
    if roi_mask is not None:
        img_to_process = cv2.bitwise_and(img_gray, img_gray, mask=roi_mask)
    else:
        img_to_process = img_gray.copy()

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img_to_process)
    
    if max_val < 50: # Se não houver pixels suficientemente brilhantes, não há glints significativos.
        return glints
    
    # AJUSTE CHAVE: Usar limiares de brilho diferentes dependendo do tipo de pupila
    if is_dark_pupil_frame:
        glint_threshold = int(max_val * GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL)
    else: # Para pupila brilhante
        glint_threshold = int(max_val * GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL)
    
    # Garantir que o limiar não seja muito baixo para evitar ruído, ou muito alto para perder glints
    if glint_threshold < 100: # Valor mínimo do limiar, ajuste se necessário
        glint_threshold = 100
    if glint_threshold > 240: # Valor máximo do limiar, ajuste se necessário
        glint_threshold = 240

    _, bright_spots_thresh = cv2.threshold(img_to_process, glint_threshold, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((3,3), np.uint8)
    # Operações morfológicas para limpar os glints:
    # Fechamento para unir pequenos pontos e abertura para remover ruído.
    bright_spots_thresh = cv2.morphologyEx(bright_spots_thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    bright_spots_thresh = cv2.morphologyEx(bright_spots_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(bright_spots_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Ajuste estes valores se seus glints forem maiores o menores
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


def segmentar_pupilas_individual_frames(frame_escura, frame_brilhante, par_idx):
    """
    Segmenta las pupilas escura y brillante individualmente usando ROI y Watershed.
    También detecta glints y modifica la imagen para la segmentación de la pupila
    para minimizar la interferencia de los glints, utilizando la lógica de segmentación preferida.
    """
    if frame_escura is None or frame_brilhante is None:
        # print(f"Error: Uno de los frames para segmentación (par {par_idx}) es None.") # Descomentar para depurar
        return None

    escura_gray = cv2.cvtColor(frame_escura, cv2.COLOR_BGR2GRAY)
    brilhante_gray = cv2.cvtColor(frame_brilhante, cv2.COLOR_BGR2GRAY)

    # --- Refinamiento de la ROI (Región de Interés) ---
    diff = cv2.subtract(brilhante_gray, escura_gray)
    cv2.imshow('diferencia',diff)
    diff_blurred = cv2.GaussianBlur(diff, (9, 9), 0)
    cv2.imshow('blurred',diff_blurred)
    _, roi_mask = cv2.threshold(diff_blurred, 40, 255, cv2.THRESH_BINARY)
    cv2.imshow('roi', roi_mask) 
    
    kernel_small = np.ones((3,3), np.uint8)
    kernel_medium = np.ones((5,5), np.uint8)

    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, kernel_medium, iterations=2)
    roi_mask = cv2.erode(roi_mask, kernel_small, iterations=1)
    roi_mask = cv2.dilate(roi_mask, kernel_small, iterations=1)

    cv2.imshow('roi_mejorado', roi_mask)

    all_escura_pupils_info = []
    all_brilhante_pupils_info = []
    glints_info_escura_frame = []
    glints_info_brilhante_frame = []

    # --- Detecção de Glints ---
    # Passar o flag is_dark_pupil_frame=True quando os glints são detectados no frame escuro
    glints_info_escura_frame = detectar_glints(escura_gray, roi_mask, is_dark_pupil_frame=True)
    glints_info_brilhante_frame = detectar_glints(brilhante_gray, roi_mask, is_dark_pupil_frame=False)

    # --- Preparar imagens para a segmentação da pupila (eliminar glints) ---
    escura_gray_no_glints = escura_gray.copy()
    brilhante_gray_no_glints = brilhante_gray.copy()

    # Desenha um círculo preto sobre os glints para eliminá-los da imagem
    # antes da segmentação da pupila. Isso evita que os glints sean
    # confundidos con parte de la pupila.
    for glint in glints_info_escura_frame:
        center = glint['center']
        # Pintar de preto una región alredeor del glint para eliminar su influencia en la segmentación
        # Usado un radio basado en la área del glint, multiplicado por 1.5 para garantir que sea bien cubierto.
        cv2.circle(escura_gray_no_glints, center, int(max(2, np.sqrt(glint['area']/np.pi) * 1.5)), 0, -1) 
    
    for glint in glints_info_brilhante_frame:
        center = glint['center']
        # Usar un tamaño de máscara para glints más conservador para pupilas brilhantes, si es necesario.
        # Podría-se tentar con 2.0 si aún hay problemas, pero cuidado para no apagar la pupila.
        cv2.circle(brilhante_gray_no_glints, center, int(max(2, np.sqrt(glint['area']/np.pi) * 1.5)), 0, -1)

    # --- Segmentar Pupila Escura na ROI (usando imagem sem glints) ---
    result_escura_borda = frame_escura.copy()
    pupil_mask_escura = np.zeros_like(escura_gray) # Máscara para la pupila escura
    escura_circle_fit_contour = None # Para guardar el contorno del círculo ajustado
    
    try:
        # Os limiares para a segmentação da pupila são mantidos,
        # já que a imagem de entrada não possui mais os glints que a confundem.
        _, escura_thresh_inv = cv2.threshold(escura_gray_no_glints, 50, 255, cv2.THRESH_BINARY_INV)
        escura_thresh_inv_roi = cv2.bitwise_and(escura_thresh_inv, escura_thresh_inv, mask=roi_mask)
        
        opening_escura = cv2.morphologyEx(escura_thresh_inv_roi, cv2.MORPH_OPEN, kernel_small, iterations=2)
        sure_bg_escura = cv2.dilate(opening_escura, kernel_small, iterations=3) 
        dist_transform_escura = cv2.distanceTransform(opening_escura, cv2.DIST_L2, 5)
        _, sure_fg_escura = cv2.threshold(dist_transform_escura, 0.5 * dist_transform_escura.max(), 255, 0)
        sure_fg_escura = np.uint8(sure_fg_escura)
        
        unknown_escura = cv2.subtract(sure_bg_escura, sure_fg_escura)
        _, markers_escura = cv2.connectedComponents(sure_fg_escura)
        markers_escura = markers_escura + 1
        markers_escura[unknown_escura == 255] = 0
        
        if np.max(markers_escura) > 1:
            markers_escura_final = cv2.watershed(frame_escura.copy(), markers_escura.copy())
            
            mask_escura_all = np.zeros_like(escura_gray)
            mask_escura_all[markers_escura_final > 1] = 255
            
            # Asignar la máscara de la pupila escura
            pupil_mask_escura = mask_escura_all.copy()

            contours_escura, _ = cv2.findContours(mask_escura_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt_escura in contours_escura:
                area = cv2.contourArea(cnt_escura)
                if 50 < area < 6000: 
                    perimeter = cv2.arcLength(cnt_escura, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter * perimeter)
                        if circularity > 0.35: 
                            
                            points_escura = cnt_escura.squeeze()
                            if points_escura.ndim == 1 or points_escura.shape[0] < 3: 
                                continue
                            
                            cx, cy, r = fit_circle_to_points(points_escura)
                            
                            if cx is not None:
                                escura_center = (int(cx), int(cy))
                                cv2.drawContours(result_escura_borda, [cnt_escura], -1, (255, 0, 0), 2) # Vermelho para pupila
                                all_escura_pupils_info.append({
                                    'center': escura_center,
                                    'radius': r, 
                                    'segmentada': True,
                                    'contour': cnt_escura,
                                    'fit_circle': (cx, cy, r) # Guardar los datos del círculo ajustado
                                })
                                # Si hay múltiples pupilas, tomamos la primera o la más grande.
                                # Por simplicidad, aquí tomamos la última que cumpla los criterios.
                                # Si se desea el mejor ajuste, se necesitaría lógica adicional.
                                escura_circle_fit_contour = {'center': (int(cx), int(cy)), 'radius': int(r)}
                                
    except Exception as e:
        print(f"Erro ao segmentar pupila escura no par {par_idx}: {e}")
    
    # --- Segmentar Pupila Brilhante na ROI (usando imagem sem glints) ---
    result_brilhante_borda = frame_brilhante.copy()
    pupil_mask_brilhante = np.zeros_like(brilhante_gray) # Máscara para la pupila brillante
    brilhante_circle_fit_contour = None # Para guardar el contorno del círculo ajustado
    
    try:
        brilhante_roi_no_glints = cv2.bitwise_and(brilhante_gray_no_glints, brilhante_gray_no_glints, mask=roi_mask)
        cv2.imshow('glints', brilhante_roi_no_glints)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(brilhante_roi_no_glints, mask=roi_mask)
        
        # --- AJUSTES PROPOSTOS PARA PUPILA BRILHANTE ---
        brilhante_threshold_value = 60 # Valor padrão

        if max_val > 0:
            brilhante_threshold_value = int(max_val * 0.5) 

            if brilhante_threshold_value < 50: 
                brilhante_threshold_value = 50
            elif brilhante_threshold_value > 200: 
                brilhante_threshold_value = 200

        _, brilhante_thresh = cv2.threshold(brilhante_roi_no_glints, brilhante_threshold_value, 255, cv2.THRESH_BINARY)
        
        brilhante_thresh = cv2.morphologyEx(brilhante_thresh, cv2.MORPH_OPEN, kernel_small, iterations=1) 
        brilhante_thresh = cv2.morphologyEx(brilhante_thresh, cv2.MORPH_CLOSE, kernel_small, iterations=1)


        # --- Watershed ---
        opening_brilhante = cv2.morphologyEx(brilhante_thresh, cv2.MORPH_OPEN, kernel_small, iterations=2)
        sure_bg_brilhante = cv2.dilate(opening_brilhante, kernel_small, iterations=3) 
        dist_transform_brilhante = cv2.distanceTransform(opening_brilhante, cv2.DIST_L2, 5)
        dist_max = dist_transform_brilhante.max()

        if dist_max > 0:
              _, sure_fg_brilhante = cv2.threshold(dist_transform_brilhante, 0.4 * dist_max, 255, 0) 
        else:
              sure_fg_brilhante = np.zeros_like(brilhante_gray)
        sure_fg_brilhante = np.uint8(sure_fg_brilhante)
        unknown_brilhante = cv2.subtract(sure_bg_brilhante, sure_fg_brilhante)
        _, markers_brilhante = cv2.connectedComponents(sure_fg_brilhante)
        markers_brilhante = markers_brilhante + 1
        markers_brilhante[unknown_brilhante == 255] = 0
        
        if np.max(markers_brilhante) > 1:
            markers_brilhante_final = cv2.watershed(frame_brilhante.copy(), markers_brilhante.copy())
            
            mask_brilhante_all = np.zeros_like(brilhante_gray)
            mask_brilhante_all[markers_brilhante_final > 1] = 255

            # Asignar la máscara de la pupila brillante
            pupil_mask_brilhante = mask_brilhante_all.copy()
            
            contours_brilhante, _ = cv2.findContours(mask_brilhante_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt_brilhante in contours_brilhante:
                area = cv2.contourArea(cnt_brilhante)
                if 40 < area < 7000: 
                    perimeter = cv2.arcLength(cnt_brilhante, True)
                    if perimeter > 0:
                        circularity = (4 * np.pi * area) / (perimeter * perimeter)
                        if circularity > 0.30: 
                            
                            points_brilhante = cnt_brilhante.squeeze()
                            if points_brilhante.ndim == 1 or points_brilhante.shape[0] < 3:
                                continue

                            cx, cy, r = fit_circle_to_points(points_brilhante)
                            
                            if cx is not None:
                                brilhante_center = (int(cx), int(cy))
                                cv2.drawContours(result_brilhante_borda, [cnt_brilhante], -1, (0, 255, 0), 2) # Verde para pupila
                                all_brilhante_pupils_info.append({
                                    'center': brilhante_center,
                                    'radius': r, 
                                    'segmentada': True,
                                    'contour': cnt_brilhante,
                                    'fit_circle': (cx, cy, r) # Guardar los datos del círculo ajustado
                                })
                                # Si hay múltiples pupilas, tomamos la primera o la más grande.
                                brilhante_circle_fit_contour = {'center': (int(cx), int(cy)), 'radius': int(r)}
                                
    except Exception as e:
        print(f"Erro ao segmentar pupila brilhante no par {par_idx}: {e}")

    # Desenhar os glints nos resultados finais de cada frame
    # Os glints em pupila escura são desenhados em VERMELHO
    for glint in glints_info_escura_frame:
        center = glint['center']
        cv2.circle(result_escura_borda, center, 5, (0, 0, 255), -1) # VERMELHO, tamaño 5, preenchido
    
    # Os glints em pupila brilhante são desenhados em AMARELO (como antes)
    for glint in glints_info_brilhante_frame:
        center = glint['center']
        cv2.circle(result_brilhante_borda, center, 3, (0, 255, 255), -1) # Amarelo, tamaño 3, preenchido


    return {
        'escura_gray': escura_gray, # Añadido para visualización
        'brilhante_gray': brilhante_gray, # Añadido para visualización
        'diferenca': diff,
        'roi_mask': roi_mask,
        'resultado_escura_borda': result_escura_borda,
        'pupilas_escuras': all_escura_pupils_info,
        'glints_escura_frame': glints_info_escura_frame, 
        'resultado_brilhante_borda': result_brilhante_borda,
        'pupilas_brillantes': all_brilhante_pupils_info,
        'glints_brilhante_frame': glints_info_brilhante_frame, 
        'pupil_mask_escura': pupil_mask_escura, # Añadido
        'pupil_mask_brilhante': pupil_mask_brilhante, # Añadido
        'escura_circle_fit_contour': escura_circle_fit_contour, # Añadido
        'brilhante_circle_fit_contour': brilhante_circle_fit_contour # Añadido
    }

class PupilTrackerApp:
    # Modos de visualização
    NORMAL = 0
    PUPILA_CLARA = 1
    PUPILA_ESCURA = 2
    DIFERENCA_IMAGENS = 3
    CONTORNO_CLARA = 4
    CONTORNO_ESCURA = 5

    def __init__(self, video_path, output_video_path, max_frames_buffer=2):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.max_frames_buffer = max_frames_buffer
        self.current_display_mode = self.NORMAL
        self.mode_names = {
            self.NORMAL: "MODO: Normal (Teclas: 1-6 para mudar)",
            self.PUPILA_CLARA: "MODO: Somente Frames de Pupila Clara (FILTRADO)",
            self.PUPILA_ESCURA: "MODO: Somente Frames de Pupila Escura (FILTRADO)",
            self.DIFERENCA_IMAGENS: "MODO: Mascara ROI (Diferenca Binarizada)",
            self.CONTORNO_CLARA: "MODO: Contorno Pupila Clara (FILTRADO)",
            self.CONTORNO_ESCURA: "MODO: Contorno Pupila Escura (FILTRADO)"
        }
        self.loop_video = True 
        self.paused = False 
        self.frame_to_process = None # Variable para guardar el frame actual para procesamiento

    def processar_video_pupilas(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo: {self.video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 25 
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Define fourcc (video codec) before it's used
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 

        # Para los modos filtrados, ajustamos el FPS de salida para que el video no se vea acelerado
        # Esto es clave para que los videos "filtrados" no se vean en cámara rápida.
        # Por ejemplo, si solo la mitad de los frames son de pupila clara, el video resultante
        # debe tener la mitad de los frames por segundo para mantener la duración.
        # Sin embargo, VideoWriter necesita un FPS fijo. En lugar de cambiar el FPS de salida,
        # lo que haremos es *escribir menos frames* al archivo de salida para los modos filtrados,
        # manteniendo el FPS original para la reproducción fluida del video final.
        # Esto significa que los videos filtrados serán más cortos.
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))


        if not out.isOpened():
            print(f"Erro ao criar o vídeo de saída: {self.output_video_path}")
            print("Certifique-se de que o codec 'XVID' está disponível em seu sistema.")
            print("Você pode precisar instalar ou atualizar pacotes de codecs para AVI, como `ffmpeg` ou `gstreamer`.")
            cap.release()
            return

        print(f"Processando vídeo: {self.video_path}")
        print(f"Dimensões: {frame_width}x{frame_height}, FPS: {fps}, Total de Frames: {total_frames}\n")
        print("Pressione '1' para modo Normal (segmentación y glints).")
        print("Pressione '2' para modo Somente Frames de Pupila Clara (FILTRADO).")
        print("Pressione '3' para modo Somente Frames de Pupila Escura (FILTRADO).")
        print("Pressione '4' para modo Máscara ROI (Diferença Binarizada - SEM MARCADORES).")
        print("Pressione '5' para modo Contorno Pupila Clara (FILTRADO).")
        print("Pressione '6' para modo Contorno Pupila Escura (FILTRADO).")
        print("Pressione 'L' para alternar modo de bucle (Loop ON/OFF).")
        print("Pressione 'P' para Pausar/Reproduzir.")
        print("Pressione 'N' para avançar um frame (somente em pausa).")
        print("Pressione 'q' para sair.")

        frame_buffer = collections.deque(maxlen=self.max_frames_buffer)
        frame_idx = 0
        pares_segmentados_count = 0
        segmentacoes_fallidas_escura = 0
        segmentacoes_fallidas_brilhante = 0

        all_pupil_centers_data = [] 
        all_glints_data = [] 

        # Leer el primer frame para comenzar
        ret, self.frame_to_process = cap.read()
        if not ret:
            print("No se pudo leer el primer frame. El video podría estar vacío o corrupto.")
            cap.release()
            out.release()
            return

        while True:
            # Determinar el delay para waitKey
            wait_delay = 0 if self.paused else 1

            # --- Manejo de eventos de teclado ---
            key = cv2.waitKey(wait_delay) & 0xFF 

            if key == ord('q'):
                break
            elif key == ord('l') or key == ord('L'):
                self.loop_video = not self.loop_video
                print(f"Modo de bucle: {'Activado' if self.loop_video else 'Desactivado'}")
            elif key == ord('p') or key == ord('P'):
                self.paused = not self.paused
                print(f"Reproducción: {'Pausada' if self.paused else 'Reproduciendo'}")
            elif key == ord('n') or key == ord('N'):
                if self.paused: # Solo avanza frame a frame si ya está pausado
                    ret, self.frame_to_process = cap.read()
                    if not ret: # Si no hay más frames al avanzar un paso
                        if self.loop_video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0 # Reiniciar contador de frames
                            frame_buffer.clear() # Limpiar buffer
                            ret, self.frame_to_process = cap.read()
                            if not ret: break # Si aun así no lee, salir
                        else:
                            break # Salir si no hay más frames y no hay bucle
                    frame_idx += 1 # Incrementar frame_idx solo si se leyó un nuevo frame
                # Si no está pausado y presiona 'N', no hace nada, la reproducción continúa.
            elif key == ord('1'):
                self.current_display_mode = self.NORMAL
            elif key == ord('2'):
                self.current_display_mode = self.PUPILA_CLARA
            elif key == ord('3'):
                self.current_display_mode = self.PUPILA_ESCURA
            elif key == ord('4'):
                self.current_display_mode = self.DIFERENCA_IMAGENS
            elif key == ord('5'):
                self.current_display_mode = self.CONTORNO_CLARA
            elif key == ord('6'):
                self.current_display_mode = self.CONTORNO_ESCURA

            # --- Lógica de lectura de frames para reproducción normal ---
            if not self.paused and key not in [ord('n'), ord('N')]: # Si no está pausado Y no se presionó N (porque N ya lee un frame)
                ret, self.frame_to_process = cap.read()
                if not ret: # Si no hay más frames
                    if self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        frame_buffer.clear()
                        ret, self.frame_to_process = cap.read()
                        if not ret: break
                    else: break
                frame_idx += 1 # Incrementar frame_idx solo si se leyó un nuevo frame
            
            # Si no hay un frame válido para procesar, salir del bucle
            if self.frame_to_process is None:
                break


            # --- Procesa el frame actual ---
            frame_gray = cv2.cvtColor(self.frame_to_process, cv2.COLOR_BGR2GRAY)
            
            tipo_pupila, _ = identificar_tipo_pupila_por_conteo_pixeles(
                frame_gray, MIN_BRIGHT_PIXEL_VALUE, MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL
            )

            frame_info = {
                'frame_num': frame_idx,
                'frame_data': self.frame_to_process,
                'tipo': tipo_pupila,
                'gray_data': frame_gray 
            }
            frame_buffer.append(frame_info)

            current_frame_to_write = self.frame_to_process.copy() 
            segmentation_results = None

            pupil_centers_current_frame = {
                'frame_num': frame_idx,
                'pupilas': []
            }
            glints_current_frame = {
                'frame_num': frame_idx,
                'glints': []
            }

            # Lógica para determinar si se procesará un par de frames
            # y cuál frame del par se usará para la visualización actual
            process_pair = False
            frame_to_display_from_pair = None # Guarda el frame que se mostrará y procesará
            current_frame_type_for_display = 'desconocido' # Tipo de pupila del frame_to_display_from_pair

            if len(frame_buffer) == 2:
                f1 = frame_buffer[0] 
                f2 = frame_buffer[1] 

                # Si tenemos un par escura/brilhante, los usamos para segmentación
                if (f1['tipo'] == 'escura' and f2['tipo'] == 'brilhante') or \
                   (f1['tipo'] == 'brilhante' and f2['tipo'] == 'escura'):
                    escura_frame_data = f1['frame_data'] if f1['tipo'] == 'escura' else f2['frame_data']
                    brilhante_frame_data = f1['frame_data'] if f1['tipo'] == 'brilhante' else f2['frame_data']
                    process_pair = True
                    # El frame para mostrar es siempre el más reciente en el buffer (f2)
                    frame_to_display_from_pair = f2['frame_data']
                    current_frame_type_for_display = f2['tipo']
                else:
                    # Si no es un par válido, solo consideramos el último frame (f2)
                    # para clasificar y detectar glints si aplica.
                    frame_to_display_from_pair = f2['frame_data']
                    current_frame_type_for_display = f2['tipo']
            else:
                # Si el buffer tiene menos de 2 frames (solo el primero),
                # solo consideramos el frame actual (frame_info)
                frame_to_display_from_pair = frame_info['frame_data']
                current_frame_type_for_display = frame_info['tipo']

            current_frame_to_write = frame_to_display_from_pair.copy()

            if process_pair:
                pares_segmentados_count += 1
                segmentation_results = segmentar_pupilas_individual_frames(
                    escura_frame_data, brilhante_frame_data, pares_segmentados_count
                )
                
                if segmentation_results:
                    if segmentation_results['pupilas_brillantes']:
                        for pupil_info in segmentation_results['pupilas_brillantes']:
                            center = pupil_info['center']
                            pupil_centers_current_frame['pupilas'].append({
                                'tipo': 'brilhante',
                                'center_x': center[0],
                                'center_y': center[1]
                            })
                    else:
                        if current_frame_type_for_display == 'brilhante': 
                            segmentacoes_fallidas_brilhante += 1
                    
                    for glint_info in segmentation_results['glints_brilhante_frame']:
                        center = glint_info['center']
                        glints_current_frame['glints'].append({
                            'tipo': 'glint_brilhante',
                            'center_x': center[0],
                            'center_y': center[1]
                        })
                    
                    if segmentation_results['pupilas_escuras']:
                        for pupil_info in segmentation_results['pupilas_escuras']:
                            center = pupil_info['center']
                            pupil_centers_current_frame['pupilas'].append({
                                'tipo': 'escura',
                                'center_x': center[0],
                                'center_y': center[1]
                            })
                    else:
                        if current_frame_type_for_display == 'escura': 
                                segmentacoes_fallidas_escura += 1

                    for glint_info in segmentation_results['glints_escura_frame']:
                        center = glint_info['center']
                        glints_current_frame['glints'].append({
                            'tipo': 'glint_escura',
                            'center_x': center[0],
                            'center_y': center[1]
                        })

                    # --- Lógica de Visualización basada en el modo ---
                    if self.current_display_mode == self.NORMAL:
                        if current_frame_type_for_display == 'brilhante':
                            current_frame_to_write = segmentation_results['resultado_brilhante_borda']
                        elif current_frame_type_for_display == 'escura':
                            current_frame_to_write = segmentation_results['resultado_escura_borda']
                        
                        # Dibujar centros de pupila en modo NORMAL
                        for pupil in pupil_centers_current_frame['pupilas']:
                            color = (255, 0, 255) if pupil['tipo'] == 'brilhante' else (0, 0, 255)
                            cv2.circle(current_frame_to_write, (pupil['center_x'], pupil['center_y']), 3, color, -1)

                    elif self.current_display_mode == self.PUPILA_CLARA:
                        # Solo muestra el frame si es de tipo 'brilhante'
                        if current_frame_type_for_display == 'brilhante':
                            current_frame_to_write = frame_to_display_from_pair.copy() # Sin glints ni centros
                        else:
                            current_frame_to_write = None # No mostrar frame

                    elif self.current_display_mode == self.PUPILA_ESCURA:
                        # Solo muestra el frame si es de tipo 'escura'
                        if current_frame_type_for_display == 'escura':
                            current_frame_to_write = frame_to_display_from_pair.copy() # Sin glints ni centros
                        else:
                            current_frame_to_write = None # No mostrar frame

                    elif self.current_display_mode == self.DIFERENCA_IMAGENS:
                        if segmentation_results['roi_mask'] is not None and np.any(segmentation_results['roi_mask']):
                            current_frame_to_write = cv2.cvtColor(segmentation_results['roi_mask'], cv2.COLOR_GRAY2BGR)
                        else:
                            current_frame_to_write = None # No mostrar frame

                    elif self.current_display_mode == self.CONTORNO_CLARA:
                        # Solo muestra el frame si es de tipo 'brilhante' y se encontró un círculo ajustado
                        if current_frame_type_for_display == 'brilhante' and segmentation_results['brilhante_circle_fit_contour']:
                            current_frame_to_write = frame_to_display_from_pair.copy() 
                            circle_info = segmentation_results['brilhante_circle_fit_contour']
                            center = circle_info['center']
                            radius = circle_info['radius']
                            cv2.circle(current_frame_to_write, center, radius, (0, 255, 0), 2) 
                            cv2.circle(current_frame_to_write, center, 3, (0, 255, 0), -1) 
                        else:
                            current_frame_to_write = None # No mostrar frame

                    elif self.current_display_mode == self.CONTORNO_ESCURA:
                        # Solo muestra el frame si es de tipo 'escura' y se encontró un círculo ajustado
                        if current_frame_type_for_display == 'escura' and segmentation_results['escura_circle_fit_contour']:
                            current_frame_to_write = frame_to_display_from_pair.copy() 
                            circle_info = segmentation_results['escura_circle_fit_contour']
                            center = circle_info['center']
                            radius = circle_info['radius']
                            cv2.circle(current_frame_to_write, center, radius, (255, 0, 0), 2) 
                            cv2.circle(current_frame_to_write, center, 3, (255, 0, 0), -1) 
                        else:
                            current_frame_to_write = None # No mostrar frame
                else:
                    # Si la segmentación falló (no results), aún podemos mostrar el frame original
                    # pero solo para el modo NORMAL y si corresponde al tipo de frame.
                    # Para otros modos, se oculta o muestra glints si aplica.
                    if self.current_display_mode == self.NORMAL:
                        current_frame_to_write = frame_to_display_from_pair.copy()
                        glints_info = []
                        if current_frame_type_for_display == 'escura':
                            glints_info = detectar_glints(cv2.cvtColor(frame_to_display_from_pair, cv2.COLOR_BGR2GRAY), is_dark_pupil_frame=True)
                        elif current_frame_type_for_display == 'brilhante':
                            glints_info = detectar_glints(cv2.cvtColor(frame_to_display_from_pair, cv2.COLOR_BGR2GRAY), is_dark_pupil_frame=False)
                        
                        for glint in glints_info:
                            color = (0, 0, 255) if current_frame_type_for_display == 'escura' else (0, 255, 255)
                            radius = 5 if current_frame_type_for_display == 'escura' else 3
                            cv2.circle(current_frame_to_write, glint['center'], radius, color, -1)
                    else:
                        current_frame_to_write = None # No mostrar frame

            # Solo mostrar y escribir si el frame_to_write no es None
            if current_frame_to_write is not None:
                if pupil_centers_current_frame['pupilas']:
                    all_pupil_centers_data.append(pupil_centers_current_frame)
                
                if glints_current_frame['glints']:
                    all_glints_data.append(glints_current_frame)

                # --- Mostrar el modo actual en la ventana ---
                mode_text = self.mode_names.get(self.current_display_mode, "MODO: Desconocido")
                loop_status = "ON" if self.loop_video else "OFF"
                pause_status = "PAUSADO" if self.paused else "REPRODUCIENDO"
                status_text = f"{mode_text} | Loop: {loop_status} | {pause_status}"

                cv2.putText(current_frame_to_write, status_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(current_frame_to_write, "Q: Sair | 1-6: Mudar Modo | L: Loop | P: Pausa | N: Avancar Frame", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                cv2.imshow("Pupil Segmentation", current_frame_to_write)
                out.write(current_frame_to_write)
            else:
                # Si el frame es None (debe ser negro), no actualizamos la ventana
                # para que el frame anterior se mantenga hasta que se muestre uno nuevo.
                pass # No hacer nada si el frame no debe ser mostrado


        print(f"\nProcessamento de vídeo concluído.")
        print(f"Total de pares tentados para segmentação: {pares_segmentados_count}")
        print(f"Segmentações de pupila escura falhas: {segmentacoes_fallidas_escura}")
        print(f"Segmentações de pupila brilhante falhas: {segmentacoes_fallidas_brilhante}")
        print(f"Total de frames processados: {total_frames}")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Vídeo de saída salvo em: {self.output_video_path}")

        output_pupil_centers_filename = "pupil_centers_data.txt"
        output_pupil_centers_path = os.path.join(os.getcwd(), output_pupil_centers_filename)

        with open(output_pupil_centers_path, 'w') as f:
            f.write("Dados dos Centros de Pupilas Segmentadas:\n")
            for frame_data in all_pupil_centers_data:
                f.write(f"Frame {frame_data['frame_num']}:\n")
                for pupil in frame_data['pupilas']:
                    f.write(f"     Tipo: {pupil['tipo']}, Centro (x,y): ({pupil['center_x']}, {pupil['center_y']})\n")
                f.write("\n")

        print(f"Dados dos centros de pupilas salvos em: {output_pupil_centers_path}")

        output_glints_filename = "glints_data.txt"
        output_glints_path = os.path.join(os.getcwd(), output_glints_filename)

        with open(output_glints_path, 'w') as f:
            f.write("Dados dos Glints Detectados:\n")
            for frame_data in all_glints_data:
                f.write(f"Frame {frame_data['frame_num']}:\n")
                for glint in frame_data['glints']:
                    f.write(f"     Tipo: {glint['tipo']}, Centro (x,y): ({glint['center_x']}, {glint['center_y']})\n")
                f.write("\n")

        print(f"Dados dos glints salvos em: {output_glints_path}")


def main():
    """
    Função principal para executar o processamento do vídeo.
    """
    #video_path_input = 'grabacion_prueba.avi' # Assegure-se de que este arquivo existe no mesmo diretório
    video_path_input = 'video_prueba.avi'
    #video_path_input = 'gravacao_1754176523.avi'
    output_video_filename = 'video_pupila_segmentada_con_filtros.avi' # Nombre de salida actualizado
    output_video_path = os.path.join(os.getcwd(), output_video_filename)

    if not os.path.exists(video_path_input):
        print(f"Erro: O arquivo de vídeo de entrada '{video_path_input}' não foi encontrado.")
        print("Por favor, certifique-se de que o vídeo '{video_path_input}' está na MESMA PASTA que este script.")
        print("Não é possível prosseguir sem o arquivo de vídeo de entrada.")
        return

    app = PupilTrackerApp(video_path_input, output_video_path)
    app.processar_video_pupilas()

if __name__ == '__main__':
    main()

