import cv2
import numpy as np
import os
import collections
import time
from scipy.optimize import least_squares

# Constantes globais (ajustar según necesidad)
MIN_BRIGHT_PIXEL_VALUE = 200
MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL = 40

GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL = 0.85
GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL = 0.90
GLINT_MIN_AREA = 3
GLINT_MAX_AREA = 120
GLINT_CIRCULARITY_THRESHOLD = 0.65

class CircleFitter:
    @staticmethod
    def fit_circle(points):
        """
        Ajusta un círculo a un conjunto de puntos 2D usando mínimos cuadrados.
        """
        if points is None or len(points) < 3:
            return None, None, None
        
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
            return result.x  # cx, cy, r
        except Exception:
            return None, None, None

class PupilTypeClassifier:
    @staticmethod
    def classify_by_bright_pixels(img_gray, min_bright_val, min_pixel_count):
        if img_gray is None:
            return 'desconhecido', {}
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        bright_pixel_count = np.sum(hist[min_bright_val:256])
        metrics = {'bright_pixel_count': int(bright_pixel_count)}
        if bright_pixel_count >= min_pixel_count:
            return 'brilhante', metrics
        else:
            return 'escura', metrics

class GlintDetector:
    @staticmethod
    def detect_glints(img_gray, roi_mask=None, is_dark_pupil_frame=False):
        if img_gray is None:
            return []

        img_to_process = img_gray.copy()
        if roi_mask is not None:
            img_to_process = cv2.bitwise_and(img_gray, img_gray, mask=roi_mask)

        min_val, max_val, _, _ = cv2.minMaxLoc(img_to_process)
        if max_val < 50:
            return []  # No glints expected

        # Seleccionar threshold para glints según tipo de pupila
        if is_dark_pupil_frame:
            glint_thresh = int(max_val * GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_DARK_PUPIL)
        else:
            glint_thresh = int(max_val * GLINT_BRIGHTNESS_THRESHOLD_PERCENTAGE_BRIGHT_PUPIL)

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

class PupilSegmenter:
    def __init__(self):
        self.kernel_small = np.ones((3, 3), np.uint8)
        self.kernel_medium = np.ones((5, 5), np.uint8)

    def refine_roi(self, escura_gray, brilhante_gray):
        diff = cv2.subtract(brilhante_gray, escura_gray)
        diff_blurred = cv2.GaussianBlur(diff, (9, 9), 0)
        _, roi_mask = cv2.threshold(diff_blurred, 40, 255, cv2.THRESH_BINARY)

        # Morphological processing for ROI refinement
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, self.kernel_medium, iterations=2)
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, self.kernel_medium, iterations=2)
        roi_mask = cv2.erode(roi_mask, self.kernel_small, iterations=1)
        roi_mask = cv2.dilate(roi_mask, self.kernel_small, iterations=1)

        return roi_mask, diff

    def remove_glints_from_image(self, img_gray, glints_info):
        img_no_glints = img_gray.copy()
        for glint in glints_info:
            center = glint['center']
            radius = int(max(2, np.sqrt(glint['area'] / np.pi) * 1.5))
            cv2.circle(img_no_glints, center, radius, 0, -1)  # Black circle
        return img_no_glints

    def segment_pupil_dark(self, escura_gray_no_glints, roi_mask, frame_original):
        pupil_mask = np.zeros_like(escura_gray_no_glints)
        result_img = frame_original.copy()
        pupil_info_list = []

        try:
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

            if np.max(markers) > 1:
                markers_final = cv2.watershed(frame_original.copy(), markers.copy())
                mask_all = np.zeros_like(escura_gray_no_glints)
                mask_all[markers_final > 1] = 255
                pupil_mask = mask_all.copy()

                contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 50 < area < 6000:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = (4 * np.pi * area) / (perimeter ** 2)
                            if circularity > 0.35:
                                points = cnt.squeeze()
                                if points.ndim == 1 or points.shape[0] < 3:
                                    continue
                                cx, cy, r = CircleFitter.fit_circle(points)
                                if cx is not None:
                                    center = (int(cx), int(cy))
                                    cv2.drawContours(result_img, [cnt], -1, (255, 0, 0), 2)
                                    pupil_info_list.append({
                                        'center': center,
                                        'radius': r,
                                        'segmentada': True,
                                        'contour': cnt,
                                        'fit_circle': (cx, cy, r)
                                    })
        except Exception as e:
            print(f"Erro ao segmentar pupila escura: {e}")

        return result_img, pupil_mask, pupil_info_list

    def segment_pupil_bright(self, brilhante_gray_no_glints, roi_mask, frame_original):
        pupil_mask = np.zeros_like(brilhante_gray_no_glints)
        result_img = frame_original.copy()
        pupil_info_list = []

        try:
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

            if np.max(markers) > 1:
                markers_final = cv2.watershed(frame_original.copy(), markers.copy())
                mask_all = np.zeros_like(brilhante_gray_no_glints)
                mask_all[markers_final > 1] = 255
                pupil_mask = mask_all.copy()

                contours, _ = cv2.findContours(mask_all, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if 40 < area < 7000:
                        perimeter = cv2.arcLength(cnt, True)
                        if perimeter > 0:
                            circularity = (4 * np.pi * area) / (perimeter ** 2)
                            if circularity > 0.30:
                                points = cnt.squeeze()
                                if points.ndim == 1 or points.shape[0] < 3:
                                    continue
                                cx, cy, r = CircleFitter.fit_circle(points)
                                if cx is not None:
                                    center = (int(cx), int(cy))
                                    cv2.drawContours(result_img, [cnt], -1, (0, 255, 0), 2)
                                    pupil_info_list.append({
                                        'center': center,
                                        'radius': r,
                                        'segmentada': True,
                                        'contour': cnt,
                                        'fit_circle': (cx, cy, r)
                                    })
        except Exception as e:
            print(f"Erro ao segmentar pupila brilhante: {e}")

        return result_img, pupil_mask, pupil_info_list

    def segment_pair(self, frame_escura, frame_brilhante, par_idx):
        if frame_escura is None or frame_brilhante is None:
            return None

        escura_gray = cv2.cvtColor(frame_escura, cv2.COLOR_BGR2GRAY)
        brilhante_gray = cv2.cvtColor(frame_brilhante, cv2.COLOR_BGR2GRAY)

        roi_mask, diff = self.refine_roi(escura_gray, brilhante_gray)

        if roi_mask is None or not np.any(roi_mask):
            return None

        glints_escura = GlintDetector.detect_glints(escura_gray, roi_mask, is_dark_pupil_frame=True)
        glints_brilhante = GlintDetector.detect_glints(brilhante_gray, roi_mask, is_dark_pupil_frame=False)

        escura_no_glints = self.remove_glints_from_image(escura_gray, glints_escura)
        brilhante_no_glints = self.remove_glints_from_image(brilhante_gray, glints_brilhante)

        # Segmentar pupilas
        escura_resultado, pupil_mask_escura, pupil_escura_info = self.segment_pupil_dark(
            escura_no_glints, roi_mask, frame_escura
        )

        brilhante_resultado, pupil_mask_brilhante, pupil_brilhante_info = self.segment_pupil_bright(
            brilhante_no_glints, roi_mask, frame_brilhante
        )

        # Dibujar glints en imágenes resultantes
        for g in glints_escura:
            cv2.circle(escura_resultado, g['center'], 5, (0, 0, 255), -1)
        for g in glints_brilhante:
            cv2.circle(brilhante_resultado, g['center'], 3, (0, 255, 255), -1)

        return {
            'escura_gray': escura_gray,
            'brilhante_gray': brilhante_gray,
            'diferenca': diff,
            'roi_mask': roi_mask,
            'resultado_escura_borda': escura_resultado,
            'pupilas_escuras': pupil_escura_info,
            'glints_escura_frame': glints_escura,
            'pupil_mask_escura': pupil_mask_escura,
            'resultado_brilhante_borda': brilhante_resultado,
            'pupilas_brillantes': pupil_brilhante_info,
            'glints_brilhante_frame': glints_brilhante,
            'pupil_mask_brilhante': pupil_mask_brilhante,
        }

class PupilTrackerApp:
    NORMAL = 1
    PUPILA_CLARA = 2
    PUPILA_ESCURA = 3
    DIFERENCA_IMAGENS = 4
    CONTORNO_CLARA = 5
    CONTORNO_ESCURA = 6

    def __init__(self, video_path, output_video_path, max_frames_buffer=2):
        self.video_path = video_path
        self.output_video_path = output_video_path
        self.max_frames_buffer = max_frames_buffer
        self.current_display_mode = self.NORMAL
        self.mode_names = {
            self.NORMAL: "MODO: Normal (segmentación y glints)",
            self.PUPILA_CLARA: "MODO: Somente Frames de Pupila Clara (FILTRADO)",
            self.PUPILA_ESCURA: "MODO: Somente Frames de Pupila Escura (FILTRADO)",
            self.DIFERENCA_IMAGENS: "MODO: Máscara ROI (Diferença Binarizada - SEM MARCADORES)",
            self.CONTORNO_CLARA: "MODO: Contorno Pupila Clara (FILTRADO)",
            self.CONTORNO_ESCURA: "MODO: Contorno Pupila Escura (FILTRADO)",
        }
        self.loop_video = True
        self.paused = False
        self.frame_to_process = None
        self.segmenter = PupilSegmenter()

    def processar_video_pupilas(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"Erro ao abrir o vídeo: {self.video_path}")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_video_path, fourcc, fps, (frame_width, frame_height))

        if not out.isOpened():
            print(f"Erro ao criar o vídeo de saída: {self.output_video_path}")
            cap.release()
            return

        frame_buffer = collections.deque(maxlen=self.max_frames_buffer)
        frame_idx = 0
        pares_segmentados_count = 0

        all_pupil_centers_data = []
        all_glints_data = []

        ret, self.frame_to_process = cap.read()
        if not ret:
            print("No se pudo leer el primer frame. El video podría estar vacío o corrupto.")
            cap.release()
            out.release()
            return

        print(f"Processando vídeo: {self.video_path}")
        print(f"Dimensões: {frame_width}x{frame_height}, FPS: {fps}, Total de Frames: {total_frames}")
        print("\n--- Menu de Opções ---")
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
        print("----------------------\n")

        while True:
            wait_delay = 0 if self.paused else 1
            key = cv2.waitKey(wait_delay) & 0xFF

            if key == ord('q'):
                break
            elif key in (ord('l'), ord('L')):
                self.loop_video = not self.loop_video
                print(f"Modo de bucle: {'Activado' if self.loop_video else 'Desactivado'}")
            elif key in (ord('p'), ord('P')):
                self.paused = not self.paused
                print(f"Reproducción: {'Pausada' if self.paused else 'Reproduciendo'}")
            elif key in (ord('n'), ord('N')):
                if self.paused:
                    ret, self.frame_to_process = cap.read()
                    if not ret:
                        if self.loop_video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            frame_idx = 0
                            frame_buffer.clear()
                            ret, self.frame_to_process = cap.read()
                            if not ret:
                                break
                        else:
                            break
                    frame_idx += 1
            elif key >= ord('1') and key <= ord('6'):
                new_mode = int(chr(key))
                if new_mode in self.mode_names:
                    self.current_display_mode = new_mode
                    print(f"Cambiando a {self.mode_names[self.current_display_mode]}")

            if not self.paused and key not in (ord('n'), ord('N')):
                ret, self.frame_to_process = cap.read()
                if not ret:
                    if self.loop_video:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        frame_buffer.clear()
                        ret, self.frame_to_process = cap.read()
                        if not ret:
                            break
                    else:
                        break
                frame_idx += 1

            if self.frame_to_process is None:
                break

            frame_gray = cv2.cvtColor(self.frame_to_process, cv2.COLOR_BGR2GRAY)
            tipo_pupila, _ = PupilTypeClassifier.classify_by_bright_pixels(
                frame_gray, MIN_BRIGHT_PIXEL_VALUE, MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL
            )

            frame_info = {
                'frame_num': frame_idx,
                'frame_data': self.frame_to_process,
                'tipo': tipo_pupila,
                'gray_data': frame_gray
            }
            frame_buffer.append(frame_info)

            current_frame_to_write = None
            segmentation_results = None

            pupil_centers_current_frame = {'frame_num': frame_idx, 'pupilas': []}
            glints_current_frame = {'frame_num': frame_idx, 'glints': []}

            process_pair = False
            frame_to_display_from_pair = None
            current_frame_type_for_display = 'desconhecido'

            if len(frame_buffer) == 2:
                f1, f2 = frame_buffer[0], frame_buffer[1]
                valid_pair = (
                    (f1['tipo'] == 'escura' and f2['tipo'] == 'brilhante') or
                    (f1['tipo'] == 'brilhante' and f2['tipo'] == 'escura')
                )
                if valid_pair:
                    escura_frame_data = f1['frame_data'] if f1['tipo'] == 'escura' else f2['frame_data']
                    brilhante_frame_data = f1['frame_data'] if f1['tipo'] == 'brilhante' else f2['frame_data']
                    process_pair = True
                    frame_to_display_from_pair = f2['frame_data']
                    current_frame_type_for_display = f2['tipo']
                else:
                    frame_to_display_from_pair = f2['frame_data']
                    current_frame_type_for_display = f2['tipo']
            else:
                frame_to_display_from_pair = frame_info['frame_data']
                current_frame_type_for_display = frame_info['tipo']
            
            if process_pair:
                pares_segmentados_count += 1
                segmentation_results = self.segmenter.segment_pair(
                    escura_frame_data, brilhante_frame_data, pares_segmentados_count
                )
                if segmentation_results:
                    for pupil in segmentation_results['pupilas_brillantes']:
                        center = pupil['center']
                        pupil_centers_current_frame['pupilas'].append({
                            'tipo': 'brilhante',
                            'center_x': center[0],
                            'center_y': center[1]
                        })
                    for pupil in segmentation_results['pupilas_escuras']:
                        center = pupil['center']
                        pupil_centers_current_frame['pupilas'].append({
                            'tipo': 'escura',
                            'center_x': center[0],
                            'center_y': center[1]
                        })

                    for glint in segmentation_results['glints_brilhante_frame']:
                        glints_current_frame['glints'].append({
                            'tipo': 'glint_brilhante',
                            'center_x': glint['center'][0],
                            'center_y': glint['center'][1]
                        })
                    for glint in segmentation_results['glints_escura_frame']:
                        glints_current_frame['glints'].append({
                            'tipo': 'glint_escura',
                            'center_x': glint['center'][0],
                            'center_y': glint['center'][1]
                        })
            
            if self.current_display_mode == self.NORMAL:
                if process_pair and segmentation_results:
                    if current_frame_type_for_display == 'brilhante':
                        current_frame_to_write = segmentation_results['resultado_brilhante_borda']
                    elif current_frame_type_for_display == 'escura':
                        current_frame_to_write = segmentation_results['resultado_escura_borda']
                    else:
                        current_frame_to_write = frame_to_display_from_pair.copy()

                    if current_frame_to_write is not None:
                        for pupil in pupil_centers_current_frame['pupilas']:
                            color = (255, 0, 255) if pupil['tipo'] == 'brilhante' else (0, 0, 255)
                            cv2.circle(current_frame_to_write, (pupil['center_x'], pupil['center_y']), 3, color, -1)
                else:
                    current_frame_to_write = frame_to_display_from_pair.copy()
                    gray = cv2.cvtColor(frame_to_display_from_pair, cv2.COLOR_BGR2GRAY)
                    glints_info = []
                    if current_frame_type_for_display == 'escura':
                        glints_info = GlintDetector.detect_glints(gray, is_dark_pupil_frame=True)
                    elif current_frame_type_for_display == 'brilhante':
                        glints_info = GlintDetector.detect_glints(gray, is_dark_pupil_frame=False)
                    for glint in glints_info:
                        color = (0, 0, 255) if current_frame_type_for_display == 'escura' else (0, 255, 255)
                        radius = 5 if current_frame_type_for_display == 'escura' else 3
                        cv2.circle(current_frame_to_write, glint['center'], radius, color, -1)

            elif self.current_display_mode == self.PUPILA_CLARA:
                if current_frame_type_for_display == 'brilhante':
                    current_frame_to_write = frame_to_display_from_pair.copy()
                else:
                    current_frame_to_write = None

            elif self.current_display_mode == self.PUPILA_ESCURA:
                if current_frame_type_for_display == 'escura':
                    current_frame_to_write = frame_to_display_from_pair.copy()
                else:
                    current_frame_to_write = None

            elif self.current_display_mode == self.DIFERENCA_IMAGENS:
                if process_pair and segmentation_results and segmentation_results['roi_mask'] is not None and np.any(segmentation_results['roi_mask']):
                    roi_mask = segmentation_results['roi_mask']
                    current_frame_to_write = cv2.cvtColor(roi_mask, cv2.COLOR_GRAY2BGR)
                else:
                    current_frame_to_write = None

            elif self.current_display_mode == self.CONTORNO_CLARA:
                if current_frame_type_for_display == 'brilhante' and segmentation_results and segmentation_results['pupilas_brillantes']:
                    current_frame_to_write = frame_to_display_from_pair.copy()
                    for pupil_info in segmentation_results['pupilas_brillantes']:
                        fit = pupil_info['fit_circle']
                        if fit:
                            center = (int(fit[0]), int(fit[1]))
                            radius = int(fit[2])
                            cv2.circle(current_frame_to_write, center, radius, (0, 255, 0), 2)
                            cv2.circle(current_frame_to_write, center, 3, (0, 255, 0), -1)
                else:
                    current_frame_to_write = None

            elif self.current_display_mode == self.CONTORNO_ESCURA:
                if current_frame_type_for_display == 'escura' and segmentation_results and segmentation_results['pupilas_escuras']:
                    current_frame_to_write = frame_to_display_from_pair.copy()
                    for pupil_info in segmentation_results['pupilas_escuras']:
                        fit = pupil_info['fit_circle']
                        if fit:
                            center = (int(fit[0]), int(fit[1]))
                            radius = int(fit[2])
                            cv2.circle(current_frame_to_write, center, radius, (255, 0, 0), 2)
                            cv2.circle(current_frame_to_write, center, 3, (255, 0, 0), -1)
                else:
                    current_frame_to_write = None

            if current_frame_to_write is None:
                continue
            
            if pupil_centers_current_frame['pupilas']:
                all_pupil_centers_data.append(pupil_centers_current_frame)
            if glints_current_frame['glints']:
                all_glints_data.append(glints_current_frame)
            
            mode_text = self.mode_names.get(self.current_display_mode, "MODO: Desconocido")
            loop_status = "ON" if self.loop_video else "OFF"
            pause_status = "PAUSADO" if self.paused else "REPRODUCIENDO"
            status_text = f"{mode_text} | Loop: {loop_status} | {pause_status}"

            cv2.putText(current_frame_to_write, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(current_frame_to_write, "Q: Sair | 1-6: Mudar Modo | L: Loop | P: Pausa | N: Avancar Frame",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("Pupil Segmentation", current_frame_to_write)
            out.write(current_frame_to_write)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        self.save_data(all_pupil_centers_data, all_glints_data)
        print(f"Vídeo de saída salvo em: {self.output_video_path}")

    @staticmethod
    def save_data(pupil_centers_data, glints_data):
        output_pupil_centers_path = os.path.join(os.getcwd(), "pupil_centers_data.txt")
        with open(output_pupil_centers_path, 'w') as f:
            f.write("Dados dos Centros de Pupilas Segmentadas:\n")
            for frame_data in pupil_centers_data:
                f.write(f"Frame {frame_data['frame_num']}:\n")
                for pupil in frame_data['pupilas']:
                    f.write(f"     Tipo: {pupil['tipo']}, Centro (x,y): ({pupil['center_x']}, {pupil['center_y']})\n")
                f.write("\n")
        print(f"Dados dos centros de pupilas salvos em: {output_pupil_centers_path}")

        output_glints_path = os.path.join(os.getcwd(), "glints_data.txt")
        with open(output_glints_path, 'w') as f:
            f.write("Dados dos Glints Detectados:\n")
            for frame_data in glints_data:
                f.write(f"Frame {frame_data['frame_num']}:\n")
                for glint in frame_data['glints']:
                    f.write(f"     Tipo: {glint['tipo']}, Centro (x,y): ({glint['center_x']}, {glint['center_y']})\n")
                f.write("\n")
        print(f"Dados dos glints salvos em: {output_glints_path}")

def main():
    video_path_input = 'video_prueba.avi'
    output_video_filename = 'video_pupila_segmentada_con_filtros.avi'
    output_video_path = os.path.join(os.getcwd(), output_video_filename)

    if not os.path.exists(video_path_input):
        print(f"Erro: O arquivo de vídeo de entrada '{video_path_input}' não foi encontrado.")
        return

    app = PupilTrackerApp(video_path_input, output_video_path)
    app.processar_video_pupilas()

if __name__ == "__main__":
    main()