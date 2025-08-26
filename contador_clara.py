# -*- coding: utf-8 -*-
import cv2
import numpy as np

# --- Configuración ---
#VIDEO_PATH = "pupila_video1_30hz.avi" # Asegúrate de que esta ruta sea correcta
#VIDEO_PATH = "gravacao5.avi" # Asegúrate de que esta ruta sea correcta
VIDEO_PATH = "grabacion_1753302530.avi" # Asegúrate de que esta ruta sea correcta
#VIDEO_PATH = "grabacion_prueba.avi"   

# --- Umbrales para la clasificación ---
# Este es el valor MÍNIMO de intensidad que un PÍXEL debe tener para ser considerado "muy brillante".
MIN_BRIGHT_PIXEL_VALUE = 200 # <--- Ajusta este valor (ej. 230, 240, 250) según tu video.
                             # Debe ser un valor que solo aparezca en pupilas claras (sin contar glints).

# Este es el NÚMERO MÍNIMO de píxeles que deben alcanzar MIN_BRIGHT_PIXEL_VALUE
# para que el frame sea clasificado como "Pupila Clara".
# Sirve para ignorar pequeños reflejos (glints) o ruido.
MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL = 40 # <--- ¡AJUSTA ESTE VALOR! Prueba con 50, 100, 200, etc.

# --- Función principal de análisis ---
def analyze_pupil_states_full_frame_simplified(video_path, min_bright_pixel_val, min_pixel_count):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {video_path}")
        return

    bright_pupil_frames = 0
    dark_pupil_frames = 0
    total_frames = 0

    print("\n--- Analizando frames del video con Clasificación Mejorada ---")
    print("¡METODOLOGÍA DEL PROFESOR: PAUSA CUADRO A CUADRO!")
    print(f"Clasificación: 'Pupila Clara' si al menos {min_pixel_count} píxeles tienen intensidad >= {min_bright_pixel_val},")
    print("                'Pupila Oscura' en cualquier otro caso.")
    print("Presiona CUALQUIER TECLA para avanzar al siguiente frame.")
    print("Presiona 'q' para salir del análisis en cualquier momento.")
    print("--------------------------------------------------------------------")

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Fin del video

        total_frames += 1

        # Convertir el frame completo a escala de grises
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculamos el histograma para contar cuántos píxeles están en el rango de alto brillo.
        hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
        
        # Sumamos los píxeles que tienen una intensidad igual o mayor a MIN_BRIGHT_PIXEL_VALUE
        # (desde MIN_BRIGHT_PIXEL_VALUE hasta 255).
        bright_pixels_in_range_count = np.sum(hist[min_bright_pixel_val : 256])

        # Clasificación binaria con la nueva condición de conteo de píxeles:
        classification = "Pupila Oscura" # Por defecto
        color = (0, 0, 255) # Rojo (para Pupila Oscura)

        # Si el CONTEO de píxeles muy brillantes es mayor o igual al mínimo requerido
        if bright_pixels_in_range_count >= min_pixel_count:
            bright_pupil_frames += 1
            classification = "Pupila Clara"
            color = (0, 255, 0) # Verde (para Pupila Clara)
        else:
            dark_pupil_frames += 1 # Si no se cumple el mínimo de píxeles brillantes, es Pupila Oscura

        # Mostrar el frame con la información actualizada
        display_frame = frame.copy()
        cv2.putText(display_frame, f"Frame: {total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        #cv2.putText(display_frame, f"Clasificacion: {classification}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        #cv2.putText(display_frame, f"Píxeles >= {min_bright_pixel_val}: {int(bright_pixels_in_range_count)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2) # Muestra el conteo de píxeles muy brillantes.
        
        cv2.imshow('Analizando Video (Clasificacion Mejorada)', display_frame)

        # PAUSA CUADRO A CUADRO: Espera una tecla para continuar
        key = cv2.waitKey(0) & 0xFF 
        if key == ord('q'):
            print("Análisis interrumpido por el usuario.")
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n--- Resultados Finales del Análisis de Clasificación Mejorada ---")
    print(f"Video analizado: {video_path}")
    print(f"Total de frames procesados: {total_frames}")
    print(f"Frames con Pupila Clara: {bright_pupil_frames}")
    print(f"Frames con Pupila Oscura: {dark_pupil_frames}")
    print("\n--- NOTA IMPORTANTE: Ajusta los Umbrales ---")
    print(f"El valor de 'MIN_BRIGHT_PIXEL_VALUE' (actualmente {MIN_BRIGHT_PIXEL_VALUE})")
    print(f"y 'MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL' (actualmente {MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL}) son CRÍTICOS.")
    print("Debes ajustarlos observando los frames y el conteo de píxeles en pantalla,")
    print("para encontrar los valores que mejor discriminen entre pupila clara y oscura en tu video.")


if __name__ == "__main__":
    analyze_pupil_states_full_frame_simplified(VIDEO_PATH, MIN_BRIGHT_PIXEL_VALUE, MIN_PIXEL_COUNT_FOR_CLEAR_PUPIL)
