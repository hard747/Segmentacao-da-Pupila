import cv2
import numpy as np
import time

# Parámetros generales
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080
BG_COLOR = (0, 0, 0)
GRID_COLOR = (255, 255, 255)   # Blanco
CROSS_COLOR = (0, 255, 0)      # Verde

CROSS_SIZE = 40                # Longitud de la cruz
THICKNESS_GRID = 1             # Grosor de las líneas de la cuadrícula
THICKNESS_CROSS = 2            # Grosor de las líneas de la cruz

# Configuración de cuadrículas
GRIDS = {
    '3x3': {'cols': 3, 'rows': 3},
    '5x7': {'cols': 5, 'rows': 7}
}

def truncate(f, n):
    """Trunca un float f a n decimales sin redondear"""
    return int(f * 10**n) / 10**n

def get_cell_centers(cols, rows, width, height, n_decimals=2):
    cell_w = width / cols
    cell_h = height / rows
    centers = []
    print(f"\nCoordenadas de los centros para {cols}x{rows} (truncadas a {n_decimals} decimales):")
    for row in range(rows):
        for col in range(cols):
            x_float = col * cell_w + cell_w / 2
            y_float = row * cell_h + cell_h / 2
            x_trunc = truncate(x_float, n_decimals)
            y_trunc = truncate(y_float, n_decimals)
            print(f"({x_trunc}, {y_trunc})")
            centers.append((x_trunc, y_trunc))
    return centers

def draw_grid(image, cols, rows, width, height):
    cell_w = width / cols
    cell_h = height / rows
    # Líneas verticales
    for i in range(1, cols):
        x = int(i * cell_w)
        cv2.line(image, (x, 0), (x, height), GRID_COLOR, THICKNESS_GRID)
    # Líneas horizontales
    for i in range(1, rows):
        y = int(i * cell_h)
        cv2.line(image, (0, y), (width, y), GRID_COLOR, THICKNESS_GRID)

def draw_cross(image, center, color, size):
    # Convertir coordenadas truncadas a enteros para dibujar
    x, y = int(center[0]), int(center[1])
    cv2.line(image, (x - size//2, y), (x + size//2, y), color, THICKNESS_CROSS)
    cv2.line(image, (x, y - size//2), (x, y + size//2), color, THICKNESS_CROSS)

def main():
    grid_key = '3x3'
    paused = False

    cv2.namedWindow("Grid Cross", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Grid Cross", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        cols = GRIDS[grid_key]['cols']
        rows = GRIDS[grid_key]['rows']
        centers = get_cell_centers(cols, rows, SCREEN_WIDTH, SCREEN_HEIGHT, n_decimals=2)
        target_idx = 0

        while target_idx < len(centers):
            start_time = time.time()
            while True:
                # Crear imagen base negra
                img = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
                draw_grid(img, cols, rows, SCREEN_WIDTH, SCREEN_HEIGHT)
                draw_cross(img, centers[target_idx], CROSS_COLOR, CROSS_SIZE)
                cv2.imshow("Grid Cross", img)

                key = cv2.waitKey(10)
                if key == 27:  # ESC para salir
                    cv2.destroyAllWindows()
                    return
                elif key == 32:  # ESPACIO para pausar/reanudar
                    paused = not paused
                    if paused:
                        print("PAUSADO - Presiona ESPACIO para continuar")
                    else:
                        print("Reanudado")
                elif key == 9:  # TAB para cambiar cuadrícula
                    grid_key = '5x7' if grid_key == '3x3' else '3x3'
                    break  # reiniciar ciclo con nueva cuadrícula

                if not paused and (time.time() - start_time >= 2.0):
                    break

            if key == 9:  # Si fue cambio de cuadrícula, reiniciar ciclo externo
                break
            target_idx += 1

        if target_idx == len(centers):  # Terminó la secuencia
            time.sleep(1)
            cv2.destroyAllWindows()
            return

if __name__ == "__main__":
    main()