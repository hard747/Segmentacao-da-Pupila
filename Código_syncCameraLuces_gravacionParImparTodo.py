# -*- coding: utf-8 -*-
import cv2
import time
import sys
import serial # Importa la librería pyserial
import select # Importa el módulo select para entrada no bloqueante (Linux/macOS)
import atexit # Para asegurar la limpieza al salir

# --- Configuración de la Cámara ---
PS3_EYE_VENDOR_ID = '1415'
PS3_EYE_MODEL_ID = '2000'
TARGET_FPS = 30 # FPS objetivo para la cámara (ajustado según tu última indicación)

# --- Configuración Serial del Arduino ---
# !!! IMPORTANTE: REEMPLAZA ESTO CON EL PUERTO SERIAL REAL DE TU ARDUINO !!!
# Ejemplos: '/dev/ttyACM0' (Linux), 'COM3' (Windows), '/dev/tty.usbmodemXXXX' (macOS)
ARDUINO_PORT = '/dev/ttyACM0'
BAUD_RATE = 115200 # Debe coincidir con la velocidad de Serial.begin() en tu Arduino

# --- Variables Globales ---
cap = None
frame_count = 0
ser = None # Objeto para la comunicación serial con Arduino
last_even_frame = None # Almacena el último frame par capturado
last_odd_frame = None  # Almacena el último frame impar capturado

# --- Variables para Grabación de Video ---
is_recording = False
recording_start_time = 0
video_writer_even = None    # Para frames pares
video_writer_odd = None     # Para frames impares
video_writer_all = None     # Para todos los frames (pares + impares)
RECORDING_DURATION_SEC = 10 # Duración del video a ser grabado en segundos
frame_width = 0
frame_height = 0
# --- Fin de Variables de Grabación ---

# --- Funciones de Ayuda ---

def get_ps3_eye_camera_id(vendor_id, model_id):
    """
    Encuentra el índice numérico (ID) de la cámara PS3 Eye por su ID de Vendedor y Modelo usando pyudev.
    Devuelve el ID numérico de OpenCV (ej., 0, 1) o None.
    """
    try:
        import pyudev
        context = pyudev.Context()
        for device in context.list_devices(subsystem='video4linux'):
            properties = device.properties
            if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
                if properties['ID_VENDOR_ID'] == vendor_id and properties['ID_MODEL_ID'] == model_id:
                    try:
                        camera_index = int(device.device_node.replace('/dev/video', ''))
                        print(f"Cámara PS3 Eye encontrada en: {device.device_node} (OpenCV ID: {camera_index})")
                        return camera_index
                    except ValueError:
                        print(f"Advertencia: No se pudo parsear el ID numérico de {device.device_node}.")
                        return None
        print(f"Dispositivo PS3 Eye (Vendor:{vendor_id}, Modelo:{model_id}) no encontrado usando pyudev.")
        return None
    except ImportError:
        print("La librería 'pyudev' no está instalada. No se puede buscar la cámara por ID de Vendor/Model.")
        print("Instálala con: pip install pyudev")
        return None
    except Exception as e:
        print(f"Error al buscar dispositivos udev: {e}")
        print("Asegúrate de tener permisos para acceder a /dev/ (ej. sudo apt install udev, agregar usuario a grupo video).")
        return None

def init_camera_and_serial():
    """Inicializa la cámara PS3 Eye y la conexión serial con Arduino."""
    global cap, ser, frame_width, frame_height
    print("--- Inicializando Cámara PS3 Eye ---")
    camera_id = get_ps3_eye_camera_id(PS3_EYE_VENDOR_ID, PS3_EYE_MODEL_ID)

    if camera_id is None:
        print("Error: Cámara PS3 Eye no encontrada o ID no disponible.")
        return False

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir la cámara con OpenCV (ID: {camera_id}).")
        return False
    print(f"Cámara abierta con OpenCV (ID: {camera_id}).")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    print(f"Intentando configurar la cámara a {TARGET_FPS} FPS...")
    time.sleep(1)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS real de la cámara medido: {actual_fps:.2f} FPS (puede variar de lo solicitado).")
    
    # Obtener dimensiones del frame para la grabación
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolución de la cámara: {frame_width}x{frame_height}")

    print("\n--- Inicializando conexión Serial con Arduino ---")
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Da tiempo para que el Arduino se resetee después de la conexión serial
        print(f"Conexión serial establecida con Arduino en {ARDUINO_PORT}.")
        
        # --- OPCIONAL: Elimina o comenta esta línea si quieres que el Arduino mantenga valores anteriores ---
        # --- Si la dejas, al iniciar Python SIEMPRE fijará la duración a 100ms ---
        # ser.write(b'S100000\n') # Establece la duración del estrobo a 100ms para depuración inicial
        # print("Enviado comando 'S100000' al Arduino para duración de estrobo (100ms) al inicio.")
        # --------------------------------------------------------------------------------------------------

    except serial.SerialException as e:
        print(f"Error: No se pudo establecer conexión serial con Arduino en {ARDUINO_PORT}. {e}")
        if cap:
            cap.release()
        return False
    
    return True

def get_camera_frame_and_trigger(current_frame_count):
    """
    Captura un frame de la cámara y envía el comando de estrobo apropiado al Arduino.
    'B' para estrobo de Pupila Brillante o 'D' para estrobo de Pupila Oscura.
    """
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if ser and ser.is_open:
                try:
                    if current_frame_count % 2 == 0:
                        ser.write(b'B\n')
                    else:
                        ser.write(b'D\n')
                except serial.SerialException as e:
                    print(f"Error al enviar trigger a Arduino: {e}")
            return ret, frame
    return False, None

def cleanup():
    """
    Libera los recursos de la cámara y serial, y cierra las ventanas de OpenCV al salir del programa.
    """
    global cap, ser, video_writer_even, video_writer_odd, video_writer_all
    print("\n--- Realizando limpieza al salir ---")

    if cap and cap.isOpened():
        cap.release()
        print("Cámara liberada (OpenCV).")
    else:
        print("Cámara no estaba abierta o ya se liberó.")

    if ser and ser.is_open:
        ser.close()
        print("Puerto serial de Arduino cerrado.")
    else:
        print("Puerto serial no estaba abierto o ya se cerró.")

    # Liberar grabadores de video si están abiertos
    if video_writer_even is not None and video_writer_even.isOpened():
        video_writer_even.release()
        print("Grabador de video (frames pares) liberado.")
    elif video_writer_even is not None:
        print("Advertencia: Grabador de video (frames pares) no estaba abierto o ya se liberó.")

    if video_writer_odd is not None and video_writer_odd.isOpened():
        video_writer_odd.release()
        print("Grabador de video (frames impares) liberado.")
    elif video_writer_odd is not None:
        print("Advertencia: Grabador de video (frames impares) no estaba abierto o ya se liberó.")
        
    if video_writer_all is not None and video_writer_all.isOpened():
        video_writer_all.release()
        print("Grabador de video (todos los frames) liberado.")
    elif video_writer_all is not None:
        print("Advertencia: Grabador de video (todos los frames) no estaba abierto o ya se liberó.")

    cv2.destroyAllWindows()
    print("Ventanas de OpenCV cerradas.")
    print("--- Limpieza completa ---")

# Register the cleanup function to be called automatically on program exit
atexit.register(cleanup)

# --- Lógica Principal del Programa ---

def main():
    global frame_count, is_recording, recording_start_time, video_writer_even, video_writer_odd, video_writer_all
    global last_even_frame, last_odd_frame # Declarar estas variables como globales aquí también

    camera_and_serial_ready = init_camera_and_serial()

    if camera_and_serial_ready:
        print("\n--- Captura de frames iniciada. Presiona 'q' para salir. ---")
        print("--- Para ajustar parámetros en tiempo real, escribe 'S<valor>' (duración) o 'P<valor>' (retraso) en esta consola y presiona Enter. ---")
        print("--- Para grabar TRES videos (pares, impares, y todos) de 10 segundos, escribe 'r' y presiona Enter. ---")
        print("--- Enviando trigger 'B' (Pupila Brillante) o 'D' (Pupila Oscura) al Arduino con cada nuevo frame. ---")
        try:
            while True:
                # --- SECCIÓN PARA LEER ENTRADA DE TECLADO EN CONSOLA (NO BLOQUEANTE) ---
                if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                    line = sys.stdin.readline().strip()
                    if line: # Si se ingresó una línea
                        if line.startswith('S') or line.startswith('P'):
                            try:
                                command_type = line[0]
                                value = int(line[1:]) 

                                if (command_type == 'S' and value > 0) or \
                                   (command_type == 'P' and value >= 0):
                                    command_to_send = (line + '\n').encode('utf-8')
                                    ser.write(command_to_send)
                                    print(f"Comando '{line}' enviado al Arduino.")
                                else:
                                    print(f"Valor inválido. Para 'S' > 0, para 'P' >= 0. Comando: '{line}'")
                            except ValueError:
                                print(f"Formato de comando inválido: '{line}'. Usa 'S<valor>' o 'P<valor>'.")
                        elif line == 'r': # Funcionalidad: Grabar video
                            if not is_recording:
                                # Obtener el FPS real de la cámara (si es diferente del TARGET_FPS configurado)
                                fps_for_recording = cap.get(cv2.CAP_PROP_FPS)
                                if fps_for_recording == 0: # En caso de que no reporte bien, usar el TARGET_FPS
                                    fps_for_recording = TARGET_FPS

                                # Nombres de archivo para los tres videos
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                filename_even = f"grabacion_pares_{timestamp}.avi"
                                filename_odd = f"grabacion_impares_{timestamp}.avi"
                                filename_all = f"grabacion_todos_{timestamp}.avi" # Nuevo archivo para todos los frames
                                
                                # Codec: MJPG es compatible en muchos sistemas y crea archivos .avi
                                fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
                                
                                try:
                                    # Inicializar grabadores de video
                                    video_writer_even = cv2.VideoWriter(filename_even, fourcc, fps_for_recording, (frame_width, frame_height))
                                    video_writer_odd = cv2.VideoWriter(filename_odd, fourcc, fps_for_recording, (frame_width, frame_height))
                                    video_writer_all = cv2.VideoWriter(filename_all, fourcc, fps_for_recording, (frame_width, frame_height)) # Inicializar el tercer grabador

                                    if video_writer_even.isOpened() and video_writer_odd.isOpened() and video_writer_all.isOpened():
                                        is_recording = True
                                        recording_start_time = time.time()
                                        print(f"Iniciando grabación de frames PARES en '{filename_even}' por {RECORDING_DURATION_SEC} segundos...")
                                        print(f"Iniciando grabación de frames IMPARES en '{filename_odd}' por {RECORDING_DURATION_SEC} segundos...")
                                        print(f"Iniciando grabación de TODOS los frames en '{filename_all}' por {RECORDING_DURATION_SEC} segundos...")
                                    else:
                                        print("Error: No se pudo iniciar la grabación de video para uno o más archivos. Asegúrate de tener los codecs necesarios (ej. ffmpeg) instalados.")
                                        # Liberar cualquier grabador que se haya abierto parcialmente
                                        if video_writer_even and video_writer_even.isOpened(): video_writer_even.release()
                                        if video_writer_odd and video_writer_odd.isOpened(): video_writer_odd.release()
                                        if video_writer_all and video_writer_all.isOpened(): video_writer_all.release()
                                        video_writer_even = None
                                        video_writer_odd = None
                                        video_writer_all = None
                                except Exception as e:
                                    print(f"Error al configurar VideoWriter: {e}")
                                    # Asegurarse de que los grabadores se cierren si hubo un error
                                    if video_writer_even and video_writer_even.isOpened(): video_writer_even.release()
                                    if video_writer_odd and video_writer_odd.isOpened(): video_writer_odd.release()
                                    if video_writer_all and video_writer_all.isOpened(): video_writer_all.release()
                                    video_writer_even = None
                                    video_writer_odd = None
                                    video_writer_all = None
                            else:
                                print("Ya se está grabando un video. Espera a que termine o reinicia.")
                        else:
                            print(f"Comando desconocido en consola: '{line}'")
                # --- FIN DE LA SECCIÓN DE ENTRAD A DE TECLADO ---

                ret, frame = get_camera_frame_and_trigger(frame_count)
                if not ret:
                    print("Error: No se pudo leer el frame. Posiblemente la cámara se desconectó.")
                    break

                # --- Mostrar las tres ventanas ---
                # La ventana 'Todos los Frames' siempre muestra el frame actual
                cv2.imshow('Todos los Frames', frame)

                if frame_count % 2 == 0:
                    # Si el frame es par, actualizamos la ventana de Pupila Brillante con el frame actual
                    cv2.imshow('Pupila Brillante (Frame Par - Strobe A4 ON)', frame)
                    # Y almacenamos una copia de este frame par
                    last_even_frame = frame.copy()
                    # Si ya tenemos un frame impar anterior, lo mostramos en la ventana de Pupila Oscura
                    # para mantener la última visualización de un frame impar.
                    if last_odd_frame is not None:
                        cv2.imshow('Pupila Oscura (Frame Impar - Strobe A5 ON)', last_odd_frame)
                else:
                    # Si el frame es impar, actualizamos la ventana de Pupila Oscura con el frame actual
                    cv2.imshow('Pupila Oscura (Frame Impar - Strobe A5 ON)', frame)
                    # Y almacenamos una copia de este frame impar
                    last_odd_frame = frame.copy()
                    # Si ya tenemos un frame par anterior, lo mostramos en la ventana de Pupila Brillante
                    # para mantener la última visualización de un frame par.
                    if last_even_frame is not None:
                        cv2.imshow('Pupila Brillante (Frame Par - Strobe A4 ON)', last_even_frame)

                # --- Lógica de Grabación de Video (se mantiene sin cambios) ---
                if is_recording:
                    if time.time() - recording_start_time < RECORDING_DURATION_SEC:
                        # Grabar en el video de todos los frames (siempre)
                        if video_writer_all is not None and video_writer_all.isOpened():
                            video_writer_all.write(frame)
                        else:
                            print("Advertencia: Grabador de TODOS los frames no está abierto. Deteniendo grabación.")
                            is_recording = False # Detener si un grabador falla
                            
                        # Grabar en el video de frames pares o impares según corresponda
                        if frame_count % 2 == 0:
                            if video_writer_even is not None and video_writer_even.isOpened():
                                video_writer_even.write(frame)
                            else:
                                print("Advertencia: Grabador de frames PARES no está abierto. Deteniendo grabación.")
                                is_recording = False
                        else: # Frame impar
                            if video_writer_odd is not None and video_writer_odd.isOpened():
                                video_writer_odd.write(frame)
                            else:
                                print("Advertencia: Grabador de frames IMPARES no está abierto. Deteniendo grabación.")
                                is_recording = False
                    else: # Duración de grabación terminada
                        if video_writer_even is not None:
                            video_writer_even.release()
                            print(f"Grabación de frames PARES terminada después de {RECORDING_DURATION_SEC} segundos.")
                        if video_writer_odd is not None:
                            video_writer_odd.release()
                            print(f"Grabación de frames IMPARES terminada después de {RECORDING_DURATION_SEC} segundos.")
                        if video_writer_all is not None:
                            video_writer_all.release()
                            print(f"Grabación de TODOS los frames terminada después de {RECORDING_DURATION_SEC} segundos.")
                        
                        is_recording = False
                        video_writer_even = None # Resetear los objetos para futuras grabaciones
                        video_writer_odd = None
                        video_writer_all = None
                # --- Fin Lógica de Grabación ---

                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Detención manual por el usuario.")
                    break

        except KeyboardInterrupt:
            print("Proceso interrumpido por el usuario (Ctrl+C).")
        finally:
            print(f"Visualización completada. Se procesaron {frame_count} frames.")
    else:
        print("--- Inicialización fallida (cámara o serial). No se iniciará la captura. ---")

if __name__ == "__main__":
    main()

