# -*- coding: utf-8 -*-
import cv2
import time
import sys
import serial
import select
import atexit
import pyudev
import numpy as np
from scipy.ndimage import convolve1d 

# --- Configuração da Câmera ---
PS3_EYE_VENDOR_ID = '1415'
PS3_EYE_MODEL_ID = '2000'
TARGET_FPS = 30 # FPS objetivo para a câmera

# --- Configuração Serial do Arduino ---
ARDUINO_PORT = '/dev/ttyACM0' # !!! IMPORTANTE: SUBSTITUA ISSO PELA PORTA SERIAL REAL DO SEU ARDUINO !!!
BAUD_RATE = 115200 # Deve coincidir com a velocidade de Serial.begin() no seu Arduino

# --- Variáveis Globais ---
cap = None # Objeto para captura do vídeo com a câmera
ser = None # Objeto para comunicação serial com Arduino
frame_width = 0
frame_height = 0
video_writer = None # Objeto para gravar vídeo

# --- Parâmetros de Calibração de Borsato ---
# Período mestre inicial do Arduino (clke) baseado em 30 FPS
# 1,000,000 us / 30 frames/s = 33333.33 us/frame
MASTER_PERIOD_INITIAL_US = 33333 

# Passos de ajuste para o período mestre (em microssegundos)
PERIOD_ADJUSTMENT_STEP_US = 10 # Pequeno ajuste para afinar (ex. 10 us)
CONVERGENCE_THRESHOLD_PX = 1 # Se o movimento médio da banda for menor que X pixels, consideramos estável
BAND_HISTORY_LENGTH = 10 # Quantos frames recentes usar para calcular o movimento da banda

# Parâmetros para a Etapa 3 (Mover a banda para fora da vista)
BIAS_FOR_SHIFT_US = -500 # Um viés para forçar o movimento da banda para baixo e fora da vista. (Voltamos a -500)
MAX_SHIFT_ATTEMPTS = 1000 # Número máximo de frames para esperar que a banda desapareça

# Tempo de espera após enviar um comando ao Arduino para que surta efeito
ARDUINO_WAIT_TIME_SEC = 0.1 # Pequena pausa para garantir que o Arduino processe o comando

# Duração e atraso inicial do estroboscópio para garantir uma banda detectável
INITIAL_STROBE_DURATION_US = 16000 # S16000 (Voltamos a 16000 para estabilidade)
INITIAL_STROBE_PRE_DELAY_US = 8000 # P8000

# --- NOVOS Parámetros de Debug Visual ---
DEBUG_VISUAL_MODE = True # Defina como False para desativar o debug visual y la pausa
PAUSE_BEFORE_STAGE_3 = True # Só é eficaz se DEBUG_VISUAL_MODE for True

# --- Parámetros para a Nova Etapa de Calibração Grossa ---
GROSS_CALIBRATION_DURATION_SEC = 3 # Duração em segundos para contar frames
MIN_FRAMES_FOR_GROSS_CALIBRATION = 60 # Mínimo de frames esperados em GROSS_CALIBRATION_DURATION_SEC

# --- NUEVA VARIABLE DE CONTROL PARA LA CALIBRACIÓN GRUESA ---
# Establece esto en 'True' para saltar la calibración gruesa y comenzar directamente en el ajuste fino.
# Establece en 'False' para que la calibración gruesa se ejecute normalmente.
SKIP_GROSS_CALIBRATION = True # <--- ¡CAMBIA ESTO PARA ALTERNAR!

# --- Estados do Processo de Calibração ---
CALIBRATION_STATE_INIT = 0           # Inicializando (enviando parâmetros base ao Arduino)
CALIBRATION_STATE_GROSS_CALIBRATION = 6 #Calibração grossa inicial por contagem de FPS
CALIBRATION_STATE_FINE_TUNING = 1    # Etapa 2: Ajustando o período mestre para estabilizar a banda
CALIBRATION_STATE_PAUSED = 5         # Novo estado: Pausado antes da Etapa 3
CALIBRATION_STATE_SHIFTING_BAND = 2  # Etapa 3: Movendo a banda para fora do campo visível
CALIBRATION_STATE_STABLE = 3         # Calibração concluída, sistema operando normalmente
CALIBRATION_STATE_ERROR = 4          # Ocorreu um erro durante a calibração
CALIBRATION_STATE_DETECTION_CHECK = 7 # NUEVO ESTADO: Verificar la detección de la banda

# current_calibration_state se inicializa en main() ahora
current_calibration_state = CALIBRATION_STATE_INIT # Valor por defecto, será sobreescrito

current_master_period_us_arduino = MASTER_PERIOD_INITIAL_US # Se inicializa aquí
band_history = [] # Lista para armazenar D_band de frames recentes
no_band_detection_count = 0 # Contador para frames consecutivos sem detecção de banda (para Etapa 3)

# Variáveis para a calibração grossa
gross_calibration_frame_count = 0
gross_calibration_start_time = 0

# --- Funções de Ajuda ---

def get_ps3_eye_camera_id(vendor_id, model_id):
    """
    Encontra o índice numérico (ID) da câmera PS3 Eye pelo seu ID de Vendedor e Modelo usando pyudev.
    Retorna o ID numérico do OpenCV (ex., 0, 1) ou None.
    """
    try:
        context = pyudev.Context()
        for device in context.list_devices(subsystem='video4linux'):
            properties = device.properties
            if 'ID_VENDOR_ID' in properties and 'ID_MODEL_ID' in properties:
                if properties['ID_VENDOR_ID'] == vendor_id and properties['ID_MODEL_ID'] == model_id:
                    try:
                        camera_index = int(device.device_node.replace('/dev/video', ''))
                        print(f"Câmera PS3 Eye encontrada em: {device.device_node} (OpenCV ID: {camera_index})")
                        return camera_index
                    except ValueError:
                        print(f"Aviso: Não foi possível analisar o ID numérico de {device.device_node}.")
                        return None
        print(f"Dispositivo PS3 Eye (Vendedor:{vendor_id}, Modelo:{model_id}) não encontrado usando pyudev.")
        return None
    except ImportError:
        print("A biblioteca 'pyudev' não está instalada. Não é possível procurar a câmera por ID de Vendedor/Modelo.")
        print("Instale-a com: pip install pyudev")
        return None
    except Exception as e:
        print(f"Erro ao procurar dispositivos udev: {e}")
        print("Certifique-se de ter permissões para acessar /dev/ (ex. sudo apt install udev, adicionar usuário ao grupo video).")
        return None

def init_camera_and_serial():
    """Inicializa a câmera PS3 Eye e a conexão serial com Arduino."""
    global cap, ser, frame_width, frame_height
    print("--- Inicializando Câmera PS3 Eye ---")
    camera_id = get_ps3_eye_camera_id(PS3_EYE_VENDOR_ID, PS3_EYE_MODEL_ID)

    if camera_id is None:
        print("Erro: Câmera PS3 Eye não encontrada ou ID não disponível.")
        return False

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Erro: Não foi possível abrir a câmera com OpenCV (ID: {camera_id}).")
        return False
    print(f"Câmera aberta com OpenCV (ID: {camera_id}).")

    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    print(f"Tentando configurar a câmera para {TARGET_FPS} FPS...")
    time.sleep(1)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS real da câmera medido: {actual_fps:.2f} FPS (pode variar do solicitado).")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Resolução da câmera: {frame_width}x{frame_height}")

    print("\n--- Inicializando conexão Serial com Arduino ---")
    try:
        ser = serial.Serial(ARDUINO_PORT, BAUD_RATE, timeout=1)
        time.sleep(2) # Dá tempo para o Arduino resetar após a conexão serial
        print(f"Conexão serial estabelecida com Arduino em {ARDUINO_PORT}.")
        
        # Enviar valores iniciais de duração e pré-atraso ao Arduino
        # O período será enviado na etapa correspondente.
        
        print(f"Enviando duração inicial S{INITIAL_STROBE_DURATION_US} ao Arduino...")
        ser.write(f'S{INITIAL_STROBE_DURATION_US}\n'.encode('utf-8'))
        time.sleep(ARDUINO_WAIT_TIME_SEC)
        
        print(f"Enviando pré-atraso inicial P{INITIAL_STROBE_PRE_DELAY_US} ao Arduino...")
        ser.write(f'P{INITIAL_STROBE_PRE_DELAY_US}\n'.encode('utf-8'))
        time.sleep(ARDUINO_WAIT_TIME_SEC)
        
    except serial.SerialException as e:
        print(f"Erro: Não foi possível estabelecer conexão serial com Arduino em {ARDUINO_PORT}. {e}")
        if cap:
            cap.release()
        return False
    
    return True

def cleanup():
    """
    Libera os recursos da câmera e serial, e fecha as janelas do OpenCV ao sair do programa.
    """
    global cap, ser, video_writer
    print("\n--- Realizando limpeza ao sair ---")

    if video_writer:
        video_writer.release()
        print("Gravação de vídeo finalizada e arquivo fechada.")

    if cap and cap.isOpened():
        cap.release()
        print("Câmera liberada (OpenCV).")
    else:
        print("Câmera não estava aberta ou já foi liberada.")

    if ser and ser.is_open:
        # Opcional: redefinir Arduino para um estado conhecido se necessário
        # Por exemplo, enviar um período muito longo para "parar" os pulsos, ou um valor M0 se o seu Arduino o suportar
        # ser.write(b'M10000000\n') # Definir um período de 10 segundos para parar os pulsos rápidos
        # time.sleep(ARDUINO_WAIT_TIME_SEC)
        ser.close()
        print("Porta serial do Arduino fechada.")
    else:
        print("Porta serial não estava aberta ou já foi fechada.")

    cv2.destroyAllWindows() # Cierra todas las ventanas de OpenCV
    print("Janelas do OpenCV fechadas.")
    print("--- Limpeza completa ---")

atexit.register(cleanup)

# --- Implementação do Algoritmo de Detecção de Banda (MODIFICADO) ---
def detect_illumination_band(image_frame):
    """
    Detecta a posição vertical (linha) da banda de luz (o la transición más fuerte)
    en una imagen de rolling shutter, utilizando un enfoque de detección de dos bordes.
    Retorna: (D_band_center, Dmax_idx, Dmin_idx) o (None, None, None) si falla o no hay banda significativa.
    """
    if image_frame.ndim != 2:
        image_frame = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Calcular el perfil de intensidad promedio por fila
    Ic = np.mean(image_frame.astype(float), axis=1) # Convertir a float para evitar overflow en np.mean

    # Kernel para detectar gradientes (transiciones de claro para escuro ou vice-versa)
    # Ajusta esto: prueba con [1, 0, -1], [-1, 0, 1] o incluso filtros más complejos como Laplaciano [1, -2, 1]
    kernel_size = 5 # Tamaño del kernel, debe ser impar para tener un centro
    if kernel_size % 2 == 0: # Garantizar que sea impar
        kernel_size += 1 
    K = np.array([-1] * (kernel_size // 2) + [0] + [1] * (kernel_size // 2), dtype=float)
    
    BRIGHT_CVAL = 255.0 

    # Aplicar convolución 1D al perfil Ic
    Iband = convolve1d(Ic, K, mode='constant', cval=0.0) 

    # --- Nueva Lógica de Detecção Melhorada ---
    # Si Iband es muy pequeño o plano (sin contraste significativo de una banda)
    # np.ptp() calcula el alcance (máximo - mínimo)
    # Aumentar este umbral para ignorar ruido o fondos uniformes.
    CONTRAST_THRESHOLD = 25 # Aumentado de 25 a 30. Ajusta este valor si es necesario.
    if len(Iband) < 2 or np.ptp(Iband) < CONTRAST_THRESHOLD: 
        return None, None, None

    Dmax_idx = np.argmax(Iband)
    Dmin_idx = np.argmin(Iband)

    # Asegurarse de que Dmin_idx sea el borde superior y Dmax_idx el inferior
    if Dmin_idx > Dmax_idx:
        Dmin_idx, Dmax_idx = Dmax_idx, Dmin_idx # Intercambiar

    # Validar que los índices estén dentro del rango de la imagen
    if not (0 <= Dmin_idx < len(Ic) and 0 <= Dmax_idx < len(Ic)):
        return None, None, None

    # --- Critério Adicional: Detecção de Banda nas Bordas Extremas ou Tela Uniforme ---
    # Solo se aplica este filtro en la etapa de SHIFTING_BAND para confirmar que la banda "desapareció".
    # En otras etapas, queremos detectar la banda aunque esté en los bordes.
    
    edge_margin = max(10, kernel_size // 2 + 5) # Margen dinámico.
    avg_image_intensity = np.mean(image_frame)
    BRIGHT_IMAGE_THRESHOLD = 180 
    is_at_extreme_edges = (Dmin_idx <= edge_margin or Dmax_idx >= len(Ic) - 1 - edge_margin)

    if is_at_extreme_edges and avg_image_intensity > BRIGHT_IMAGE_THRESHOLD and \
        current_calibration_state == CALIBRATION_STATE_SHIFTING_BAND: # <--- SOLO APLICAR EN SHIFTING_BAND
        return None, None, None # Consideramos que no hay banda significativa

    # El centro de la banda es estimado como el punto medio entre el gradiente máximo y mínimo.
    D_band_center = (Dmax_idx + Dmin_idx) * 0.5

    return D_band_center, Dmax_idx, Dmin_idx





# --- Lógica Principal do Programa ---

def main():
    global current_calibration_state, current_master_period_us_arduino, band_history, video_writer, no_band_detection_count
    global gross_calibration_frame_count, gross_calibration_start_time 

    calibration_start_time = time.time()
    
    stable_period_us = 0
    shift_attempts_count = 0
    band_disappeared = False

    is_recording = False
    recording_start_time = 0
    recording_duration_sec = 10 

    D_band = None
    Dmax_idx = None
    Dmin_idx = None

    frame_count = 0 # Inicializar el contador de frames para frames pares/impares
    
    # Variables para calcular FPS de frames pares e impares
    even_frame_count_for_fps = 0
    odd_frame_count_for_fps = 0
    last_even_fps_time = time.time()
    last_odd_fps_time = time.time()
    even_fps_display = 0.0
    odd_fps_display = 0.0


    camera_and_serial_ready = init_camera_and_serial()

    if not camera_and_serial_ready:
        print("--- Inicialização falhou. Saindo. ---")
        return

    # --- LÓGICA DE INICIO DE ESTADO SEGÚN SKIP_GROSS_CALIBRATION ---
    if SKIP_GROSS_CALIBRATION:
        current_calibration_state = CALIBRATION_STATE_DETECTION_CHECK
        print(f"Skipping Gross Calibration. Starting directly at Verification de Deteccao.")
        # Se envía el periodo inicial para que la banda sea visible
        print(f"Enviando período inicial M{MASTER_PERIOD_INITIAL_US} ao Arduino para visualização...")
        ser.write(f'M{MASTER_PERIOD_INITIAL_US}\n'.encode('utf-8'))
        time.sleep(ARDUINO_WAIT_TIME_SEC)
    else:
        current_calibration_state = CALIBRATION_STATE_INIT # Inicia el proceso normal de calibración
    # --- FIN LÓGICA DE INICIO DE ESTADO ---


    print("\n--- Iniciando calibração do relógio externo (Algoritmo de Borsato) ---")
    print("--- Pressione 'q' para sair a qualquer momento. ---")
    print("--- Pressione 'r' para gravar 10 segundos de vídeo. ---")
    if DEBUG_VISUAL_MODE:
        print("--- MODO DE DEBUG VISUAL ATIVADO: Você verá linhas na banda e poderá pausar. ---")
        if PAUSE_BEFORE_STAGE_3:
            print("--- PAUSA ANTES DA ETAPA 3 ATIVADA: Pressione 'ESPACO' para continuar. ---")
    print(f"--- O período inicial do Arduino é {current_master_period_us_arduino} us ({1e6 / current_master_period_us_arduino:.2f} FPS).")

    while True:
        # --- Leitura de comandos manuais (apenas para depuração ou ajustes extras) ---
        # Certifique-se de que stdin não esteja bloqueando se não houver entrada
        if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline().strip()
            if line:
                if line.startswith('M') or line.startswith('S') or line.startswith('P'):
                    try:
                        command_value = int(line[1:])
                        if (line.startswith('M') and command_value > 0) or \
                           (line.startswith('S') and command_value > 0) or \
                           (line.startswith('P') and command_value >= 0):
                            ser.write((line + '\n').encode('utf-8'))
                            print(f"Comando manual '{line}' enviado ao Arduino.")
                            if line.startswith('M'): 
                                current_master_period_us_arduino = command_value
                                # Si se ajusta M manualmente, y no estamos en la fase de detección (donde se permiten cambios)
                                # se reinicia el proceso a la verificación de detección.
                                if current_calibration_state not in [CALIBRATION_STATE_DETECTION_CHECK]:
                                    current_calibration_state = CALIBRATION_STATE_DETECTION_CHECK 
                                    band_history.clear() 
                                    calibration_start_time = time.time() 
                                    print("Calibração reiniciada para 'Verificação de Deteção' devido a ajuste manual do período.")
                            time.sleep(ARDUINO_WAIT_TIME_SEC) 
                        else:
                            print(f"Valor inválido para comando manual: '{line}'.")
                    except ValueError:
                        print(f"Formato de comando inválido: '{line}'. Use 'M<valor>', 'S<valor>' ou 'P<valor>'.")
                elif line == 'r':
                    if not is_recording:
                        output_filename = f"gravacao_{int(time.time())}.avi"
                        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
                        if frame_width == 0 or frame_height == 0:
                            print("Erro: Resolução da câmera não disponível para iniciar gravação.")
                        else:
                            video_writer = cv2.VideoWriter(output_filename, fourcc, TARGET_FPS, (frame_width, frame_height))
                            if video_writer.isOpened():
                                is_recording = True
                                recording_start_time = time.time()
                                print(f"Iniciando gravação de vídeo em '{output_filename}' por {recording_duration_sec} segundos.")
                            else:
                                print("Erro: Não foi possível criar o objeto VideoWriter. Verifique os codecs ou permissões.")
                    else:
                        print("Já está gravando um vídeo. Espere terminar.")
                else:
                    print(f"Comando manual desconhecido: '{line}'")
        # --- Fim Leitura de comandos manuais ---

        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível ler o frame. Possivelmente a câmera foi desconectada. Saindo.")
            break

        frame_count += 1 # Incrementar el contador de frames

        # Crear una copia del frame para las ventanas pares/impares para evitar modificar el original
        frame_for_even_odd = frame.copy()

        # Si estuviera grabando, el frame se escribe ANTES de ser procesado para las ventanas pares/impares
        if is_recording:
            video_writer.write(frame)
            elapsed_recording_time = time.time() - recording_start_time
            if elapsed_recording_time >= recording_duration_sec:
                is_recording = False
                video_writer.release()
                video_writer = None
                print(f"Gravação de vídeo concluída ({recording_duration_sec} segundos).")
            else:
                remaining_time = recording_duration_sec - elapsed_recording_time
                cv2.putText(frame, f"GRAVANDO: {remaining_time:.1f}s", (10, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        # Convertir para escala de cinza para el procesamiento de la banda
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- Lógica del Algoritmo de Calibração de Borsato ---

        if current_calibration_state == CALIBRATION_STATE_INIT:
            # Este bloque solo se ejecuta si SKIP_GROSS_CALIBRATION es False
            print("Iniciando Etapa de Calibração Grossa (contagem de FPS)...")
            gross_calibration_frame_count = 0
            gross_calibration_start_time = time.time()
            current_calibration_state = CALIBRATION_STATE_GROSS_CALIBRATION
            # Enviar el periodo inicial para que la banda sea visible durante la cal. gruesa
            print(f"Enviando período inicial M{MASTER_PERIOD_INITIAL_US} ao Arduino para calibração grossa...")
            ser.write(f'M{MASTER_PERIOD_INITIAL_US}\n'.encode('utf-8'))
            time.sleep(ARDUINO_WAIT_TIME_SEC)


        elif current_calibration_state == CALIBRATION_STATE_GROSS_CALIBRATION:
            gross_calibration_frame_count += 1
            elapsed_time = time.time() - gross_calibration_start_time

            if elapsed_time >= GROSS_CALIBRATION_DURATION_SEC:
                if gross_calibration_frame_count < MIN_FRAMES_FOR_GROSS_CALIBRATION:
                    print(f"Erro: Poucos frames ({gross_calibration_frame_count}) capturados em {GROSS_CALIBRATION_DURATION_SEC}s para calibração grossa. Verifique a câmera e o FPS.")
                    current_calibration_state = CALIBRATION_STATE_ERROR
                else:
                    estimated_fps = gross_calibration_frame_count / elapsed_time
                    estimated_master_period_us = int(1_000_000 / estimated_fps)
                    
                    print(f"Calibração Grossa Concluída: Capturados {gross_calibration_frame_count} frames em {elapsed_time:.2f}s.")
                    print(f"FPS estimado da câmera: {estimated_fps:.2f} FPS.")
                    print(f"Período mestre estimado do Arduino: {estimated_master_period_us} us.")

                    # Envia o período estimado para o Arduino
                    ser.write(f'M{estimated_master_period_us}\n'.encode('utf-8'))
                    time.sleep(ARDUINO_WAIT_TIME_SEC)
                    current_master_period_us_arduino = estimated_master_period_us
                    
                    print("Calibração Grossa Completa. Passando para a Etapa: Verificação de Deteção de Banda.")
                    current_calibration_state = CALIBRATION_STATE_DETECTION_CHECK
                    # No reseteamos calibration_start_time aquí, se hará al entrar a FINE_TUNING
            else:
                cv2.putText(frame, f"CAL. GROSSA: {gross_calibration_frame_count} frames ({elapsed_time:.1f}/{GROSS_CALIBRATION_DURATION_SEC}s)", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # NUEVA ETAPA: VERIFICACIÓN DE DETECCIÓN DE BANDA
        elif current_calibration_state == CALIBRATION_STATE_DETECTION_CHECK:
            D_band, Dmax_idx, Dmin_idx = detect_illumination_band(gray_frame)

            if D_band is None:
                cv2.putText(frame, "AJUSTAR ILUMINACAO/PARAMETROS!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Banda nao detectada ou sem contraste.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Ajuste S/P no Arduino e kernel/threshold no codigo.", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Dibujar las líneas de debug solo si la banda se detecta
                cv2.line(frame, (0, int(D_band)), (frame_width, int(D_band)), (0, 0, 255), 3) 
                cv2.line(frame, (0, int(Dmin_idx)), (frame_width, int(Dmin_idx)), (255, 0, 0), 1) 
                cv2.line(frame, (0, int(Dmax_idx)), (frame_width, int(Dmax_idx)), (255, 0, 0), 1) 
                cv2.putText(frame, f"Banda @ {int(D_band)}px (Detectada!)", (int(frame_width*0.7), int(D_band) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                cv2.putText(frame, "Banda detectada. Pressione ESPACO para continuar.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Certifique-se que as linhas seguem a banda.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): # La barra de espacio
                print("Detección de banda confirmada por el usuario. Pasando a la Etapa 2: Ajuste Fino.")
                current_calibration_state = CALIBRATION_STATE_FINE_TUNING
                calibration_start_time = time.time() # Resetear timer para ajuste fino
            elif key == ord('q'):
                print("Interrupción manual del usuario durante la verificación de detección.")
                break # Salir del loop


        elif current_calibration_state == CALIBRATION_STATE_FINE_TUNING:
            D_band, Dmax_idx, Dmin_idx = detect_illumination_band(gray_frame)

            if D_band is None:
                cv2.putText(frame, "AVISO: Banda nao detectada! (Ajuste Fino)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                # Opcional: Volver a CALIBRATION_STATE_DETECTION_CHECK si la banda se pierde
                # Esto es una decisión de diseño: si la banda se pierde en ajuste fino, ¿qué hacer?
                # current_calibration_state = CALIBRATION_STATE_DETECTION_CHECK
            else:
                band_history.append(D_band)
                if len(band_history) > BAND_HISTORY_LENGTH:
                    band_history.pop(0) 

                if len(band_history) >= BAND_HISTORY_LENGTH:
                    avg_band_pos = np.mean(band_history[:-1]) 
                    current_movement = band_history[-1] - avg_band_pos 

                    # print(f"Banda em: {D_band:.2f}px. Movimento médio: {current_movement:.2f}px.")

                    if abs(current_movement) < CONVERGENCE_THRESHOLD_PX:
                        print(f"Período estável encontrado: {current_master_period_us_arduino} us. Passando para a Etapa 3.")
                        stable_period_us = current_master_period_us_arduino 
                        
                        if DEBUG_VISUAL_MODE and PAUSE_BEFORE_STAGE_3:
                            current_calibration_state = CALIBRATION_STATE_PAUSED
                            print("PAUSADO. Pressione 'ESPACO' para iniciar a Etapa 3.")
                        else:
                            current_calibration_state = CALIBRATION_STATE_SHIFTING_BAND
                        
                        shift_attempts_count = 0 
                        band_disappeared = False 
                        no_band_detection_count = 0 # Reiniciar contador ao entrar na Etapa 3
                    else:
                        if current_movement > 0: # Se a banda se move para baixo
                            current_master_period_us_arduino += PERIOD_ADJUSTMENT_STEP_US
                            # print(f"Movimento + (baixo): Incrementando período para {current_master_period_us_arduino} us.")
                        else: # Se a banda se move para cima
                            current_master_period_us_arduino -= PERIOD_ADJUSTMENT_STEP_US
                            # print(f"Movimento - (cima): Decrementando período para {current_master_period_us_arduino} us.")
                        
                        ser.write(f'M{current_master_period_us_arduino}\n'.encode('utf-8'))
                        time.sleep(ARDUINO_WAIT_TIME_SEC) 
            
            # --- Desenhar debug visual na Etapa 2 (só na janela principal) ---
            if DEBUG_VISUAL_MODE and D_band is not None:
                # Linha central da banda (Vermelha, mais grossa)
                cv2.line(frame, (0, int(D_band)), (frame_width, int(D_band)), (0, 0, 255), 3) 
                # Borda superior da banda (Azul)
                cv2.line(frame, (0, int(Dmin_idx)), (frame_width, int(Dmin_idx)), (255, 0, 0), 1) 
                # Borda inferior da banda (Azul)
                cv2.line(frame, (0, int(Dmax_idx)), (frame_width, int(Dmax_idx)), (255, 0, 0), 1) 
                cv2.putText(frame, f"Banda @ {int(D_band)}px", (int(frame_width*0.7), int(D_band) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if time.time() - calibration_start_time > 60: 
                print("Erro: Calibração de período não converge após 60s. Revise a iluminação ou os parâmetros.")
                current_calibration_state = CALIBRATION_STATE_ERROR
        
        elif current_calibration_state == CALIBRATION_STATE_PAUSED:
            cv2.putText(frame, "PAUSADO: Banda ESTAVEL!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Pressione ESPACO para iniciar a Etapa 3", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Dibujar la banda mientras está pausado (solo en la ventana principal)
            if DEBUG_VISUAL_MODE and D_band is not None:
                cv2.line(frame, (0, int(D_band)), (frame_width, int(D_band)), (0, 0, 255), 3) 
                cv2.line(frame, (0, int(Dmin_idx)), (frame_width, int(Dmin_idx)), (255, 0, 0), 1) 
                cv2.line(frame, (0, int(Dmax_idx)), (frame_width, int(Dmax_idx)), (255, 0, 0), 1) 
                cv2.putText(frame, f"Banda @ {int(D_band)}px", (int(frame_width*0.7), int(D_band) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            # Esperar que la tecla ESPACIO sea presionada
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '): # La barra de espacio
                print("Continuando para a Etapa 3 por ESPAÇO.")
                current_calibration_state = CALIBRATION_STATE_SHIFTING_BAND
                # Las líneas de la banda dejarán de ser dibujadas al cambiar de estado
            elif key == ord('q'):
                print("Interrupção manual do usuário durante a pausa.")
                break # Sair do loop

        elif current_calibration_state == CALIBRATION_STATE_SHIFTING_BAND:
            D_band, Dmax_idx, Dmin_idx = detect_illumination_band(gray_frame)
            
            if not band_disappeared:
                if shift_attempts_count == 0: 
                    # Solo aplica el bias una vez al inicio de esta etapa
                    biased_period = stable_period_us + BIAS_FOR_SHIFT_US 
                    
                    print(f"Aplicando viés: {BIAS_FOR_SHIFT_US} us. Novo período: {biased_period} us para mover a banda.")
                    ser.write(f'M{biased_period}\n'.encode('utf-8'))
                    time.sleep(ARDUINO_WAIT_TIME_SEC * 5) # Un poco más de tiempo para el viés inicial hacer efecto
                    current_master_period_us_arduino = biased_period 

                shift_attempts_count += 1
                
                # --- Criterio para que la banda haya "desaparecido": ---
                if D_band is None: # Si la función de detección ya retorna None, la banda desapareció
                    no_band_detection_count += 1
                    cv2.putText(frame, f"Banda nao detectada: {no_band_detection_count}/10", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if no_band_detection_count >= 10: # Requer 10 frames consecutivos sin detección para confirmar
                        band_disappeared = True
                        print(f"Banda *confirmada* como desaparecida após {shift_attempts_count} frames e {no_band_detection_count} frames sem detecção. Restaurando período estável.")
                        ser.write(f'M{stable_period_us}\n'.encode('utf-8'))
                        time.sleep(ARDUINO_WAIT_TIME_SEC)
                        current_master_period_us_arduino = stable_period_us 
                        current_calibration_state = CALIBRATION_STATE_STABLE 
                        no_band_detection_count = 0 
                else:
                    no_band_detection_count = 0 # Si detectamos la banda, resetar el contador de no-detección
                    # Las líneas de la banda NO son dibujadas aquí en la Etapa 3 para no interferir con la "desaparición".

                if shift_attempts_count > MAX_SHIFT_ATTEMPTS:
                    print(f"Erro: Banda não desapareceu após {MAX_SHIFT_ATTEMPTS} tentativas. Revise BIAS_FOR_SHIFT_US ou iluminação.")
                    current_calibration_state = CALIBRATION_STATE_ERROR
                    ser.write(f'M{stable_period_us}\n'.encode('utf-8'))
                    time.sleep(ARDUINO_WAIT_TIME_SEC)
                    no_band_detection_count = 0 

        elif current_calibration_state == CALIBRATION_STATE_STABLE:
            cv2.putText(frame, "CALIBRADO! SISTEMA ESTAVEL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            pass 

        elif current_calibration_state == CALIBRATION_STATE_ERROR:
            print("O sistema está em estado de ERRO. Calibração falhou. Pressione 'q' para sair.")
            cv2.putText(frame, "ERRO DE sincronizacao", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # --- Mostrar el frame principal ---
        # Mostrar el estado actual en la pantalla
        state_names = {
            CALIBRATION_STATE_INIT: "INICIALIZANDO",
            CALIBRATION_STATE_FINE_TUNING: "AJUSTE FINO (ETAPA 2)",
            CALIBRATION_STATE_SHIFTING_BAND: "MOVENDO BANDA (ETAPA 3)",
            CALIBRATION_STATE_STABLE: "CALIBRADO",
            CALIBRATION_STATE_ERROR: "ERRO",
            CALIBRATION_STATE_PAUSED: "PAUSADO",
            CALIBRATION_STATE_GROSS_CALIBRATION: "CAL. GROSSA",
            CALIBRATION_STATE_DETECTION_CHECK: "VERIFICACAO DETECCAO (ETAPA 1)"
        }
        current_state_name = state_names.get(current_calibration_state, "DESCONHECIDO")

        cv2.putText(frame, f"Estado: {current_state_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Periodo Arduino: {current_master_period_us_arduino}us ({1e6 / current_master_period_us_arduino:.2f} FPS)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow('Camera PS3 Eye - Calibracao Borsato', frame) 

        # --- Mostrar frames pares e impares ---
        if frame_count % 2 == 0: # Frame par
            even_frame_count_for_fps += 1
            elapsed_even_time = time.time() - last_even_fps_time
            if elapsed_even_time >= 1.0: # Actualizar FPS cada segundo
                even_fps_display = even_frame_count_for_fps / elapsed_even_time
                last_even_fps_time = time.time()
                even_frame_count_for_fps = 0
            
            cv2.putText(frame_for_even_odd, f"FPS: {even_fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Frames Pares', frame_for_even_odd)
        else: # Frame impar
            odd_frame_count_for_fps += 1
            elapsed_odd_time = time.time() - last_odd_fps_time
            if elapsed_odd_time >= 1.0: # Actualizar FPS cada segundo
                odd_fps_display = odd_frame_count_for_fps / elapsed_odd_time
                last_odd_fps_time = time.time()
                odd_frame_count_for_fps = 0

            cv2.putText(frame_for_even_odd, f"FPS: {odd_fps_display:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Frames Impares', frame_for_even_odd)

        # Manejo de teclas global (solo 'q')
        key = cv2.waitKey(1) & 0xFF # Llamar waitKey solo una vez
        
        # Lógica de manejo de teclas específicas para estados
        if current_calibration_state == CALIBRATION_STATE_PAUSED or \
           current_calibration_state == CALIBRATION_STATE_DETECTION_CHECK: 
            # En estos estados, la lógica de pausa con ESPACIO ya está dentro de sus bloques
            # Aquí solo nos interesa si es 'q'
            if key == ord('q'):
                print("Interrupción manual del usuario.")
                break
            # Si no es 'q', y no es ESPACIO (que se maneja internamente), waitKey simplemente retorna
            # Sin embargo, si la clave es ESPACIO, esto también la capturaría, así que
            # es importante que la lógica de ESPACIO en los estados específicos se ejecute *antes* de este if global.
            # La forma en que está ahora, el key se lee una vez y luego se usa.
            # Si un estado maneja la tecla ESPACIO, esa lógica sobrescribirá la necesidad de una `break` global.
            pass # No hacer nada adicional si la tecla ya fue manejada por el estado específico
        elif key == ord('q'): # Si 'q' fue presionada y no estamos en un estado de pausa especial
            print("Interrupción manual del usuario.")
            break
        

    print(f"Visualização concluída.")

if __name__ == "__main__":
    main()

