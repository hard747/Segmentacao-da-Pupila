# -*- coding: utf-8 -*-
#
# Comentários:
#   - Este script utiliza OpenCV, PySerial e Pyudev para sincronizar uma câmera PS3 Eye com um Arduino.
#   - O algoritmo principal (ControladorSincronizacao) segue o método de Borsato para sincronização de relógio.
#   - O script está organizado em classes para modularidade e clareza.
#   - As classes `GerenciadorDispositivos` e `DetectorFaixa` gerenciam as interações de hardware e o processamento de imagens.
#   - A classe `ControladorSincronizacao` é a lógica central que orquestra o processo.
#   - NOVAS ADIÇÕES: A função para identificar o tipo de pupila foi integrada
#     e seus resultados são exibidos em tempo real nas janelas de frames.

# ------------------------------
# PARTE 1: IMPORTAÇÕES DE BIBLIOTECAS
# ------------------------------
import cv2
import time
import sys
import serial
import select
import atexit
import pyudev
import numpy as np
from scipy.ndimage import convolve1d

# ----------------------------------------------------
# PARTE 2: GERENCIAMENTO DE DISPOSITIVOS
# ----------------------------------------------------
# Esta classe gerencia a conexão e comunicação com a câmera e o Arduino.
# Seu objetivo é abstrair as interações de hardware do resto do código.
class GerenciadorDispositivos:
    # --- Parâmetros de Hardware ---
    # Constantes fixas para a identificação da câmera e a porta serial.
    PS3_EYE_VENDOR_ID = '1415'
    PS3_EYE_MODEL_ID = '2000'
    PORTA_ARDUINO = '/dev/ttyACM0'
    TAXA_BAUD = 115200
    TEMPO_ESPERA_ARDUINO_SEG = 0.1

    def __init__(self, fps_alvo, duracao_estrobe, pre_atraso):
        # Inicialização de atributos como None. Serão preenchidos ao iniciar a conexão.
        self.captura = None  # Objeto de captura de vídeo do OpenCV
        self.serial = None   # Objeto de porta serial
        self.largura_frame = 0
        self.altura_frame = 0

        # Novos atributos que agora são parâmetros de inicialização.
        self.FPS_ALVO = fps_alvo
        self.DURACAO_ESTROBE_INICIAL_US = duracao_estrobe
        self.PRE_ATRASO_ESTROBE_INICIAL_US = pre_atraso

    def obter_id_camera_ps3_eye(self):
        # Usa pyudev para encontrar o índice da câmera PS3 Eye de forma robusta.
        print("Buscando câmera PS3 Eye usando pyudev...")
        try:
            contexto = pyudev.Context()
            # Itera sobre os dispositivos de vídeo para encontrar a câmera específica por Vendor ID e Model ID.
            for dispositivo in contexto.list_devices(subsystem='video4linux'):
                propriedades = dispositivo.properties
                if 'ID_VENDOR_ID' in propriedades and 'ID_MODEL_ID' in propriedades:
                    if propriedades['ID_VENDOR_ID'] == self.PS3_EYE_VENDOR_ID and propriedades['ID_MODEL_ID'] == self.PS3_EYE_MODEL_ID:
                        try:
                            # Extrai o índice numérico do nó do dispositivo (ex: /dev/video0 -> 0)
                            indice_camera = int(dispositivo.device_node.replace('/dev/video', ''))
                            print(f"Câmera PS3 Eye encontrada em: {dispositivo.device_node} (ID OpenCV: {indice_camera})")
                            return indice_camera
                        except ValueError:
                            print(f"Aviso: Não foi possível analisar o ID numérico de {dispositivo.device_node}.")
                            return None
            print(f"Dispositivo PS3 Eye (Fornecedor:{self.PS3_EYE_VENDOR_ID}, Modelo:{self.PS3_EYE_MODEL_ID}) não encontrado usando pyudev.")
            return None
        except ImportError:
            # Lida com o erro se a biblioteca pyudev não estiver instalada.
            print("A biblioteca 'pyudev' não está instalada. Instale-a com: pip install pyudev")
            return None
        except Exception as e:
            print(f"Erro ao procurar dispositivos udev: {e}")
            return None

    def iniciar_camera_e_serial(self):
        # Inicializa a câmera e a conexão serial com o Arduino.
        print("--- Inicializando Câmera PS3 Eye ---")
        id_camera = self.obter_id_camera_ps3_eye()
        if id_camera is None:
            print("Erro: Câmera PS3 Eye não encontrada ou ID não disponível.")
            return False

        # --- Conexão e configuração da Câmera ---
        self.captura = cv2.VideoCapture(id_camera)
        if not self.captura.isOpened():
            print(f"Erro: Não foi possível abrir a câmera com OpenCV (ID: {id_camera}).")
            return False
        print(f"Câmera aberta com OpenCV (ID: {id_camera}).")

        self.captura.set(cv2.CAP_PROP_FPS, self.FPS_ALVO)
        time.sleep(1) # Espera para que a configuração tenha efeito.
        fps_real = self.captura.get(cv2.CAP_PROP_FPS)
        print(f"FPS real da câmera medido: {fps_real:.2f} FPS (pode variar do solicitado).")

        self.largura_frame = int(self.captura.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.altura_frame = int(self.captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Resolução da câmera: {self.largura_frame}x{self.altura_frame}")

        # --- Conexão e configuração do Arduino ---
        print("\n--- Inicializando conexão Serial com Arduino ---")
        try:
            self.serial = serial.Serial(self.PORTA_ARDUINO, self.TAXA_BAUD, timeout=1)
            time.sleep(2)  # Pausa para o Arduino se reiniciar e ficar pronto.
            print(f"Conexão serial estabelecida com Arduino em {self.PORTA_ARDUINO}.")

            # Envia valores iniciais de estrobo para o Arduino.
            self.enviar_comando(f'S{self.DURACAO_ESTROBE_INICIAL_US}')
            self.enviar_comando(f'P{self.PRE_ATRASO_ESTROBE_INICIAL_US}')
        except serial.SerialException as e:
            print(f"Erro: Não foi possível estabelecer conexão serial com Arduino em {self.PORTA_ARDUINO}. {e}")
            if self.captura:
                self.captura.release()
            return False
        return True

    def enviar_comando(self, cmd):
        # Envia um comando de texto para o Arduino via porta serial.
        if self.serial and self.serial.is_open:
            self.serial.write(f"{cmd}\n".encode('utf-8'))
            time.sleep(self.TEMPO_ESPERA_ARDUINO_SEG) # Pequena pausa para evitar sobrecarga no Arduino.
            print(f"Comando enviado ao Arduino: {cmd}")

    def liberar(self):
        # Fecha as conexões de hardware de forma segura.
        # Esta função é registrada para ser executada ao sair do programa (`atexit.register`).
        print("\n--- Realizando limpeza ao sair ---")
        if self.captura and self.captura.isOpened():
            self.captura.release()
            print("Câmera liberada (OpenCV).")
        else:
            print("Câmera não estava aberta ou já foi liberada.")

        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Porta serial do Arduino fechada.")
        else:
            print("Porta serial não estava aberta ou já foi fechada.")

        cv2.destroyAllWindows()
        print("Janelas do OpenCV fechadas.")
        print("--- Limpeza completa ---")

# ----------------------------------------------------
# PARTE 3: DETECÇÃO DE IMAGEM (AGORA COM DETECÇÃO DE PUPILA INCLUÍDA)
# ----------------------------------------------------
# Esta classe se encarrega de processar os frames da câmera
# para detectar a posição da banda preta E o tipo de pupila.
class DetectorFaixa:
    def __init__(self, tamanho_kernel=5, limiar_contraste=25, min_bright_pixel_val=200, min_pixel_count_for_bright_pupil=5000):
        # Inicializa os parâmetros de detecção.
        self.tamanho_kernel = tamanho_kernel if tamanho_kernel % 2 == 1 else tamanho_kernel + 1
        self.limiar_contraste = limiar_contraste
        self.kernel = np.array([-1] * (self.tamanho_kernel // 2) + [0] + [1] * (self.tamanho_kernel // 2), dtype=float)
        
        # Parâmetros para a detecção da pupila, agora atributos da classe.
        self.min_bright_pixel_val = min_bright_pixel_val
        self.min_pixel_count_for_bright_pupil = min_pixel_count_for_bright_pupil

    def detectar_faixa_preta(self, frame_imagem, estado_sincronizacao_atual, ESTADO_SINCRONIZACAO_MOVENDO_FAIXA):
        # Processa um frame da câmera para encontrar a banda preta.
        # A lógica se baseia na convolução unidimensional do perfil de intensidade da imagem.
        if frame_imagem.ndim != 2:
            frame_imagem = cv2.cvtColor(frame_imagem, cv2.COLOR_BGR2GRAY)

        # 1. Média os valores de pixels por cada linha para obter um perfil de intensidade.
        Ic = np.mean(frame_imagem.astype(float), axis=1)
        # 2. Convoluciona o perfil com o kernel para encontrar o centro da banda.
        Iband = convolve1d(Ic, self.kernel, mode='constant', cval=0.0)

        # Verificações para lidar com casos sem detecção ou com baixo contraste.
        if len(Iband) < 2 or np.ptp(Iband) < self.limiar_contraste:
            return None, None, None

        # 3. Encontra os índices dos valores máximo e mínimo, que correspondem aos bordos da banda.
        Dmax_idx = np.argmax(Iband)
        Dmin_idx = np.argmin(Iband)

        if Dmin_idx > Dmax_idx:
            Dmin_idx, Dmax_idx = Dmax_idx, Dmin_idx

        if not (0 <= Dmin_idx < len(Ic) and 0 <= Dmax_idx < len(Ic)):
            return None, None, None

        # 4. Verifica se a banda está nos bordos extremos da imagem, o que pode indicar um erro.
        margem_borda = max(10, self.tamanho_kernel // 2 + 5)
        intensidade_media_imagem = np.mean(frame_imagem)
        LIMIAR_IMAGEM_BRILHANTE = 180
        esta_nas_bordas_extremas = (Dmin_idx <= margem_borda or Dmax_idx >= len(Ic) - 1 - margem_borda)

        if esta_nas_bordas_extremas and intensidade_media_imagem > LIMIAR_IMAGEM_BRILHANTE and \
           estado_sincronizacao_atual == ESTADO_SINCRONIZACAO_MOVENDO_FAIXA:
            return None, None, None

        # 5. Calcula a posição central da banda.
        centro_faixa_D = (Dmax_idx + Dmin_idx) * 0.5
        return centro_faixa_D, Dmax_idx, Dmin_idx

    def identificar_tipo_pupila(self, img_gray):
        """
        NOVO MÉTODO DA CLASSE:
        Identifica se uma imagem contém pupila escura ou brilhante
        com base na contagem de pixels acima de um limiar de brilho.
        """
        if img_gray is None:
            return 'desconhecido', {}

        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        bright_pixels_in_range_count = np.sum(hist[self.min_bright_pixel_val : 256])

        metricas = {
            'bright_pixel_count': int(bright_pixels_in_range_count)
        }

        if bright_pixels_in_range_count >= self.min_pixel_count_for_bright_pupil:
            return 'brilhante', metricas
        else:
            return 'escura', metricas

# ----------------------------------------------------
# PARTE 4: LÓGICA DE SINCRONIZAÇÃO
# ----------------------------------------------------
# Esta é a classe principal que implementa o algoritmo de sincronização.
# Controla o fluxo do programa, os estados de sincronização e as interações com as outras classes.
class ControladorSincronizacao:
    # --- Parâmetros de Sincronização ---
    PASSO_AJUSTE_PERIODO_US = 10
    LIMIAR_CONVERGENCIA_PX = 1
    COMPRIMENTO_HISTORICO_FAIXA = 10
    VIÉS_PARA_MOVIMENTO_US = -500
    MAX_TENTATIVAS_MOVIMENTO = 1000
    DURACAO_SINCRONIZACAO_GROSSA_SEG = 3
    MIN_FRAMES_PARA_SINCRONIZACAO_GROSSA = 60

    # --- Estados do Algoritmo ---
    # Constantes que representam os diferentes estados do processo de sincronização.
    ESTADO_SINCRONIZACAO_INICIAL = 0
    ESTADO_SINCRONIZACAO_GROSSA = 6
    ESTADO_SINCRONIZACAO_AJUSTE_FINO = 1
    ESTADO_SINCRONIZACAO_PAUSADO = 5
    ESTADO_SINCRONIZACAO_MOVENDO_FAIXA = 2
    ESTADO_SINCRONIZACAO_ESTAVEL = 3
    ESTADO_SINCRONIZACAO_ERRO = 4
    ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO = 7

    def __init__(self, gerenciador_dispositivos, detector_faixa, pular_sincronizacao_grossa=True,
                 modo_visual_depuracao=True, pausar_antes_etapa_3=True, periodo_mestre_inicial_us=33333):
        # Inicialização das dependências e do estado interno.
        self.gerenciador_dispositivos = gerenciador_dispositivos
        self.detector_faixa = detector_faixa
        self.pular_sincronizacao_grossa = pular_sincronizacao_grossa
        self.modo_visual_depuracao = modo_visual_depuracao
        self.pausar_antes_etapa_3 = pausar_antes_etapa_3

        # Novo atributo de inicialização.
        self.periodo_mestre_inicial_us = periodo_mestre_inicial_us

        self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_INICIAL
        self.periodo_mestre_atual_us_arduino = self.periodo_mestre_inicial_us
        self.historico_faixa = []
        self.contador_sem_deteccao_faixa = 0
        self.contador_frames_sincronizacao_grossa = 0
        self.tempo_inicio_sincronizacao_grossa = 0
        self.periodo_estavel_us = 0
        self.contador_tentativas_movimento = 0
        self.faixa_desapareceu = False
        self.tempo_inicio_sincronizacao = 0

        self.esta_gravando = False
        self.tempo_inicio_gravacao = 0
        self.duracao_gravacao_seg = 10
        self.escritor_video = None

        self.contador_frames = 0
        self.contador_frames_pares_para_fps = 0
        self.contador_frames_impares_para_fps = 0
        self.ultimo_tempo_fps_par = time.time()
        self.ultimo_tempo_fps_impar = time.time()
        self.fps_par_exibicao = 0.0
        self.fps_impar_exibicao = 0.0

    def executar(self):
        # Método principal que inicia e executa o loop de sincronização.
        if not self.gerenciador_dispositivos.iniciar_camera_e_serial():
            print("--- Inicialização falhou. Saindo. ---")
            return

        # --- Lógica de Salto de Sincronização Grossa ---
        if self.pular_sincronizacao_grossa:
            self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO
            print(f"Pulando Sincronização Grossa. Iniciando diretamente na Verificação de Detecção.")
            self.gerenciador_dispositivos.enviar_comando(f'M{self.periodo_mestre_inicial_us}')
            self.periodo_mestre_atual_us_arduino = self.periodo_mestre_inicial_us
        else:
            self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_INICIAL

        print("\n--- Iniciando sincronização do relógio externo (Algoritmo de Borsato) ---")
        print("--- Pressione 'q' para sair a qualquer momento. ---")
        print("--- Pressione 'r' para gravar 10 segundos de vídeo. ---")
        if self.modo_visual_depuracao:
            print("--- MODO DE DEBUG VISUAL ATIVADO ---")
            if self.pausar_antes_etapa_3:
                print("--- PAUSA ANTES DA ETAPA 3 ATIVADA ---")
        print(f"Período inicial do Arduino: {self.periodo_mestre_atual_us_arduino} us")

        # --- Loop Principal de Captura e Processamento ---
        while True:
            # Verifica a entrada do usuário para comandos manuais.
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                linha = sys.stdin.readline().strip()
                self.lidar_com_comando_manual(linha)

            # Lê um novo frame da câmera.
            ret, frame = self.gerenciador_dispositivos.captura.read()
            if not ret:
                print("Erro: Não foi possível ler o frame. Saindo.")
                break

            self.contador_frames += 1
            frame_para_par_impar = frame.copy()

            # Lógica de gravação de vídeo.
            if self.esta_gravando:
                self.escritor_video.write(frame)
                decorrido = time.time() - self.tempo_inicio_gravacao
                if decorrido >= self.duracao_gravacao_seg:
                    self.esta_gravando = False
                    self.escritor_video.release()
                    self.escritor_video = None
                    print(f"Gravação de vídeo concluída ({self.duracao_gravacao_seg} segundos).")
                else:
                    restante = self.duracao_gravacao_seg - decorrido
                    cv2.putText(frame, f"GRAVANDO: {restante:.1f}s", (10, self.gerenciador_dispositivos.altura_frame - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            frame_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.processar_estado(frame_cinza, frame)
            self.exibir_frames(frame, frame_para_par_impar, frame_cinza)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Interrupção manual do usuário.")
                break

        print("Visualização concluída.")
        self.gerenciador_dispositivos.liberar()

    def lidar_com_comando_manual(self, linha):
        # Processa comandos digitados pelo usuário na console.
        if not linha:
            return
        if linha.startswith(('M', 'S', 'P')):
            # Lógica para gerenciar comandos de ajuste de período, estrobo e pré-atraso.
            try:
                valor_comando = int(linha[1:])
                if (linha.startswith('M') and valor_comando > 0) or \
                   (linha.startswith('S') and valor_comando > 0) or \
                   (linha.startswith('P') and valor_comando >= 0):
                    self.gerenciador_dispositivos.enviar_comando(linha)
                    if linha.startswith('M'):
                        self.periodo_mestre_atual_us_arduino = valor_comando
                        if self.estado_sincronizacao_atual not in [self.ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO]:
                            self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO
                            self.historico_faixa.clear()
                            print("Sincronização reiniciada para 'Verificação de Detecção' devido a ajuste manual do período.")
                else:
                    print(f"Valor inválido para comando manual: '{linha}'.")
            except ValueError:
                print(f"Formato de comando inválido: '{linha}'. Use 'M<valor>', 'S<valor>' ou 'P<valor>'.")
        elif linha == 'r':
            self.iniciar_gravacao()
        else:
            print(f"Comando manual desconhecido: '{linha}'")

    def iniciar_gravacao(self):
        # Inicia a gravação de um vídeo.
        if self.esta_gravando:
            print("Já está gravando um vídeo. Espere terminar.")
            return
        nome_arquivo = f"gravacao_{int(time.time())}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        if self.gerenciador_dispositivos.largura_frame == 0 or self.gerenciador_dispositivos.altura_frame == 0:
            print("Erro: Resolução da câmera não disponível para iniciar gravação.")
            return
        self.escritor_video = cv2.VideoWriter(nome_arquivo, fourcc, self.gerenciador_dispositivos.FPS_ALVO,
                                            (self.gerenciador_dispositivos.largura_frame, self.gerenciador_dispositivos.altura_frame))
        if self.escritor_video.isOpened():
            self.esta_gravando = True
            self.tempo_inicio_gravacao = time.time()
            print(f"Iniciando gravação de vídeo em '{nome_arquivo}' por {self.duracao_gravacao_seg} segundos.")
        else:
            print("Erro: Não foi possível criar o objeto VideoWriter.")

    def processar_estado(self, frame_cinza, frame):
        # A lógica central que muda de estado de acordo com o progresso do algoritmo.
        s = self.estado_sincronizacao_atual
        gd = self.gerenciador_dispositivos
        df = self.detector_faixa

        # --- Etapa 0: Sincronização Inicial ---
        if s == self.ESTADO_SINCRONIZACAO_INICIAL:
            print("Iniciando Etapa de Sincronização Grossa (contagem de FPS)...")
            self.contador_frames_sincronizacao_grossa = 0
            self.tempo_inicio_sincronizacao_grossa = time.time()
            self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_GROSSA
            gd.enviar_comando(f'M{self.periodo_mestre_inicial_us}')

        # --- Sincronização Grossa ---
        elif s == self.ESTADO_SINCRONIZACAO_GROSSA:
            self.contador_frames_sincronizacao_grossa += 1
            decorrido = time.time() - self.tempo_inicio_sincronizacao_grossa
            if decorrido >= self.DURACAO_SINCRONIZACAO_GROSSA_SEG:
                if self.contador_frames_sincronizacao_grossa < self.MIN_FRAMES_PARA_SINCRONIZACAO_GROSSA:
                    print(f"Erro: Poucos frames ({self.contador_frames_sincronizacao_grossa}) capturados em {self.DURACAO_SINCRONIZACAO_GROSSA_SEG}s.")
                    self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_ERRO
                else:
                    fps_estimado = self.contador_frames_sincronizacao_grossa / decorrido
                    periodo_estimado = int(1_000_000 / fps_estimado)
                    print(f"Sincronização Grossa Concluída: FPS estimado {fps_estimado:.2f}, período {periodo_estimado} us.")
                    gd.enviar_comando(f'M{periodo_estimado}')
                    self.periodo_mestre_atual_us_arduino = periodo_estimado
                    self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO
            else:
                cv2.putText(frame, f"SINC. GROSSA: {self.contador_frames_sincronizacao_grossa} frames ({decorrido:.1f}/{self.DURACAO_SINCRONIZACAO_GROSSA_SEG}s)",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # --- Etapa 1: Verificação de Detecção ---
        elif s == self.ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO:
            faixa_D, Dmax_idx, Dmin_idx = df.detectar_faixa_preta(frame_cinza, s, self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA)
            if faixa_D is None:
                cv2.putText(frame, "AJUSTAR ILUMINACAO/PARAMETROS!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "Faixa nao detectada ou sem contraste.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                # Desenha as linhas de detecção no frame de visualização.
                cv2.line(frame, (0, int(faixa_D)), (gd.largura_frame, int(faixa_D)), (0, 0, 255), 3)
                cv2.line(frame, (0, int(Dmin_idx)), (gd.largura_frame, int(Dmin_idx)), (255, 0, 0), 1)
                cv2.line(frame, (0, int(Dmax_idx)), (gd.largura_frame, int(Dmax_idx)), (255, 0, 0), 1)
                cv2.putText(frame, f"Faixa @ {int(faixa_D)}px (Detectada!)", (int(gd.largura_frame*0.7), int(faixa_D) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv2.putText(frame, "Faixa detectada. Pressione ESPACO para continuar.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, "Certifique-se que as linhas seguem a faixa.", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print("Detecção de faixa confirmada pelo usuário. Passando para a Etapa 2: Ajuste Fino.")
                self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_AJUSTE_FINO
                self.historico_faixa.clear()
                self.tempo_inicio_sincronizacao = time.time()
            elif key == ord('q'):
                print("Interrupção manual do usuário durante a verificação de detecção.")
                sys.exit(0)

        # --- Etapa 2: Ajuste Fino ---
        elif s == self.ESTADO_SINCRONIZACAO_AJUSTE_FINO:
            faixa_D, Dmax_idx, Dmin_idx = df.detectar_faixa_preta(frame_cinza, s, self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA)
            if faixa_D is None:
                cv2.putText(frame, "AVISO: Faixa nao detectada! (Ajuste Fino)", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Lógica para ajustar o período do Arduino com base no movimento da banda.
                self.historico_faixa.append(faixa_D)
                if len(self.historico_faixa) > self.COMPRIMENTO_HISTORICO_FAIXA:
                    self.historico_faixa.pop(0)

                if len(self.historico_faixa) >= self.COMPRIMENTO_HISTORICO_FAIXA:
                    pos_media_faixa = np.mean(self.historico_faixa[:-1])
                    movimento_atual = self.historico_faixa[-1] - pos_media_faixa

                    if abs(movimento_atual) < self.LIMIAR_CONVERGENCIA_PX:
                        print(f"Período estável encontrado: {self.periodo_mestre_atual_us_arduino} us. Passando para a Etapa 3.")
                        self.periodo_estavel_us = self.periodo_mestre_atual_us_arduino
                        if self.modo_visual_depuracao and self.pausar_antes_etapa_3:
                            self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_PAUSADO
                            print("PAUSADO. Pressione 'ESPACO' para iniciar a Etapa 3.")
                        else:
                            self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA
                        self.contador_tentativas_movimento = 0
                        self.faixa_desapareceu = False
                        self.contador_sem_deteccao_faixa = 0
                    else:
                        if movimento_atual > 0:
                            self.periodo_mestre_atual_us_arduino += self.PASSO_AJUSTE_PERIODO_US
                        else:
                            self.periodo_mestre_atual_us_arduino -= self.PASSO_AJUSTE_PERIODO_US
                        gd.enviar_comando(f'M{self.periodo_mestre_atual_us_arduino}')

            if self.modo_visual_depuracao and faixa_D is not None:
                cv2.line(frame, (0, int(faixa_D)), (gd.largura_frame, int(faixa_D)), (0, 0, 255), 3)
                cv2.line(frame, (0, int(Dmin_idx)), (gd.largura_frame, int(Dmin_idx)), (255, 0, 0), 1)
                cv2.line(frame, (0, int(Dmax_idx)), (gd.largura_frame, int(Dmax_idx)), (255, 0, 0), 1)
                cv2.putText(frame, f"Faixa @ {int(faixa_D)}px", (int(gd.largura_frame*0.7), int(faixa_D) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            if time.time() - self.tempo_inicio_sincronizacao > 60:
                print("Erro: Sincronização de período não converge após 60s.")
                self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_ERRO

        # --- Estado Pausado (opcional em debug) ---
        elif s == self.ESTADO_SINCRONIZACAO_PAUSADO:
            cv2.putText(frame, "PAUSADO: Faixa ESTAVEL!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, "Pressione ESPACO para iniciar a Etapa 3", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            faixa_D, Dmax_idx, Dmin_idx = df.detectar_faixa_preta(frame_cinza, s, self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA)
            if self.modo_visual_depuracao and faixa_D is not None:
                cv2.line(frame, (0, int(faixa_D)), (gd.largura_frame, int(faixa_D)), (0, 0, 255), 3)
                cv2.line(frame, (0, int(Dmin_idx)), (gd.largura_frame, int(Dmin_idx)), (255, 0, 0), 1)
                cv2.line(frame, (0, int(Dmax_idx)), (gd.largura_frame, int(Dmax_idx)), (255, 0, 0), 1)
                cv2.putText(frame, f"Faixa @ {int(faixa_D)}px", (int(gd.largura_frame*0.7), int(faixa_D) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):
                print("Continuando para a Etapa 3 por ESPAÇO.")
                self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA
            elif key == ord('q'):
                print("Interrupção manual do usuário durante a pausa.")
                sys.exit(0)

        # --- Etapa 3: Movimento da Faixa ---
        elif s == self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA:
            faixa_D, Dmax_idx, Dmin_idx = df.detectar_faixa_preta(frame_cinza, s, self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA)
            if not self.faixa_desapareceu:
                if self.contador_tentativas_movimento == 0:
                    # Aplica um "viés" (bias) ao período para mover a banda.
                    periodo_viciado = self.periodo_estavel_us + self.VIÉS_PARA_MOVIMENTO_US
                    print(f"Aplicando viés: {self.VIÉS_PARA_MOVIMENTO_US} us. Novo período: {periodo_viciado} us para mover a faixa.")
                    gd.enviar_comando(f'M{periodo_viciado}')
                    self.periodo_mestre_atual_us_arduino = periodo_viciado
                    time.sleep(gd.TEMPO_ESPERA_ARDUINO_SEG * 5)

                self.contador_tentativas_movimento += 1

                if faixa_D is None:
                    # Conta os frames em que a banda não é detectada para confirmar seu "desaparecimento".
                    self.contador_sem_deteccao_faixa += 1
                    cv2.putText(frame, f"Faixa nao detectada: {self.contador_sem_deteccao_faixa}/10", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    if self.contador_sem_deteccao_faixa >= 10:
                        self.faixa_desapareceu = True
                        print(f"Faixa confirmada como desaparecida após {self.contador_tentativas_movimento} frames.")
                        # Retorna ao período estável para que a banda reapareça.
                        gd.enviar_comando(f'M{self.periodo_estavel_us}')
                        self.periodo_mestre_atual_us_arduino = self.periodo_estavel_us
                        self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_ESTAVEL
                        self.contador_sem_deteccao_faixa = 0
                else:
                    self.contador_sem_deteccao_faixa = 0

                if self.contador_tentativas_movimento > self.MAX_TENTATIVAS_MOVIMENTO:
                    print(f"Erro: Faixa não desapareceu após {self.MAX_TENTATIVAS_MOVIMENTO} tentativas.")
                    self.estado_sincronizacao_atual = self.ESTADO_SINCRONIZACAO_ERRO
                    gd.enviar_comando(f'M{self.periodo_estavel_us}')
                    self.contador_sem_deteccao_faixa = 0

        # --- Estado: Sincronizado e Estável ---
        elif s == self.ESTADO_SINCRONIZACAO_ESTAVEL:
            cv2.putText(frame, "SINCRONIZADO! SISTEMA ESTAVEL", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # --- Estado de Erro ---
        elif s == self.ESTADO_SINCRONIZACAO_ERRO:
            print("O sistema está em estado de ERRO. Sincronização falhou. Pressione 'q' para sair.")
            cv2.putText(frame, "ERRO DE SINCRONIZACAO", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def exibir_frames(self, frame, frame_para_par_impar, frame_cinza):
        # Mostra os frames nas janelas do OpenCV, incluindo o cálculo de FPS e a detecção de pupila.
        gd = self.gerenciador_dispositivos
        self.exibir_info_estado(frame)

        # Realiza a detecção de pupila usando o método da classe DetectorFaixa.
        tipo_pupila, metricas = self.detector_faixa.identificar_tipo_pupila(frame_cinza)

        cv2.imshow('Camera PS3 Eye - Sincronizacao', frame)

        # Atualiza as janelas de frames pares e ímpares com as informações da pupila
        if self.contador_frames % 2 == 0:
            self.contador_frames_pares_para_fps += 1
            decorrido = time.time() - self.ultimo_tempo_fps_par
            if decorrido >= 1.0:
                self.fps_par_exibicao = self.contador_frames_pares_para_fps / decorrido
                self.ultimo_tempo_fps_par = time.time()
                self.contador_frames_pares_para_fps = 0

            cv2.putText(frame_para_par_impar, f"FPS: {self.fps_par_exibicao:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_para_par_impar, f"Pupila: {tipo_pupila}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            #cv2.putText(frame_para_par_impar, f"Pixeis Brilhantes: {metricas['bright_pixel_count']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Frames Pares', frame_para_par_impar)
        else:
            self.contador_frames_impares_para_fps += 1
            decorrido = time.time() - self.ultimo_tempo_fps_impar
            if decorrido >= 1.0:
                self.fps_impar_exibicao = self.contador_frames_impares_para_fps / decorrido
                self.ultimo_tempo_fps_impar = time.time()
                self.contador_frames_impares_para_fps = 0

            cv2.putText(frame_para_par_impar, f"FPS: {self.fps_impar_exibicao:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_para_par_impar, f"Pupila: {tipo_pupila}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            #cv2.putText(frame_para_par_impar, f"Pixeis Brilhantes: {metricas['bright_pixel_count']}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow('Frames Impares', frame_para_par_impar)

    def exibir_info_estado(self, frame):
        # Mostra informações de estado e período na janela principal.
        nomes_estados = {
            self.ESTADO_SINCRONIZACAO_INICIAL: "INICIALIZANDO",
            self.ESTADO_SINCRONIZACAO_AJUSTE_FINO: "AJUSTE FINO (ETAPA 2)",
            self.ESTADO_SINCRONIZACAO_MOVENDO_FAIXA: "MOVENDO FAIXA (ETAPA 3)",
            self.ESTADO_SINCRONIZACAO_ESTAVEL: "SINCRONIZADO",
            self.ESTADO_SINCRONIZACAO_ERRO: "ERRO",
            self.ESTADO_SINCRONIZACAO_PAUSADO: "PAUSADO",
            self.ESTADO_SINCRONIZACAO_GROSSA: "SINC. GROSSA",
            self.ESTADO_SINCRONIZACAO_VERIFICACAO_DETECCAO: "VERIFICACAO DETECCAO (ETAPA 1)"
        }
        nome_estado_atual = nomes_estados.get(self.estado_sincronizacao_atual, "DESCONHECIDO")
        cv2.putText(frame, f"Estado: {nome_estado_atual}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Periodo Arduino: {self.periodo_mestre_atual_us_arduino}us ({1e6 / self.periodo_mestre_atual_us_arduino:.2f} FPS)", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# ----------------------------------------------------
# PARTE 5: PONTO DE ENTRADA DO PROGRAMA
# ----------------------------------------------------
# O código só é executado se o script for o arquivo principal.
if __name__ == "__main__":
    # Cria instâncias das classes.

    gerenciador_dispositivos = GerenciadorDispositivos(fps_alvo=30, duracao_estrobe=16000,pre_atraso=8000) #30fps
    #gerenciador_dispositivos = GerenciadorDispositivos(fps_alvo=60, duracao_estrobe=8000,pre_atraso=4000) # 60fps
    #gerenciador_dispositivos = GerenciadorDispositivos(fps_alvo=120, duracao_estrobe=4000,pre_atraso=2000) # 120fps

    detector_faixa = DetectorFaixa()

    controlador_sincronizacao = ControladorSincronizacao(gerenciador_dispositivos, detector_faixa, pular_sincronizacao_grossa=True,periodo_mestre_inicial_us=33333) # 30fps
    #controlador_sincronizacao = ControladorSincronizacao(gerenciador_dispositivos, detector_faixa, pular_sincronizacao_grossa=True,periodo_mestre_inicial_us=16666) # 60 fps
    #controlador_sincronizacao = ControladorSincronizacao(gerenciador_dispositivos, detector_faixa, pular_sincronizacao_grossa=True,periodo_mestre_inicial_us=8333) # 120 fps

    # Registra a função 'liberar' para ser chamada automaticamente ao sair.
    atexit.register(gerenciador_dispositivos.liberar)
    
    # Inicia a execução do algoritmo de sincronização.
    controlador_sincronizacao.executar()