import cv2
import numpy as np
import time

# Habilita o uso de OpenCL no OpenCV
cv2.ocl.setUseOpenCL(True)
print("OpenCL habilitado:", cv2.ocl.haveOpenCL())

def processar_com_aceleracao(imagem_cpu):
    """
    Demonstra o fluxo de trabalho de processamento de imagem com aceleração via OpenCL.
    Usando UMat, o OpenCV decide se usa a CPU ou a GPU automaticamente.
    """
    print("Iniciando o processamento com aceleração (CPU/GPU)...")
    start_time_accel = time.time()

    try:
        # 1. Converter para UMat (a matriz "universal" do OpenCV)
        # Este passo move os dados para a memória apropriada (pode ser a GPU)
        imagem_umat = cv2.UMat(imagem_cpu)

        # 2. Executar as operações. O OpenCV usará a GPU automaticamente
        # se a função e o hardware forem compatíveis.
        gray_umat = cv2.cvtColor(imagem_umat, cv2.COLOR_BGR2GRAY)
        
        _, binaria_umat = cv2.threshold(gray_umat, 127, 255, cv2.THRESH_BINARY)
        
        # 3. Baixar a imagem da memória de volta para a CPU
        imagem_binaria_cpu = binaria_umat.get()

        end_time_accel = time.time()
        print(f"Processamento com aceleração concluído em {end_time_accel - start_time_accel:.4f} segundos.")
        return imagem_binaria_cpu

    except Exception as e:
        print(f"Erro no processamento com aceleração: {e}")
        return None

def processar_na_cpu(imagem_cpu):
    """
    Demonstra o mesmo processamento forçando a execução na CPU.
    """
    print("Iniciando o processamento na CPU...")
    start_time_cpu = time.time()

    # Operações padrão do OpenCV (executadas na CPU)
    imagem_gray_cpu = cv2.cvtColor(imagem_cpu, cv2.COLOR_BGR2GRAY)
    _, imagem_binaria_cpu = cv2.threshold(imagem_gray_cpu, 127, 255, cv2.THRESH_BINARY)

    end_time_cpu = time.time()
    print(f"Processamento na CPU concluído em {end_time_cpu - start_time_cpu:.4f} segundos.")
    return imagem_binaria_cpu

def main():
    # Cria uma imagem de exemplo (substitua por uma imagem do seu vídeo)
    largura, altura = 1920, 1080
    imagem_teste = np.random.randint(0, 256, (altura, largura, 3), dtype=np.uint8)
    
    # Processamento com aceleração (GPU via OpenCL)
    imagem_acelerada_result = processar_com_aceleracao(imagem_teste)
    
    # Processamento na CPU
    imagem_cpu_result = processar_na_cpu(imagem_teste)
    
    if imagem_acelerada_result is not None and imagem_cpu_result is not None:
        print("\nVerificando se os resultados da GPU/CPU e CPU são iguais...")
        if np.array_equal(imagem_acelerada_result, imagem_cpu_result):
            print("Os resultados são idênticos.")
        else:
            print("Os resultados são diferentes.")

if __name__ == "__main__":
    if not cv2.ocl.haveOpenCL():
        print("OpenCL não está habilitado. Por favor, certifique-se de ter os drivers e bibliotecas OpenCL corretos.")
    else:
        main()
