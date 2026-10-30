
from pathlib import Path
import cv2
import time
import numpy as np

# Caminhos
RAIZ = Path(__file__).resolve().parent
CAMINHO_VIDEO = RAIZ / "dados" / "direcao_udacity.mp4"
PASTA_SAIDA = RAIZ / "saida"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# Abrir o video
video = cv2.VideoCapture(str(CAMINHO_VIDEO))

if not video.isOpened():
    raise FileNotFoundError(
        f"Nao foi possivel abrir: {CAMINHO_VIDEO}"
    )

tempos_ms = []
numero_frame = 0

# Processamento frame a frame
while True:
    sucesso, frame = video.read()

    if not sucesso:
        break

    inicio = time.perf_counter()

    # 1. Redimensionar para 640x480
    frame = cv2.resize(frame, (640, 480))

    # 2. Converter para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 3. Extrair o canal V (luminosidade)
    canal_v = hsv[:, :, 2]

    # 4. Aplicar filtro bilateral
    filtrado = cv2.bilateralFilter(
        canal_v,
        d=5,
        sigmaColor=50,
        sigmaSpace=50
    )

    fim = time.perf_counter()

    # Calcular tempo e FPS de processamento
    tempo_ms = (fim - inicio) * 1000
    fps = 1000 / tempo_ms if tempo_ms > 0 else 0

    tempos_ms.append(tempo_ms)
    numero_frame += 1

    print(
        f"Frame {numero_frame}: "
        f"{tempo_ms:.2f} ms | "
        f"{fps:.2f} FPS"
    )

    # Mostrar original e processado lado a lado
    filtrado_bgr = cv2.cvtColor(
        filtrado, cv2.COLOR_GRAY2BGR
    )

    comparacao = np.hstack((frame, filtrado_bgr))

    cv2.putText(
        comparacao,
        "Original",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        comparacao,
        "Canal V + Bilateral",
        (655, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("Pipeline de camera embarcada", comparacao)

    # Pressione Q para encerrar
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()

# Resultados finais
if tempos_ms:
    tempo_medio = np.mean(tempos_ms)
    fps_medio = 1000 / tempo_medio

    print("\n=== RESULTADOS FINAIS ===")
    print(f"Frames processados: {numero_frame}")
    print(f"Tempo medio: {tempo_medio:.2f} ms")
    print(f"FPS medio de processamento: {fps_medio:.2f}")

    if fps_medio >= 30:
        print("Pipeline atende ao prazo de 30 FPS.")
    else:
        print("Pipeline nao atende ao prazo de 30 FPS.")
