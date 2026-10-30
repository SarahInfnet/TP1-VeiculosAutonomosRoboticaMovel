
"""Exercicio 3 - Item A: Deteccao de faixas em video.

Baseado no Exemplo 32 da Aula 3 do professor.

Pipeline:
1. Transformacao de perspectiva BEV
2. Segmentacao HLS + Sobel X
3. Deteccao Canny + HoughLinesP
4. Ajuste das faixas
5. Transformacao inversa e sobreposicao
6. Validacao em 50 frames consecutivos
"""

from pathlib import Path

import cv2
import numpy as np


# ================= CONFIGURACAO =================

RAIZ = Path(__file__).resolve().parent

VIDEO = RAIZ / "dados" / "estrada_120_frames.mp4"

SAIDA = RAIZ / "resultados" / "ex3_itemA"
SAIDA.mkdir(parents=True, exist_ok=True)

LARGURA = 1280
ALTURA = 720

TOTAL_FRAMES = 50

# Pontos manuais do video sintetico do professor.
# Ordem: superior esquerdo, superior direito,
# inferior direito, inferior esquerdo.
SRC = np.float32([
    [570, 430],
    [710, 430],
    [1150, 710],
    [130, 710]
])

DST = np.float32([
    [290, 0],
    [990, 0],
    [990, 719],
    [290, 719]
])

M = cv2.getPerspectiveTransform(SRC, DST)
M_INVERSA = cv2.getPerspectiveTransform(DST, SRC)


# ================= HLS + SOBEL =================

def criar_mascara(bev):
    """Segmenta as faixas usando HLS e Sobel X."""

    hls = cv2.cvtColor(bev, cv2.COLOR_BGR2HLS)

    l = hls[:, :, 1]
    s = hls[:, :, 2]

    cinza = cv2.cvtColor(bev, cv2.COLOR_BGR2GRAY)

    grad = np.abs(
        cv2.Sobel(
            cinza,
            cv2.CV_64F,
            1,
            0,
            ksize=3
        )
    )

    sobel = np.uint8(
        255 * grad / (grad.max() + 1e-9)
    )

    amarelo = (s >= 90) & (l >= 75)
    branco = (l >= 165) & (s < 95)
    bordas = sobel >= 23

    mascara = np.uint8(
        (amarelo | branco | bordas) * 255
    )

    return cv2.morphologyEx(
        mascara,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), dtype=np.uint8)
    )


# ================= CANNY + HOUGH =================

def detectar_segmentos(mascara):
    """Detecta segmentos das faixas com Canny e Hough."""

    arestas = cv2.Canny(mascara, 60, 150)

    linhas = cv2.HoughLinesP(
        arestas,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=35,
        maxLineGap=70
    )

    visualizacao = cv2.cvtColor(
        mascara,
        cv2.COLOR_GRAY2BGR
    )

    if linhas is not None:
        for x1, y1, x2, y2 in linhas.reshape(-1, 4):
            cv2.line(
                visualizacao,
                (x1, y1),
                (x2, y2),
                (0, 120, 255),
                2
            )

    return linhas, visualizacao


# ================= AJUSTE DAS FAIXAS =================

def ajustar_faixas(mascara, segmentos):
    """Ajusta as duas bordas da faixa com polinomios."""

    suporte = np.zeros_like(mascara)

    if segmentos is not None:
        for x1, y1, x2, y2 in segmentos.reshape(-1, 4):

            medio_x = (x1 + x2) / 2

            if (
                abs(x2 - x1) <= 120
                and (medio_x < 510 or medio_x > 770)
            ):
                cv2.line(
                    suporte,
                    (x1, y1),
                    (x2, y2),
                    255,
                    13
                )

    suporte = cv2.bitwise_and(suporte, mascara)

    if cv2.countNonZero(suporte) < 250:
        suporte = mascara.copy()

    ys, xs = np.nonzero(suporte)

    valido = (
        (ys > 80)
        & (
            ((xs > 195) & (xs < 515))
            | ((xs > 765) & (xs < 1085))
        )
    )

    ys = ys[valido]
    xs = xs[valido]

    esquerda = xs < LARGURA // 2
    direita = xs > LARGURA // 2

    if (
        np.count_nonzero(esquerda) < 50
        or np.count_nonzero(direita) < 50
    ):
        return None, None

    ajuste_esquerda = np.polyfit(
        ys[esquerda],
        xs[esquerda],
        2
    )

    ajuste_direita = np.polyfit(
        ys[direita],
        xs[direita],
        2
    )

    return ajuste_esquerda, ajuste_direita


# ================= SOBREPOSICAO =================

def preencher_faixa(frame, esquerda, direita):
    """Projeta as faixas do BEV para a imagem original."""

    y = np.arange(ALTURA, dtype=np.float64)

    x_esquerda = np.clip(
        np.polyval(esquerda, y),
        0,
        LARGURA - 1
    )

    x_direita = np.clip(
        np.polyval(direita, y),
        0,
        LARGURA - 1
    )

    pontos_esquerda = np.column_stack((
        x_esquerda.astype(int),
        y.astype(int)
    )).astype(np.int32)

    pontos_direita = np.column_stack((
        x_direita.astype(int),
        y.astype(int)
    )).astype(np.int32)

    camada_bev = np.zeros_like(frame)

    poligono = np.vstack((
        pontos_esquerda,
        pontos_direita[::-1]
    ))

    cv2.fillPoly(
        camada_bev,
        [poligono],
        (30, 190, 45)
    )

    cv2.polylines(
        camada_bev,
        [pontos_esquerda],
        False,
        (255, 255, 255),
        7
    )

    cv2.polylines(
        camada_bev,
        [pontos_direita],
        False,
        (0, 235, 255),
        7
    )

    camada_original = cv2.warpPerspective(
        camada_bev,
        M_INVERSA,
        (LARGURA, ALTURA)
    )

    area_valida = (
        cv2.cvtColor(
            camada_original,
            cv2.COLOR_BGR2GRAY
        ) > 0
    )

    resultado = frame.copy()

    mistura = cv2.addWeighted(
        frame,
        0.65,
        camada_original,
        0.35,
        0
    )

    resultado[area_valida] = mistura[area_valida]

    return resultado


# ================= PAINEL =================

def montar_painel(original, bev, mascara, hough):
    """Monta os quatro estagios em um painel 2x2."""

    imagens = [
        ("1. ORIGINAL", original),
        ("2. BEV", bev),
        ("3. HLS + SOBEL", mascara),
        ("4. CANNY + HOUGH", hough)
    ]

    blocos = []

    for titulo, imagem in imagens:

        if imagem.ndim == 2:
            imagem = cv2.cvtColor(
                imagem,
                cv2.COLOR_GRAY2BGR
            )

        imagem = cv2.resize(imagem, (640, 360))

        cv2.rectangle(
            imagem,
            (0, 0),
            (640, 38),
            (20, 30, 45),
            -1
        )

        cv2.putText(
            imagem,
            titulo,
            (15, 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        blocos.append(imagem)

    return np.vstack((
        np.hstack(blocos[:2]),
        np.hstack(blocos[2:])
    ))


# ================= PIPELINE =================

def processar_frame(frame):
    """Executa o pipeline completo em um frame."""

    frame = cv2.resize(frame, (LARGURA, ALTURA))

    bev = cv2.warpPerspective(
        frame,
        M,
        (LARGURA, ALTURA)
    )

    mascara = criar_mascara(bev)

    segmentos, hough = detectar_segmentos(mascara)

    esquerda, direita = ajustar_faixas(
        mascara,
        segmentos
    )

    painel = montar_painel(
        frame,
        bev,
        mascara,
        hough
    )

    if esquerda is None or direita is None:

        resultado = frame.copy()

        cv2.putText(
            resultado,
            "Faixas nao identificadas",
            (35, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3
        )

        return resultado, painel, False

    resultado = preencher_faixa(
        frame,
        esquerda,
        direita
    )

    return resultado, painel, True


# ================= EXECUCAO =================

def main():

    captura = cv2.VideoCapture(str(VIDEO))

    if not captura.isOpened():
        raise FileNotFoundError(
            f"Nao foi possivel abrir o video: {VIDEO}"
        )

    fps = captura.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 20

    codec = cv2.VideoWriter_fourcc(*"mp4v")

    video_resultado = cv2.VideoWriter(
        str(SAIDA / "faixas_detectadas.mp4"),
        codec,
        fps,
        (LARGURA, ALTURA)
    )

    video_painel = cv2.VideoWriter(
        str(SAIDA / "painel_etapas.mp4"),
        codec,
        fps,
        (1280, 720)
    )

    if not video_resultado.isOpened() or not video_painel.isOpened():
        captura.release()
        video_resultado.release()
        video_painel.release()
        raise RuntimeError("Nao foi possivel criar os videos de saida.")

    processados = 0
    deteccoes = 0

    try:
        for indice in range(TOTAL_FRAMES):

            ok, frame = captura.read()

            if not ok:
                print("O video terminou antes dos 50 frames.")
                break

            resultado, painel, detectado = processar_frame(frame)

            video_resultado.write(resultado)
            video_painel.write(painel)

            processados += 1

            if detectado:
                deteccoes += 1

            if indice == 0 or indice == TOTAL_FRAMES - 1:

                cv2.imwrite(
                    str(SAIDA / f"resultado_frame_{indice:03d}.png"),
                    resultado
                )

                cv2.imwrite(
                    str(SAIDA / f"painel_frame_{indice:03d}.png"),
                    painel
                )

            print(
                f"Frame {processados:03d}/{TOTAL_FRAMES} "
                f"| Faixas detectadas: {detectado}"
            )

    finally:
        captura.release()
        video_resultado.release()
        video_painel.release()

    print("\n=== EXERCICIO 3 - ITEM A ===")
    print(f"Frames processados: {processados}")
    print(f"Frames com deteccao: {deteccoes}")

    if processados >= TOTAL_FRAMES:
        print("Validacao de 50 frames: OK")
    else:
        print("Validacao de 50 frames: INCOMPLETA")

    print(f"Resultados salvos em: {SAIDA}")


if __name__ == "__main__":
    main()
