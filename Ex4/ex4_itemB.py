
"""EXERCICIO 4 - ITEM B
Normalizacao de dataset para direcao autonoma.

REFERENCIAS:
- Aula 1, exemplo 19: variacao de luminosidade.
- Aula 3: leitura, dimensoes e salvamento de imagens.
- Aula 3, exemplo 15: transformacoes geometricas.
- Exercicio 4, Item A: warpAffine e getRotationMatrix2D.

ETAPAS:
1. Carregar 20 frames de camera frontal.
2. Normalizar para 224x224 com letterbox.
3. Padronizar por canal RGB.
4. Aplicar augmentation geometrico e de brilho.
5. Inverter steering no flip horizontal.
6. Salvar imagens, dataset e grid 4x5.

JUSTIFICATIVAS:
- Letterbox: preserva as proporcoes da imagem.
- Padronizacao: centraliza e escala cada canal RGB.
- Brilho: simula diferentes condicoes de iluminacao.
- Flip: simula trajetorias espelhadas e inverte steering.
- Rotacao: simula inclinacoes de ate 10 graus.
- Zoom: simula variacoes de escala de ate 10%.

OBSERVACAO:
Rotacao e zoom podem exigir correcao dos rotulos de
direcao. Por isso, seus rotulos supervisionados sao
marcados como ausentes quando nao ha regra de correcao.
"""

from pathlib import Path
import csv

import cv2
import numpy as np


# ================= CAMINHOS =================

RAIZ = Path(__file__).resolve().parent

CAMINHO_VIDEO = (
    RAIZ.parent / "Ex2" / "dados" / "direcao_udacity.mp4"
)

CAMINHO_STEERING = RAIZ / "dados" / "steering.csv"

PASTA_SAIDA = RAIZ / "saida" / "item_b"
PASTA_NORMALIZADAS = PASTA_SAIDA / "normalizadas"
PASTA_AUMENTADAS = PASTA_SAIDA / "aumentadas"

PASTA_NORMALIZADAS.mkdir(parents=True, exist_ok=True)
PASTA_AUMENTADAS.mkdir(parents=True, exist_ok=True)

TOTAL = 20
TAMANHO = 224

rng = np.random.default_rng(42)


# ================= CARREGAR FRAMES =================

def carregar_imagens():
    """Seleciona 20 frames distribuidos ao longo do video."""

    captura = cv2.VideoCapture(str(CAMINHO_VIDEO))

    if not captura.isOpened():
        raise FileNotFoundError(
            f"Nao foi possivel abrir: {CAMINHO_VIDEO}"
        )

    total_frames = int(
        captura.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    if total_frames < TOTAL:
        captura.release()
        raise RuntimeError(
            "O video precisa conter pelo menos 20 frames."
        )

    indices = np.linspace(
        0,
        total_frames - 1,
        TOTAL,
        dtype=int
    )

    imagens = []

    for indice in indices:

        captura.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(indice)
        )

        ok, imagem = captura.read()

        if not ok:
            captura.release()
            raise RuntimeError(
                f"Erro ao ler o frame {indice}."
            )

        imagens.append((int(indice), imagem))

    captura.release()

    return imagens


# ================= STEERING =================

def carregar_steering():
    """Le angulos reais, quando houver um CSV.

    Formato:
    frame,steering
    0,0.05
    100,-0.10
    """

    if not CAMINHO_STEERING.exists():
        print("CSV de steering nao encontrado.")
        print("Os rotulos serao mantidos como ausentes.")
        return {}

    valores = {}

    with CAMINHO_STEERING.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as arquivo:

        leitor = csv.DictReader(arquivo)

        for linha in leitor:
            valores[int(linha["frame"])] = float(
                linha["steering"]
            )

    return valores


# ================= LETTERBOX =================

def letterbox(imagem):
    """Redimensiona para 224x224 preservando aspect ratio."""

    altura, largura = imagem.shape[:2]

    fator = min(
        TAMANHO / largura,
        TAMANHO / altura
    )

    nova_largura = min(
        TAMANHO,
        max(1, round(largura * fator))
    )

    nova_altura = min(
        TAMANHO,
        max(1, round(altura * fator))
    )

    redimensionada = cv2.resize(
        imagem,
        (nova_largura, nova_altura),
        interpolation=(
            cv2.INTER_AREA
            if fator < 1
            else cv2.INTER_LINEAR
        )
    )

    resultado = np.full(
        (TAMANHO, TAMANHO, 3),
        114,
        dtype=np.uint8
    )

    x = (TAMANHO - nova_largura) // 2
    y = (TAMANHO - nova_altura) // 2

    resultado[
        y:y + nova_altura,
        x:x + nova_largura
    ] = redimensionada

    return resultado


# ================= BRILHO =================

def alterar_brilho(imagem, fator):
    """Varia luminosidade com float32 e np.clip.

    Baseado na abordagem do exemplo 19 da Aula 1.
    """

    resultado = imagem.astype(np.float32) * fator

    return np.clip(
        resultado,
        0,
        255
    ).astype(np.uint8)


# ================= ROTACAO E ZOOM =================

def transformar(imagem, angulo=0, escala=1.0):
    """Aplica transformacao afim com OpenCV."""

    altura, largura = imagem.shape[:2]

    centro = (largura / 2, altura / 2)

    matriz = cv2.getRotationMatrix2D(
        centro,
        angulo,
        escala
    )

    return cv2.warpAffine(
        imagem,
        matriz,
        (largura, altura),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101
    )


# ================= AUGMENTATION =================

def augmentation(imagem, indice, steering):
    """Aplica uma transformacao por imagem.

    O flip horizontal inverte o sinal do steering.
    Rotacao e zoom nao preservam automaticamente
    os rotulos para treinamento supervisionado.
    """

    tipo = indice % 5

    if tipo == 0:

        resultado = cv2.flip(imagem, 1)
        nome = "flip_horizontal"

        if steering is not None:
            steering = -steering

    elif tipo == 1:

        resultado = transformar(
            imagem,
            angulo=10
        )

        nome = "rotacao_+10"
        steering = None

    elif tipo == 2:

        resultado = transformar(
            imagem,
            angulo=-10
        )

        nome = "rotacao_-10"
        steering = None

    elif tipo == 3:

        resultado = transformar(
            imagem,
            escala=0.9
        )

        nome = "zoom_0.9"
        steering = None

    else:

        resultado = transformar(
            imagem,
            escala=1.1
        )

        nome = "zoom_1.1"
        steering = None

    return resultado, nome, steering


# ================= GRID 4x5 =================

def montar_grid(imagens):
    """Organiza 20 imagens em quatro linhas e cinco colunas."""

    quadrinhos = []

    for indice, imagem in enumerate(imagens):

        vista = imagem.copy()

        cv2.rectangle(
            vista,
            (0, 0),
            (TAMANHO, 26),
            (22, 31, 45),
            -1
        )

        cv2.putText(
            vista,
            f"Imagem {indice + 1:02d}",
            (8, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        quadrinhos.append(vista)

    linhas = []

    for inicio in range(0, TOTAL, 5):
        linhas.append(
            np.hstack(
                quadrinhos[inicio:inicio + 5]
            )
        )

    return np.vstack(linhas)


# ================= EXECUCAO =================

def main():

    imagens_video = carregar_imagens()
    rotulos = carregar_steering()

    normalizadas = []
    aumentadas = []
    metadados = []

    # ETAPA 1: normalizacao espacial.
    for indice, (frame_id, imagem) in enumerate(
        imagens_video
    ):

        normalizada = letterbox(imagem)

        normalizadas.append(normalizada)

        cv2.imwrite(
            str(
                PASTA_NORMALIZADAS
                / f"imagem_{indice + 1:02d}.png"
            ),
            normalizada
        )

    # ETAPA 2: padronizacao RGB.
    imagens_rgb = np.stack([
        cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )
        for imagem in normalizadas
    ])

    dados = imagens_rgb.astype(np.float32) / 255.0

    media = dados.mean(axis=(0, 1, 2))
    desvio = dados.std(axis=(0, 1, 2))

    desvio_seguro = np.maximum(desvio, 1e-6)

    dados_padronizados = (
        dados - media
    ) / desvio_seguro

    print("\nMedia RGB:")
    print(media)

    print("\nDesvio padrao RGB:")
    print(desvio)

    # ETAPA 3: data augmentation.
    for indice, (frame_id, _) in enumerate(
        imagens_video
    ):

        imagem = normalizadas[indice]

        # Variacao de brilho de +/-20%.
        fator_brilho = float(
            rng.uniform(0.8, 1.2)
        )

        imagem_brilho = alterar_brilho(
            imagem,
            fator_brilho
        )

        steering_original = rotulos.get(frame_id)

        aumentada, tipo, steering_final = augmentation(
            imagem_brilho,
            indice,
            steering_original
        )

        aumentadas.append(aumentada)

        cv2.imwrite(
            str(
                PASTA_AUMENTADAS
                / f"imagem_{indice + 1:02d}.png"
            ),
            aumentada
        )

        metadados.append([
            indice + 1,
            frame_id,
            fator_brilho,
            tipo,
            steering_original,
            steering_final
        ])

    # ETAPA 4: padronizar imagens aumentadas
    # com as mesmas estatisticas das originais.
    aumentadas_rgb = np.stack([
        cv2.cvtColor(
            imagem,
            cv2.COLOR_BGR2RGB
        )
        for imagem in aumentadas
    ])

    dados_aumentados = (
        aumentadas_rgb.astype(np.float32) / 255.0
    )

    dados_aumentados_padronizados = (
        dados_aumentados - media
    ) / desvio_seguro

    # ETAPA 5: salvar dataset.
    np.savez_compressed(
        PASTA_SAIDA / "dataset_normalizado.npz",
        imagens=dados_padronizados,
        imagens_aumentadas=dados_aumentados_padronizados,
        media_rgb=media,
        desvio_rgb=desvio
    )

    with (
        PASTA_SAIDA / "metadados.csv"
    ).open("w", newline="", encoding="utf-8") as arquivo:

        tabela = csv.writer(arquivo)

        tabela.writerow([
            "imagem",
            "frame_original",
            "fator_brilho",
            "augmentation",
            "steering_original",
            "steering_final"
        ])

        tabela.writerows(metadados)

    # ETAPA 6: grid 4x5.
    grid = montar_grid(normalizadas)

    cv2.imwrite(
        str(PASTA_SAIDA / "grid_4x5.png"),
        grid
    )

    print("\nDataset salvo com sucesso.")
    print("20 imagens normalizadas.")
    print("20 imagens aumentadas.")
    print(f"Diretorio: {PASTA_SAIDA}")

    print("\nFormato do dataset:")
    print(dados_padronizados.shape)

    cv2.namedWindow(
        "Exercicio 4 - Item B",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "Exercicio 4 - Item B",
        900,
        720
    )

    cv2.imshow(
        "Exercicio 4 - Item B",
        grid
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
