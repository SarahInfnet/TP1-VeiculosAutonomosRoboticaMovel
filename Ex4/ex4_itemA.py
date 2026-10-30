
"""EXERCICIO 4 - ITEM A - Transformacoes geometricas 2D

OBJETIVO:
Aplicar translacao, rotacao, escala e transformacao afim
customizada sobre uma imagem de camera veicular.

COMO FUNCIONA:
Usa cv2.warpAffine para transformar a imagem e
cv2.getRotationMatrix2D para rotacao e escala.

Tambem demonstra a composicao de rotacao e translacao
por multiplicacao de matrizes homogeneas.

ENTRADA:
dados/estrada_sintetica.png

SAIDAS:
saida/01_translacao.png
saida/02_rotacao.png
saida/03_escala.png
saida/04_afim_customizada.png
saida/05_composicao.png
saida/06_painel_geral.png

COMO EXECUTAR:
python ex4_itemA.py

APLICACOES:
- Translacao: simular pequenos deslocamentos da camera.
- Rotacao: simular variacoes de inclinacao da camera.
- Escala: simular zoom e variacoes de enquadramento.
- Transformacao afim: simular cisalhamento e pequenas
  variacoes geometricas.
- Composicao: combinar transformacoes em uma unica
  reamostragem da imagem.

OBSERVACAO:
Em data augmentation para direcao autonoma, algumas
transformacoes podem exigir ajustes nos rotulos de direcao.
"""

from pathlib import Path
import cv2
import numpy as np


# Os caminhos sao relativos a este arquivo.
RAIZ = Path(__file__).resolve().parent

CAMINHO_IMAGEM = RAIZ / "dados" / "estrada_sintetica.png"

PASTA_SAIDA = RAIZ / "saida"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)


# ================= LEITURA DA IMAGEM =================

imagem = cv2.imread(str(CAMINHO_IMAGEM))

if imagem is None:
    raise FileNotFoundError(
        f"Nao foi possivel ler: {CAMINHO_IMAGEM}"
    )

altura, largura = imagem.shape[:2]

print(f"Largura={largura}; Altura={altura}")


# ================= FUNCAO DE PAINEL =================

def montar_painel(*pares):
    """Monta comparacoes lado a lado, no estilo da Aula 3."""

    quadrinhos = []

    for legenda, img in pares:

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        vista = cv2.resize(img, (640, 360))

        cv2.rectangle(
            vista,
            (0, 0),
            (640, 40),
            (22, 31, 45),
            -1
        )

        cv2.putText(
            vista,
            legenda,
            (15, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        quadrinhos.append(vista)

    if len(quadrinhos) == 2:
        return np.hstack(quadrinhos)

    if len(quadrinhos) == 4:
        return np.vstack((
            np.hstack(quadrinhos[:2]),
            np.hstack(quadrinhos[2:])
        ))

    raise ValueError("Use exatamente dois ou quatro quadros.")


# ================= 1. TRANSLACAO =================

# Desloca a imagem 80 pixels para a direita
# e 30 pixels para baixo.
#
# Aplicacao: simular deslocamentos da camera
# durante a coleta de dados.

tx = 80
ty = 30

matriz_translacao = np.float32([
    [1, 0, tx],
    [0, 1, ty]
])

imagem_transladada = cv2.warpAffine(
    imagem,
    matriz_translacao,
    (largura, altura)
)

print("\n1. Matriz de translacao (2x3):")
print(matriz_translacao)

cv2.imwrite(
    str(PASTA_SAIDA / "01_translacao.png"),
    montar_painel(
        ("Imagem original", imagem),
        ("Translacao (80, 30)", imagem_transladada)
    )
)


# ================= 2. ROTACAO =================

# Rotaciona a imagem 10 graus ao redor do centro.
#
# Aplicacao: simular pequenas variacoes de
# inclinacao da camera.

centro = (largura / 2, altura / 2)
angulo = 10

matriz_rotacao = cv2.getRotationMatrix2D(
    centro,
    angulo,
    1.0
)

imagem_rotacionada = cv2.warpAffine(
    imagem,
    matriz_rotacao,
    (largura, altura)
)

print("\n2. Matriz de rotacao (2x3):")
print(matriz_rotacao)

cv2.imwrite(
    str(PASTA_SAIDA / "02_rotacao.png"),
    montar_painel(
        ("Imagem original", imagem),
        ("Rotacao 10 graus", imagem_rotacionada)
    )
)


# ================= 3. ESCALA =================

# Aplica zoom digital de 1.2 vezes.
#
# Aplicacao: simular variacoes de enquadramento
# e escala aparente dos objetos.

fator_escala = 1.2

matriz_escala = cv2.getRotationMatrix2D(
    centro,
    0,
    fator_escala
)

imagem_ampliada = cv2.warpAffine(
    imagem,
    matriz_escala,
    (largura, altura)
)

print("\n3. Matriz de escala (2x3):")
print(matriz_escala)

cv2.imwrite(
    str(PASTA_SAIDA / "03_escala.png"),
    montar_painel(
        ("Imagem original", imagem),
        ("Zoom 1.2x", imagem_ampliada)
    )
)


# ================= 4. TRANSFORMACAO AFIM =================

# Matriz customizada com cisalhamento horizontal.
#
# O termo 0.15 modifica a coordenada X de acordo
# com a coordenada Y.
#
# Aplicacao: criar pequenas variacoes geometricas
# para data augmentation.

cisalhamento = 0.15

matriz_afim = np.float32([
    [1, cisalhamento, -cisalhamento * altura / 2],
    [0, 1, 0]
])

imagem_afim = cv2.warpAffine(
    imagem,
    matriz_afim,
    (largura, altura)
)

print("\n4. Matriz afim customizada (2x3):")
print(matriz_afim)

cv2.imwrite(
    str(PASTA_SAIDA / "04_afim_customizada.png"),
    montar_painel(
        ("Imagem original", imagem),
        ("Cisalhamento 0.15", imagem_afim)
    )
)


# ================= 5. COMPOSICAO =================

# Demonstra a composicao de duas transformacoes:
# primeiro rotacao, depois translacao.
#
# Matrizes afins 2x3 nao podem ser multiplicadas
# diretamente para representar a composicao.
#
# Por isso, acrescentamos a terceira linha
# [0, 0, 1] para obter matrizes homogeneas 3x3.

rotacao_3x3 = np.vstack((
    matriz_rotacao,
    [0, 0, 1]
))

translacao_3x3 = np.vstack((
    matriz_translacao,
    [0, 0, 1]
))

# A multiplicacao T @ R aplica primeiro R e depois T.
matriz_composta_3x3 = translacao_3x3 @ rotacao_3x3

# cv2.warpAffine recebe uma matriz 2x3.
matriz_composta = matriz_composta_3x3[:2, :]

imagem_composta = cv2.warpAffine(
    imagem,
    matriz_composta,
    (largura, altura)
)

print("\n5. Matriz de rotacao homogenea (3x3):")
print(rotacao_3x3)

print("\n6. Matriz de translacao homogenea (3x3):")
print(translacao_3x3)

print("\n7. Matriz composta T @ R (3x3):")
print(matriz_composta_3x3)

print("\n8. Matriz composta usada no warpAffine (2x3):")
print(matriz_composta)

cv2.imwrite(
    str(PASTA_SAIDA / "05_composicao.png"),
    montar_painel(
        ("Imagem original", imagem),
        ("Rotacao + translacao", imagem_composta)
    )
)


# ================= 6. PAINEL GERAL =================

painel_1 = montar_painel(
    ("Original", imagem),
    ("Translacao", imagem_transladada),
    ("Rotacao", imagem_rotacionada),
    ("Escala", imagem_ampliada)
)

painel_2 = montar_painel(
    ("Original", imagem),
    ("Afim customizada", imagem_afim),
    ("Rotacao", imagem_rotacionada),
    ("Rotacao + translacao", imagem_composta)
)

painel_geral = np.vstack((painel_1, painel_2))

cv2.imwrite(
    str(PASTA_SAIDA / "06_painel_geral.png"),
    painel_geral
)

print("\nTransformacoes concluidas.")
print(f"Imagens salvas em: {PASTA_SAIDA}")

# Exibe os resultados.
cv2.namedWindow("Exercicio 4 - Item A", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Exercicio 4 - Item A", 800, 800)

cv2.imshow("Exercicio 4 - Item A", painel_geral)

cv2.waitKey(0)
cv2.destroyAllWindows()
