
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Caminhos dos arquivos
RAIZ = Path(__file__).resolve().parent
CAMINHO_IMAGEM = RAIZ / "dados" / "frontal_kitti.png"
PASTA_SAIDA = RAIZ / "saida"
PASTA_SAIDA.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. Carregar a imagem e exibir os metadados
# ============================================================

imagem = cv2.imread(str(CAMINHO_IMAGEM))

if imagem is None:
    raise FileNotFoundError(f"Nao foi possivel carregar: {CAMINHO_IMAGEM}")

altura, largura = imagem.shape[:2]
canais = imagem.shape[2] if imagem.ndim == 3 else 1

print("=== METADADOS DA IMAGEM ===")
print(f"Resolucao: {largura} x {altura}")
print(f"Tipo de dados (dtype): {imagem.dtype}")
print(f"Numero de canais: {canais}")
print(f"Tamanho em bytes: {imagem.nbytes}")

# ============================================================
# 2. Dividir a imagem em quatro quadrantes
# ============================================================

cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

meio_y = altura // 2
meio_x = largura // 2

quadrantes = {
    "Superior esquerdo": cinza[:meio_y, :meio_x],
    "Superior direito": cinza[:meio_y, meio_x:],
    "Inferior esquerdo": cinza[meio_y:, :meio_x],
    "Inferior direito": cinza[meio_y:, meio_x:]
}

print("\n=== BRILHO MEDIO POR QUADRANTE ===")

for nome, regiao in quadrantes.items():
    brilho_medio = np.mean(regiao)
    print(f"{nome}: {brilho_medio:.2f}")

# ============================================================
# 3. Aplicar CLAHE no canal L do espaco Lab
# ============================================================

# O CLAHE melhora o contraste local da imagem.
# Aplicamos no canal de luminosidade para preservar as cores.

lab = cv2.cvtColor(imagem, cv2.COLOR_BGR2LAB)

canal_l, canal_a, canal_b = cv2.split(lab)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

canal_l_equalizado = clahe.apply(canal_l)

lab_equalizado = cv2.merge(
    (canal_l_equalizado, canal_a, canal_b)
)

imagem_equalizada = cv2.cvtColor(
    lab_equalizado, cv2.COLOR_LAB2BGR
)

# ============================================================
# 4. Comparar os histogramas antes e depois do CLAHE
# ============================================================

hist_original = cv2.calcHist(
    [canal_l], [0], None, [256], [0, 256]
)

hist_equalizado = cv2.calcHist(
    [canal_l_equalizado], [0], None, [256], [0, 256]
)

plt.figure(figsize=(10, 5))

plt.plot(hist_original, label="Original")
plt.plot(hist_equalizado, label="Apos CLAHE")

plt.title("Comparacao dos histogramas antes e apos CLAHE")
plt.xlabel("Intensidade luminosa")
plt.ylabel("Quantidade de pixels")
plt.legend()
plt.grid(True)

plt.savefig(PASTA_SAIDA / "histogramas_clahe.png")
plt.show()

# ============================================================
# 5. Exibir as imagens lado a lado
# ============================================================

original_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
equalizada_rgb = cv2.cvtColor(
    imagem_equalizada, cv2.COLOR_BGR2RGB
)

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.imshow(original_rgb)
plt.title("Imagem original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(equalizada_rgb)
plt.title("Imagem com CLAHE")
plt.axis("off")

plt.tight_layout()
plt.savefig(PASTA_SAIDA / "comparacao_clahe.png")
plt.show()

# ============================================================
# 6. Comentario tecnico
# ============================================================

# Variacoes de iluminacao podem prejudicar algoritmos de
# percepcao utilizados em veiculos autonomos.
#
# Em regioes escuras, faixas e obstaculos podem apresentar
# pouco contraste em relacao ao ambiente.
#
# Em regioes muito claras, o excesso de luminosidade pode
# dificultar a identificacao de bordas e detalhes.
#
# O CLAHE melhora o contraste local e pode facilitar a
# identificacao desses elementos. Entretanto, tambem pode
# aumentar a visibilidade de ruidos presentes na imagem.

print("\nProcessamento concluido!")
print(f"Resultados salvos em: {PASTA_SAIDA}")
