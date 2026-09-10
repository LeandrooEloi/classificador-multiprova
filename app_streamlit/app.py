"""
App Streamlit - Corretor de Multiprova (V/F, 1 a 5, A a E)

Como rodar:
    pip install -r requirements.txt
    streamlit run app.py

Antes de rodar, copie para esta pasta os 3 arquivos gerados no notebook:
    modelo_binario_VF.joblib
    modelo_multiclasse_1a5.joblib
    modelo_multiclasse_AaE.joblib
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
import joblib
from PIL import Image, ImageFilter
from streamlit_mnist_canvas import st_mnist_canvas

st.set_page_config(page_title="Corretor de Multiprova", layout="wide")
st.title("Corretor de Multiprova")
st.caption("Desenhe no canvas e clique em enviar: cada quadro aciona o classificador correspondente.")

# Pasta deste arquivo: os .joblib são carregados por caminho absoluto para não
# depender do diretório de onde o `streamlit run` foi disparado (isso muda
# conforme o serviço de hospedagem).
PASTA_APP = Path(__file__).resolve().parent


@st.cache_resource
def carregar_modelos():
    modelo_vf = joblib.load(PASTA_APP / "modelo_binario_VF.joblib")
    modelo_15 = joblib.load(PASTA_APP / "modelo_multiclasse_1a5.joblib")
    modelo_ae = joblib.load(PASTA_APP / "modelo_multiclasse_AaE.joblib")
    return modelo_vf, modelo_15, modelo_ae


try:
    modelo_vf, modelo_15, modelo_ae = carregar_modelos()
except FileNotFoundError as erro:
    st.error(
        "Não encontrei os arquivos .joblib nesta pasta. "
        "Gere-os no notebook (seção 7) e copie para app_streamlit/.\n\n"
        f"Detalhe: {erro}"
    )
    st.stop()


def preparar_entrada(array_28x28):
    """Mesmo pré-processamento usado no treino: achatar 28x28 -> 784 e normalizar 0-1."""
    vetor = np.asarray(array_28x28, dtype="float32").reshape(1, -1) / 255.0
    return vetor


def recortar_e_centralizar(imagem_bytes, tamanho_final=28, tamanho_caractere=20, limiar=20):
    """Reproduz a convenção do EMNIST/MNIST: o traço é recortado (bounding box),
    redimensionado para caber numa caixa de `tamanho_caractere` px e colado
    centralizado num quadro `tamanho_final`x`tamanho_final`.

    Sem isso, o resize direto do canvas (280x280 -> 28x28) feito pelo componente
    deixa o traço em posição/escala muito diferente das imagens de treino, e os
    modelos (que trabalham em cima dos pixels brutos, sem invariância de posição)
    erram bastante mesmo desenhando o símbolo certo."""
    imagem_alta_res = Image.open(BytesIO(imagem_bytes)).convert("L")
    arr = np.array(imagem_alta_res)
    mascara = arr > limiar
    if not mascara.any():
        return Image.new("L", (tamanho_final, tamanho_final), color=0)

    ys, xs = np.where(mascara)
    y0, y1 = ys.min(), ys.max()
    x0, x1 = xs.min(), xs.max()
    recorte = imagem_alta_res.crop((x0, y0, x1 + 1, y1 + 1))

    largura, altura = recorte.size
    escala = tamanho_caractere / max(largura, altura)
    nova_largura = max(1, round(largura * escala))
    nova_altura = max(1, round(altura * escala))
    recorte = recorte.resize((nova_largura, nova_altura), Image.LANCZOS)

    tela = Image.new("L", (tamanho_final, tamanho_final), color=0)
    pos_x = (tamanho_final - nova_largura) // 2
    pos_y = (tamanho_final - nova_altura) // 2
    tela.paste(recorte, (pos_x, pos_y))
    return tela.filter(ImageFilter.GaussianBlur(radius=0.6))


def rodar_canvas(titulo, chave, modelo):
    st.subheader(titulo)
    resultado = st_mnist_canvas(key=chave)
    if resultado is not None and resultado.is_submitted:
        imagem_28x28 = recortar_e_centralizar(resultado.raw_image_bytes)
        imagem = np.array(imagem_28x28)
        # OBS: se as previsões saírem sempre erradas, é provável que as cores do
        # canvas estejam invertidas em relação ao treino (traço branco em fundo
        # preto, como no MNIST). Nesse caso, troque a linha abaixo por:
        #   imagem = 255 - imagem
        entrada = preparar_entrada(imagem)
        predicao = modelo.predict(entrada)[0]
        st.image(imagem, caption="Imagem enviada (recortada/centralizada, 28x28)", width=140)
        st.success(f"Classificado como: **{predicao}**")


coluna_vf, coluna_15, coluna_ae = st.columns(3)

with coluna_vf:
    rodar_canvas("Verdadeiro / Falso", "canvas_vf", modelo_vf)

with coluna_15:
    rodar_canvas("Dígito de 1 a 5", "canvas_15", modelo_15)

with coluna_ae:
    rodar_canvas("Letra de A a E", "canvas_ae", modelo_ae)
