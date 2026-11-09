from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Classificação de radiografias",
    page_icon="🩻",
    layout="wide"
)

st.markdown(
    """
    <style>
        .stMarkdown p,
        [data-testid="stAlert"] p {
            font-size: 18px;
        }

        [data-testid="stFileUploader"] label p {
            font-size: 18px;
        }

        .stButton button p {
            font-size: 18px;
        }

        [data-testid="stCaptionContainer"] p {
            font-size: 15px;
        }

        [data-testid="stMetricLabel"] p {
            font-size: 24px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 46px !important;
        }

        .tabela-probabilidades {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }

        .tabela-probabilidades th,
        .tabela-probabilidades td {
            font-size: 22px !important;
            padding: 16px !important;
            text-align: left;
            border-bottom: 1px solid #888;
        }

        .tabela-probabilidades th {
            font-weight: bold;
        }

        h1 {
            font-size: 44px !important;
        }

        .stMarkdown p,
        [data-testid="stAlert"] p {
            font-size: 22px !important;
            line-height: 1.5 !important;
        }

        [data-testid="stFileUploader"] label p {
            font-size: 22px !important;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploader"] button p {
            font-size: 18px !important;
        }
        </style>
        """,
        unsafe_allow_html=True
)

with st.spinner("Importando bibliotecas..."):
    import numpy as np
    import pandas as pd
    import tensorflow as tf


RAIZ = Path(__file__).resolve().parent

CLASSES = ["COVID-19", "Pneumonia", "Saudável"]

CAMINHO_MODELO = RAIZ / "melhor_vgg19_finetuning.keras"


@st.cache_resource
def carregar_modelo(caminho, modificacao):
    return tf.keras.models.load_model(
        caminho,
        compile=False
    )


def preparar_imagem(conteudo):
    imagem = tf.io.decode_image(
        conteudo,
        channels=3,
        expand_animations=False
    )

    imagem = tf.image.resize(
        imagem,
        size=(224, 224),
        method="bilinear"
    )

    imagem = tf.cast(imagem, tf.float32)

    return tf.expand_dims(imagem, axis=0)


st.title("Classificação de radiografias de tórax")
st.write("Demonstração da VGG19 desenvolvida no TCC.")

st.info(
    "Protótipo acadêmico: apresenta previsões experimentais "
    "e não fornece diagnóstico médico. "
    "O sistema não verifica se a imagem enviada é uma radiografia."
)

if not CAMINHO_MODELO.is_file():
    st.error(f"Modelo não encontrado: {CAMINHO_MODELO}")
    st.stop()

arquivo = st.file_uploader(
    "Arraste uma imagem aqui ou clique para selecionar",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=False,
    help="Envie uma imagem PNG ou JPG de até 10 MB."
)

if arquivo is not None:
    conteudo = arquivo.getvalue()

    if len(conteudo) > 10 * 1024 * 1024:
        st.error("Envie uma imagem de até 10 MB.")
        st.stop()

    try:
        entrada = preparar_imagem(conteudo)
    except (tf.errors.OpError, ValueError):
        st.error("Não foi possível ler a imagem enviada.")
        st.stop()

    esquerda, direita = st.columns(2)

    with esquerda:
        st.subheader("Imagem enviada")

        st.image(
            conteudo,
            caption=arquivo.name,
            width=450
        )

    with direita:
        st.subheader("Resultado")

        if st.button("Classificar imagem", type="primary"):
            try:
                with st.spinner("Analisando a imagem..."):
                    modelo = carregar_modelo(
                        str(CAMINHO_MODELO),
                        CAMINHO_MODELO.stat().st_mtime_ns
                    )

                    probabilidades = np.asarray(
                        modelo(entrada, training=False)
                    )[0]

                if (
                    probabilidades.shape != (3,)
                    or not np.isfinite(probabilidades).all()
                ):
                    raise ValueError(
                        "A saída do modelo não corresponde "
                        "às três classes esperadas."
                    )

                indice = int(np.argmax(probabilidades))

                st.metric(
                    label="Classe prevista",
                    value=CLASSES[indice]
                )

                tabela = pd.DataFrame({
                    "Classe": CLASSES,
                    "Probabilidade do modelo": [
                        f"{valor:.2%}"
                        for valor in probabilidades
                    ]
                })

                st.markdown(
                    tabela.to_html(
                        index=False,
                        classes="tabela-probabilidades",
                        border=0,
                        escape=True
                    ),
                    unsafe_allow_html=True
                )

                st.caption(
                    "As probabilidades são saídas do modelo; "
                    "não representam a probabilidade clínica "
                    "de o paciente ter uma doença."
                )

            except Exception as erro:
                st.error(
                    "Não foi possível executar a classificação. "
                    "Confira o modelo e o ambiente TensorFlow utilizado."
                )
                st.code(str(erro))