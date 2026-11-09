# Classificação de radiografias de tórax com VGG19

Aplicação local em Streamlit desenvolvida para demonstrar o modelo VGG19 do TCC de Igor Rayciki Figueredo, da UNESC.

Permite enviar uma imagem por upload ou arrastar e soltar, visualizar a radiografia e obter a classe prevista: **COVID-19**, **Pneumonia** ou **Saudável**, acompanhada das probabilidades do modelo.

> Protótipo acadêmico. As previsões não constituem diagnóstico médico. O aplicativo não verifica se a imagem enviada é uma radiografia.

## Arquivos necessários

Mantenha estes três arquivos na mesma pasta:

- `app.py`
- `melhor_vgg19_finetuning.keras`
- `README.md`

O modelo já foi treinado. Não é necessário executar os scripts de treinamento nem disponibilizar o conjunto de dados para utilizar a interface.

No `app.py`, o caminho do modelo deve estar configurado assim:

```python
RAIZ = Path(__file__).resolve().parent
CAMINHO_MODELO = RAIZ / "melhor_vgg19_finetuning.keras"
```

## Requisitos

- Python 3.10, preferencialmente o mesmo ambiente utilizado no treinamento.
- TensorFlow 2.20.0.
- Streamlit.
- NumPy.
- pandas.

Não é necessário utilizar uma GPU para executar a classificação.

## Instalação

Abra a pasta da aplicação no VS Code e selecione **Terminal → Novo Terminal**.

Confira a versão do Python:

```powershell
python --version
```
## Instale as dependências

abra o terminal nessa parte e execute: 

```powershell
python -m pip install -requirements.txt
```

Aguarde a instalação das bibliotecas, Esse procedimento é necessário apenas na primeira utilização ou ao configurar outro ambiente Python 

### Ambiente já configurado

Se o TensorFlow já estiver instalado no ambiente utilizado:

```powershell
python -m pip install streamlit numpy pandas
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Novo ambiente virtual

Se precisar instalar todas as dependências, utilize os comandos abaixo. É necessário ter o Python 3.10 instalado:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install tensorflow==2.20.0 streamlit numpy pandas
```

No VS Code, utilize **Python: Select Interpreter** para selecionar o ambiente `.venv`.

## Executar a aplicação

No terminal, dentro da pasta onde está o `app.py`, execute:

```powershell
python -m streamlit run app.py
```

Se criou o ambiente virtual acima, utilize:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Na primeira execução, o Streamlit pode solicitar um e-mail. O preenchimento é opcional: deixe em branco e pressione **Enter**.

Abra o endereço informado como **Local URL**, normalmente:

http://localhost:8501

Para utilizar outra porta:

```powershell
python -m streamlit run app.py --server.port 8502
```

Nesse caso, acesse:

http://localhost:8502

Se estiver utilizando o ambiente virtual, substitua `python` por `.\.venv\Scripts\python.exe`.

Mantenha o terminal aberto durante o uso. Para encerrar a aplicação, pressione **Ctrl + C**.

## Como utilizar

1. Arraste uma imagem para a área de envio ou clique para selecionar.
2. Utilize um arquivo PNG, JPG ou JPEG de até **10 MB**.
3. Confira a imagem exibida.
4. Clique em **Classificar imagem**.
5. Consulte a classe prevista e as probabilidades das três classes.
6. Para realizar outra análise, remova o arquivo e envie uma nova imagem.

A primeira classificação pode demorar mais porque o modelo precisa ser carregado. Depois, ele permanece em cache durante a execução.

## imagens exemplos

A pasta "Exemplos" contém seis imagens para demonstração, organizadas conforme os rótulos da base de dados. Para utilizá-las, selecione uma imagem pelo upload da aplicação. A previsão do modelo pode divergir do rótulo indicado pela pasta.

## Pré-processamento

A imagem é carregada com três canais, redimensionada para **224 × 224 pixels** por interpolação bilinear e convertida para `float32`.

O pré-processamento específico da VGG19 já está incorporado ao modelo. Portanto, o aplicativo não deve aplicar `preprocess_input` novamente nem dividir os pixels por 255.

A inferência utiliza `training=False`, desativando o comportamento de treinamento das camadas de aumento de dados e Dropout.

A ordem das classes é:

1. COVID-19.
2. Pneumonia.
3. Saudável.

Essa ordem deve corresponder à utilizada no treinamento.

## Problemas comuns

### Modelo não encontrado

Verifique se `melhor_vgg19_finetuning.keras` está na mesma pasta do `app.py` e se o nome está correto.

### Biblioteca não encontrada

Instale as dependências no mesmo ambiente Python utilizado para executar a aplicação.

Para conferir o executável:

```powershell
python -c "import sys; print(sys.executable)"
```

### Página sem conteúdo

Verifique se o arquivo foi salvo e se existem erros no terminal. Confira também o endereço e a porta utilizados.

Recarregue o navegador com **Ctrl + F5**.

### Erro ao carregar o modelo

Confira se o arquivo `.keras` está completo e utilize, preferencialmente, o mesmo ambiente do treinamento.

### Erro de indentação

Utilize quatro espaços por nível de recuo nos blocos `if`, `with`, `try` e `except`.

## Conferência do funcionamento

Compare as previsões de algumas imagens do conjunto de teste com os resultados registrados em `predicoes_vgg19.csv`.

Essa conferência permite verificar a integração entre o modelo e a interface, mas não constitui validação clínica.

## Limitações

O aplicativo sempre seleciona uma das três classes, mesmo quando recebe uma imagem inadequada. Ele não possui uma classe de rejeição.

As probabilidades apresentadas são saídas do modelo e não representam probabilidades clínicas de doença. Uma previsão de “Saudável” não exclui a presença de doenças.

O desempenho observado no TCC não demonstra generalização para outras instituições ou condições de aquisição. O sistema destina-se à demonstração acadêmica.

## Autor

Igor Rayciki Figueredo  
Ciência da Computação — UNESC.