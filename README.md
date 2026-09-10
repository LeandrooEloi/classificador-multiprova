# Classificador de 1 a 5, V/F, A-E

Notebook executável no Colab: https://colab.research.google.com/drive/1eb-8zUvCd8Yodb9jVZNxzrnl3XtcrIJu?usp=sharing
Link render dashboard: https://classificador-multiprova.onrender.com

## O que tem aqui

- `classificador_multiprova.ipynb` — notebook para o Colab com o pipeline completo (itens 1 a 7 do enunciado, exceto a UI).
- `app_streamlit/app.py` — data app com os 3 canvas (V/F, 1-5, A-E), cada um acionando o modelo correspondente.
- `app_streamlit/requirements.txt` — dependências do app.
- `relatorios/` — pasta onde o notebook salva um `.csv` do `classification_report` por modelo/dataset.

## Passo a passo

### 1. Rodar o notebook no Colab
1. Acesse https://colab.research.google.com
2. `Arquivo > Fazer upload de notebook` e selecione `classificador_multiprova.ipynb`.
3. Rode as células em ordem (`Ambiente de execução > Executar tudo`).
4. Confira a célula de "conferência visual" (seção 2): se as letras/dígitos aparecerem
   deitados ou espelhados, ative `TRANSPOR_IMAGENS = True` na seção 3 e rode tudo de novo.
5. No fim, a seção 7 gera 3 arquivos `.joblib` (um por dataset) e tenta baixá-los
   automaticamente para o seu computador.
6. Baixe também `resumo_resultados.csv`, `ranking_modelos.csv` e a pasta `relatorios/`
   (aba de arquivos à esquerda no Colab, clique com o botão direito > Download) —
   são as evidências pedidas nos itens 5 e 6 do enunciado.

### 2. Rodar o app Streamlit localmente
1. Copie os 3 arquivos `.joblib` baixados para dentro de `app_streamlit/`.
2. No terminal:
   ```bash
   cd app_streamlit
   pip install -r requirements.txt
   streamlit run app.py
   ```
3. Desenhe em cada canvas e confira a previsão. Se o modelo errar tudo de forma
   consistente, veja o comentário sobre inversão de cores dentro de `app.py`
   (`imagem = 255 - imagem`).

### 3. Apresentação em sala
Leve o notebook executado (com as saídas) + o app rodando localmente, conforme pedido
no enunciado ("apresentar o código e a execução do data app presencial em sala").

### 4. Publicar no Streamlit Community Cloud
1. Confirme que o repositório está no GitHub com os 3 `.joblib` versionados dentro de
   `app_streamlit/` (eles já estão neste repo — arquivos binários de modelo não são
   ignorados pelo `.gitignore`).
2. Acesse https://share.streamlit.io e clique em "New app".
3. Selecione este repositório, branch `main` e em **Main file path** informe
   `app_streamlit/app.py`.
4. O Streamlit Cloud detecta automaticamente o `app_streamlit/requirements.txt`
   (fica na mesma pasta do app) e instala as dependências, já fixadas nas versões
   usadas para treinar os modelos (`scikit-learn==1.6.1`), evitando erro de
   incompatibilidade ao carregar os `.joblib`.
5. Clique em "Deploy". O app usa caminho absoluto (baseado em `__file__`) para achar
   os `.joblib`, então funciona independente do diretório de onde o Streamlit Cloud
   executa o processo.
6. Se atualizar os modelos, copie os novos `.joblib` para `app_streamlit/`, faça commit/push
   — o Streamlit Cloud reimplanta automaticamente a cada push na branch conectada.

### 5. Alternativa: publicar no Render
Este repo já tem um [`render.yaml`](render.yaml) (Blueprint) configurado. Passo a passo:
1. Faça push do repo para o GitHub.
2. Em https://dashboard.render.com, clique em **New > Blueprint** e selecione este repositório.
   O Render lê o `render.yaml` automaticamente e já configura:
   - diretório raiz do serviço: `app_streamlit/` (onde estão o `app.py` e o `requirements.txt`);
   - build: `pip install -r requirements.txt`;
   - start: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true ...`
     (o Render define a porta pela variável `$PORT`, diferente do Streamlit Cloud).
3. Confirme o plano (o `free` funciona, mas "dorme" após inatividade e demora ~1 min para
   acordar na primeira visita) e clique em **Apply**.
4. Sem Blueprint, dá para criar manualmente um **Web Service** apontando pasta raiz para
   `app_streamlit`, com o mesmo build/start command acima.

## Decisões de projeto (para você justificar na entrega)

- **V/F** foi montado com as letras `V` e `F` do split `letters` do EMNIST.
- **1 a 5** foi montado com o split `digits` do EMNIST, filtrando só os rótulos 1-5.
- **A a E** foi montado com o split `letters` do EMNIST, filtrando só as 5 primeiras letras.
- Cada imagem é achatada de 28x28 para um vetor de 784 posições e normalizada para [0,1],
  igual ao procedimento do classificador binário do dígito 5 feito em sala.
- Os datasets de treino/teste são sub-amostrados de forma estratificada
  (`N_TREINO_MAX` / `N_TESTE_MAX` no notebook) só para caber num tempo de execução
  razoável no Colab — aumente esses valores se quiser métricas melhores e tiver tempo.
- O ranking usa `score_ranking = acurácia_teste - |gap_treino_teste|`, ou seja,
  prioriza modelos com boa acurácia de teste e pouca diferença entre treino e teste
  (menos overfitting/underfitting), exatamente como pedido no item 6.
