
# Projeto Integrador 5: Delegacia 5.0

## 1\. Visão Geral e Objetivo do Projeto

Este projeto, desenvolvido como parte do **Projeto Integrador 5**, tem como objetivo principal criar um protótipo funcional (PoC) de uma plataforma de análise de dados e **análise preditiva de crimes** para a Polícia Civil de Pernambuco (PC-PE). O público-alvo são as equipes de investigação e os setores de inteligência, que podem utilizar os *insights* gerados para otimizar operações e alocação de recursos.

A **Entrega 1** focou em responder à seguinte pergunta:

> Com os dados de ocorrências disponíveis, é possível construir um modelo de *Machine Learning* que consiga prever com acurácia o tipo de crime?

Nossa hipótese inicial era que, ao analisar características como localização, tempo e, principalmente, a descrição textual do *modus operandi*, seria possível classificar as ocorrências de forma eficaz.

## 2\. Detalhamento Técnico e Fluxo de Trabalho (Pipeline)

Para responder à pergunta central, foi construído um **pipeline de Machine Learning**, uma sequência automatizada de etapas que transforma os dados brutos em um modelo treinado.

### Fluxo de Treinamento

O fluxo de treinamento é orquestrado pelo `main.py`, que executa a sequência de scripts na ordem correta, garantindo a reprodutibilidade.

  * `start_train.py` (ou `main.py`): O script principal que orquestra a execução dos modelos de *baseline* e de alta performance.
  * `train_baseline.py`: Treina um modelo **`DummyClassifier`** para estabelecer a performance mínima. Qualquer modelo funcional deve superar esse *baseline*.
  * `train_randomforest.py`: Treina um modelo **`RandomForestClassifier`**, uma escolha robusta e interpretável, ideal para este tipo de problema.
  * `train_lightgbm.py`: Treina um modelo **`LGBMClassifier`**, um algoritmo de *gradient boosting* conhecido por sua velocidade e alta precisão. Este é o modelo final selecionado para a API.

### Etapas do Pipeline

1.  **Engenharia de Features**: A coluna `data_ocorrencia` é processada para extrair informações valiosas como `ano`, `mes`, `dia_semana` e `hora`.
2.  **Pré-processamento de Dados**: Utilizando o **`ColumnTransformer`**, cada tipo de dado é preparado para o modelo:
      * **Dados Numéricos** (`latitude`, `hora`, etc.): Padronizados com `StandardScaler`.
      * **Dados Categóricos** (`bairro`, `arma_utilizada`): Convertidos em um formato numérico com `OneHotEncoder`.
      * **Dados de Texto** (`descricao_modus_operandi`): Convertidos em vetores numéricos de 1000 posições usando **`TfidfVectorizer`**. Palavras irrelevantes são removidas (*stopwords*), e a importância das palavras-chave é calculada.
3.  **Balanceamento de Classes**: A técnica **SMOTE (`Synthetic Minority Over-sampling Technique`)** é aplicada no conjunto de treino. Ela cria exemplos sintéticos para as classes minoritárias, garantindo que o modelo aprenda a identificar todos os tipos de crime de forma justa.
4.  **Treinamento do Modelo**: O *pipeline* treina os modelos `RandomForest` e `LightGBM` usando uma **divisão temporal** dos dados (80% dos dados mais antigos para treino e 20% dos mais recentes para teste), simulando um cenário real de predição do futuro.

## 3\. Análise de Resultados e Métricas

Após o treinamento, o modelo foi avaliado no conjunto de teste. O desempenho foi medido utilizando um relatório de classificação, que confirma a principal descoberta do projeto.

### Relatório de Classificação (LightGBM)

| Classe              | Precision | Recall | F1-Score | Support |
| ------------------- | --------- | ------ | -------- | ------- |
| Ameaça              | 0.15      | 0.15   | 0.15     | 117     |
| Estelionato         | 0.07      | 0.08   | 0.07     | 93      |
| Estupro             | 0.11      | 0.09   | 0.10     | 85      |
| Furto               | 0.14      | 0.15   | 0.15     | 100     |
| Homicídio           | 0.09      | 0.12   | 0.10     | 94      |
| Latrocínio          | 0.08      | 0.13   | 0.10     | 93      |
| Roubo               | 0.12      | 0.10   | 0.11     | 115     |
| Sequestro           | 0.08      | 0.06   | 0.07     | 98      |
| Tráfico de Drogas   | 0.08      | 0.06   | 0.07     | 99      |
| Violência Doméstica | 0.10      | 0.07   | 0.09     | 107     |
| **Accuracy** | -         | -      | -        | 0.10    |
| **Macro avg** | 0.10      | 0.10   | 0.10     | 1001    |
| **Weighted avg** | 0.10      | 0.10   | 0.10     | 1001    |

### Conclusão da Análise de Métricas

O baixo desempenho generalizado, com uma **acurácia de 10%** e valores F1-Score consistentemente baixos (em torno de 0.10), não representa uma falha no processo de *Machine Learning*, mas sim a principal descoberta desta fase: **o sinal nos dados é fraco**.

As *features* disponíveis no *dataset* não contêm informação distintiva o suficiente para permitir que o modelo diferencie com confiança as 10 classes de crime. Esta descoberta valida o objetivo da **Entrega 1**, que era criar um pipeline funcional e entender a natureza dos dados para guiar os próximos passos do projeto.

-----

## 4\. Documentação da API (FastAPI)

A API foi construída com **FastAPI** para servir tanto as predições do modelo quanto as análises estatísticas, oferecendo uma interface clara e interativa.

A API é dividida em módulos, organizados em roteadores: `statistics`, `predictions` e `occurrences`.

### Endpoints da API

#### `GET /`

  * **Descrição**: Retorna uma mensagem de boas-vindas para verificar se a API está online.
  * **Funcionalidade**: Simplesmente um teste de conectividade.
  * **Exemplo de Resposta**: `{"message": "Bem-vindo à API Delegacia 5.0. Acesse /docs para a documentação interativa."}`

#### `POST /predict`

  * **Descrição**: Preve o tipo de crime com base nos dados de uma nova ocorrência.
  * **Funcionalidade**: Recebe um JSON com os dados de uma ocorrência e utiliza o pipeline treinado do `lightgbm_model.joblib` para prever a classe do crime e suas probabilidades.
  * **Entrada (JSON)**:
    ```json
    {
      "bairro": "CENTRO",
      "descricao_modus_operandi": "Homem armado invadiu a casa.",
      "arma_utilizada": "ARMA DE FOGO",
      "sexo_suspeito": "MASCULINO",
      "orgao_responsavel": "DELEGACIA DE ROUBOS E FURTOS",
      "status_investigacao": "EM ANDAMENTO",
      "quantidade_vitimas": 1,
      "quantidade_suspeitos": 1,
      "idade_suspeito": 30,
      "latitude": -8.05,
      "longitude": -34.9,
      "ano": 2025,
      "mes": 9,
      "dia_semana": 2,
      "hora": 16
    }
    ```
  * **Saída (JSON)**: Retorna o tipo de crime previsto e a probabilidade de cada classe.
    ```json
    {
      "tipo_crime_predito": "Roubo",
      "probabilidades": {
        "Ameaça": 0.05,
        "Estelionato": 0.08,
        "Roubo": 0.25,
        ...
      }
    }
    ```

#### `POST /predict/hotspots`

  * **Descrição**: Identifica *hotspots* de crimes (centros de alta concentração de ocorrências) para um bairro e hora específicos.
  * **Funcionalidade**: Utiliza o algoritmo de clusterização **`KMeans`** para agrupar ocorrências históricas por coordenadas geográficas, encontrando os pontos de maior densidade de crimes para as condições solicitadas.
  * **Entrada (JSON)**:
    ```json
    {
      "bairro": "BOA VIAGEM",
      "hora": 22,
      "n_hotspots": 3
    }
    ```
  * **Saída (JSON)**:
    ```json
    {
      "message": "3 hotspots previstos encontrados.",
      "hotspots": [
        { "lat": -8.123, "lon": -34.912 },
        { "lat": -8.115, "lon": -34.905 },
        { "lat": -8.130, "lon": -34.920 }
      ]
    }
    ```

#### `GET /occurrences`

  * **Descrição**: Retorna uma lista de ocorrências com filtros opcionais.
  * **Funcionalidade**: Permite a visualização de ocorrências no mapa, filtrando por `tipo_crime` ou `bairro`.
  * **Parâmetros de Query**: `?tipo_crime=Roubo` ou `?bairro=BOA VIAGEM`.
  * **Saída (JSON)**: Lista de ocorrências com dados simplificados (`id`, `latitude`, `longitude`, `tipo_crime`, `bairro`, `data_ocorrencia`).

#### `GET /statistics/top-bairros`

  * **Descrição**: Retorna os bairros com o maior número de ocorrências.
  * **Funcionalidade**: Útil para identificar as áreas de maior incidência criminal.
  * **Parâmetros de Query**: `?limit=5` (padrão é 10).
  * **Saída (JSON)**: `[{"bairro": "BOA VIAGEM", "ocorrencias": 500}, ...]`

#### `GET /statistics/crime-heatmap-data`

  * **Descrição**: Retorna dados agregados para a criação de mapas de calor (heatmap).
  * **Funcionalidade**: Permite analisar a distribuição de crimes por `bairro` e `hora`, com múltiplos filtros.
  * **Parâmetros de Query**: `?bairro=SANTO AMARO&hora=18`
  * **Saída (JSON)**: Lista de dados agrupados.

#### `GET /statistics/seasonality`

  * **Descrição**: Analisa a sazonalidade das ocorrências.
  * **Funcionalidade**: Retorna a contagem de crimes por mês ou dia da semana.
  * **Parâmetros de Query**: `?by=day_of_week` (padrão é `month`).
  * **Saída (JSON)**: `[{"dia_semana": "Segunda", "ocorrencias": 1200}, ...]` ou `[{"ano": 2024, "mes": 1, "ocorrencias": 500}, ...]`

#### `GET /statistics/unique-crime-types`

  * **Descrição**: Retorna uma lista de todos os tipos de crime únicos no dataset.
  * **Funcionalidade**: Ajuda a popular menus e filtros na interface do usuário.
  * **Saída (JSON)**: `["Ameaça", "Estelionato", ...]`

#### `GET /statistics/unique-bairros`

  * **Descrição**: Retorna uma lista de todos os bairros únicos no dataset.
  * **Funcionalidade**: Ajuda a popular menus e filtros na interface do usuário.
  * **Saída (JSON)**: `["BOA VIAGEM", "CENTRO", ...]`

#### `GET /statistics/unique-years`

  * **Descrição**: Retorna uma lista de todos os anos únicos no dataset.
  * **Funcionalidade**: Ajuda a popular menus e filtros na interface do usuário.
  * **Saída (JSON)**: `[2023, 2024, ...]`

-----

## 5\. Conclusão e Próximos Passos (Rumo à Entrega 2)

A **Entrega 1** teve como resultado principal não um modelo de alta acurácia, mas a criação de um **pipeline de Machine Learning robusto e funcional**, uma **API completa** e um **insight claro** sobre a natureza dos dados. Esta descoberta orienta perfeitamente as tarefas para a próxima fase.

Para a **Entrega 2**, a estratégia será a seguinte:

1.  **Análise Não Supervisionada**: Utilizar técnicas de **clusterização (`KMeans`)** para encontrar padrões no *modus operandi* e na geolocalização, ignorando os rótulos de tipo de crime.
2.  **Aprimoramento Supervisionado**: Simplificar o problema, agrupando os 10 crimes em 3 ou 4 categorias mais amplas (ex: "Crimes Contra o Patrimônio"). Isso tem um potencial muito maior de gerar um modelo preditivo com utilidade prática.