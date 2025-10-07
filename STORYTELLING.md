## 1. Análise de Tópicos do Modus Operandi (Função: `analyze_text_topics`)

### Storytelling: Desvendando a Mente do Criminoso

Imagine uma montanha de relatórios de ocorrências, cada um com uma descrição do que aconteceu. Se lermos um por um, é fácil perder o padrão. Mas e se pudéssemos destilar milhares dessas narrativas em **alguns temas-chave**?

Essa funcionalidade é o "scanner de padrões" que a Inteligência Policial precisa. Ao invés de um analista ler sobre "homem armado entrando em carro e fugindo" e "dois indivíduos em moto que abordaram a vítima no farol", o sistema identifica que ambos os casos se enquadram em um tópico maior, como **"Roubo de Veículos com Uso de Arma de Fogo e Fuga Rápida"**.

Ele transforma a complexidade do texto em *insights* acionáveis, ajudando a traçar o perfil de grupos criminosos e a antecipar suas próximas ações.

| Detalhes Técnicos     | Descrição                                                                                                                                                     |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **O que faz?**        | Identifica os temas principais (tópicos) presentes nas descrições de "modus operandi" das ocorrências.                                                        |
| **Dados de Entrada**  | Coluna `descricao_modus_operandi` de um DataFrame Pandas.                                                                                                     |
| **Modelo/Técnica**    | **Modelagem de Tópicos (Topic Modeling) - LDA (Latent Dirichlet Allocation)**.                                                                                |
| **Tipo de ML**        | **Não Supervisionado** (encontra padrões sem a necessidade de dados rotulados).                                                                               |
| **Informações Úteis** | As palavras-chave de cada tópico (ex: `moto`, `arma`, `farol`) podem orientar treinamentos, alertas para a população e a criação de patrulhas especializadas. |

---

### LGPD/Ética, Reprodutibilidade e Documentação

* **LGPD/Ética:** O LDA lida com a **descrição da ação criminosa**, que é uma informação sensível, mas não processa diretamente dados pessoais identificáveis (como nome ou RG da vítima/suspeito), o que reduz o risco de *compliance* nessa etapa. A preocupação ética está em como os tópicos são usados. Um tópico como "Assaltos a Idosos" deve ser tratado como um **padrão operacional**, e não para estigmatizar bairros ou grupos.
* **Reprodutibilidade:** O `random_state=42` no `LatentDirichletAllocation` garante que, ao rodar a função com os mesmos dados, os resultados dos tópicos serão **sempre os mesmos**, crucial para auditoria e validação de *insights*.
* **Documentação:** Os parâmetros `n_topics` e `n_keywords` permitem ajustar a granularidade da análise, sendo transparentes sobre a forma como o modelo "enxergou" os dados.

---

## 2. Detecção de Anomalias Criminais (Função: `find_anomalous_crimes`)

### Storytelling: O "Ponto Fora da Curva" que Indica Novas Ameaças

A maioria dos crimes segue um padrão: roubos de carros à noite, furtos em lojas durante o dia, etc. Mas o que acontece quando um roubo de bicicleta ocorre às 3 da manhã, no meio de um bairro residencial tranquilo, usando um martelo como arma? Esse é um **evento anômalo**.

Essa funcionalidade atua como um "sistema de alerta precoce". Ela vasculha os dados para encontrar ocorrências que **fogem drasticamente do padrão normal** de crimes na região, cruzando variáveis como local, tipo de arma, horário e dia da semana.

Essas anomalias podem significar: 1) Um novo tipo de crime, 2) Um grupo criminoso inovando o *modus operandi*, ou 3) Simplesmente um erro de registro. Em qualquer caso, a anomalia merece uma **atenção imediata** da Inteligência.

| Detalhes Técnicos     | Descrição                                                                                                                                       |
| :-------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------- |
| **O que faz?**        | Identifica ocorrências que são estatisticamente raras ou inesperadas, comparadas ao restante do histórico.                                      |
| **Dados de Entrada**  | `bairro`, `arma_utilizada`, `quantidade_vitimas`, `quantidade_suspeitos`, `hora`, `dia_semana`.                                                 |
| **Modelo/Técnica**    | **Isolation Forest**.                                                                                                                           |
| **Tipo de ML**        | **Não Supervisionado** (especificamente para detecção de anomalias/outliers).                                                                   |
| **Informações Úteis** | O parâmetro `contamination=0.01` define que se espera que 1% dos dados sejam anômalos. O `anomaly_score` indica o quão estranha a ocorrência é. |

---

### LGPD/Ética, Reprodutibilidade e Documentação

* **LGPD/Ética:** A anomalia deve ser usada para **orientar a investigação**, e não para pré-julgar indivíduos. A LGPD exige que a análise seja proporcional e justificada. A detecção de um "padrão estranho" é uma justificativa forte para dedicar mais recursos investigativos à ocorrência em questão.
* **Reprodutibilidade:** O `random_state=42` garante a repetição dos resultados. O modelo `IsolationForest` é sensível à aleatoriedade, então o controle da *seed* é crucial.
* **Documentação:** A seleção das *features* (`bairro`, `arma`, `hora`, etc.) está explícita. O uso da codificação *One-Hot Encoding* (`pd.get_dummies`) é a forma padrão de transformar dados categóricos em numéricos para o modelo, sendo transparente no fluxo de dados.

---

## 3. Rede de Similaridade de Ocorrências (Função: `generate_similarity_network`)

### Storytelling: Conectando os Pontos para Expor o Grupo

No mundo da investigação, os criminosos raramente agem isoladamente. Uma série de assaltos com características semelhantes (ocorrências no mesmo horário, dia da semana e com a mesma arma) pode indicar a **atuação de um mesmo grupo ou indivíduo**.

Essa funcionalidade cria uma **rede visual** onde cada ocorrência é um "nó" e as conexões ("arestas") representam a similaridade entre elas. A força da conexão (o *score*) aumenta se os crimes aconteceram com a mesma arma, no mesmo dia da semana e com pouca diferença de horário.

Essa rede é uma ferramenta poderosa para a **polícia judiciária**, que pode rapidamente isolar *clusters* de crimes interligados, priorizando a investigação sobre um único grupo que pode ser responsável por dezenas de ocorrências.

| Detalhes Técnicos     | Descrição                                                                                                                                                                         |
| :-------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **O que faz?**        | Constrói uma rede (grafo) que liga ocorrências com alta similaridade de *modus operandi* (arma, hora, dia da semana) em um bairro e tipo de crime específicos.                    |
| **Dados de Entrada**  | `bairro`, `tipo_crime`, `hora`, `arma_utilizada`, `dia_semana`.                                                                                                                   |
| **Modelo/Técnica**    | **Análise de Redes (Network Analysis) / Similaridade Baseada em Regras (Rule-Based Similarity)**. Não é um modelo de Machine Learning tradicional.                                |
| **Tipo de ML**        | **Não se aplica** (É uma análise estatística/regra de negócio).                                                                                                                   |
| **Informações Úteis** | A regra de similaridade é clara: *score* ≥ 3 (com 3 fatores de similaridade) gera uma conexão. Permite identificar "células" ou "séries" criminais para investigação concentrada. |

---

### LGPD/Ética, Reprodutibilidade e Documentação

* **LGPD/Ética:** O uso da rede deve ser estritamente para **fins investigativos**. É essencial que a visualização da rede não seja divulgada sem o devido cuidado com o anonimato das vítimas e dos detalhes da ocorrência. O foco é no padrão, não no detalhe individual.
* **Reprodutibilidade:** As regras de similaridade são **determinísticas**. Dado o mesmo conjunto de dados, a rede gerada será idêntica, garantindo a rastreabilidade da conclusão investigativa.
* **Documentação:** A lógica do *score* (diferença de 2 horas, mesma arma, mesmo dia da semana) está clara no código, tornando o critério de ligação entre os nós totalmente transparente.

---

## 4. Análise de Hotspots Geográficos e Alocação de Patrulha (Funções: `cluster_hotspots` e `simulate_patrol_allocation`)

### Storytelling: Otimizando a Presença Policial com Ciência

O policiamento não pode ser aleatório; deve ser **inteligente e preditivo**.

#### a) Clusterização de Hotspots (`cluster_hotspots`)

Esta funcionalidade mapeia os locais exatos onde os crimes estão concentrados (os *hotspots*). Em vez de patrulhar um bairro inteiro, é possível focar em um **pequeno grupo de esquinas** ou quarteirões onde a incidência é maior, em um determinado horário. É a mira laser do policiamento preventivo.

#### b) Simulação de Alocação de Patrulha (`simulate_patrol_allocation`)

O desafio final é: **Onde colocar as viaturas?** Esta função simula diferentes estratégias de alocação de patrulhas dentro do bairro, desde o aleatório até o mais sofisticado:

1. **Heurística (Baseada em Risco):** Coloca as viaturas nas áreas com o **maior número de crimes**.
2. **Heurística + DBSCAN (Híbrida):** Prioriza os **hotspots de crimes** encontrados (clusters) e, se sobrar viaturas, aloca nas áreas de maior risco.
3. **Q-Learning (Aprendizado por Reforço - RL):** Usa uma abordagem de **Inteligência Artificial** que "aprende" a melhor alocação ao longo do tempo, otimizando a cobertura de risco com base em recompensas.

O resultado mostra qual política de patrulhamento cobre a maior parte do risco criminal total, fornecendo uma base científica para a tomada de decisão do comando.

| Detalhes Técnicos             | Descrição                                                                                                                                                                                 |
| :---------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **O que faz?**                | Encontra concentrações geográficas de crimes (hotspots) e simula diferentes políticas de alocação de viaturas para cobrir o máximo de risco.                                              |
| **Dados de Entrada**          | `latitude`, `longitude`, `bairro`, `hora`, `tipo_crime`.                                                                                                                                  |
| **Modelo/Técnica (Hotspots)** | **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**.                                                                                                                 |
| **Modelo/Técnica (Alocação)** | **Q-Learning** (Aprendizado por Reforço - RL), Heurística e Análise de Risco.                                                                                                             |
| **Tipo de ML**                | **Não Supervisionado** (DBSCAN) e **Aprendizado por Reforço (RL)** (Q-Learning).                                                                                                          |
| **Informações Úteis**         | O DBSCAN usa `eps` (raio de busca) e `min_samples` (mínimo de crimes) para definir um hotspot. O Q-Learning usa uma matriz de risco (tabela Q) para encontrar as células mais vantajosas. |

---

### LGPD/Ética, Reprodutibilidade e Documentação

* **LGPD/Ética:** O uso da geolocalização deve ser feito exclusivamente para **mapeamento e alocação de recursos**. Deve-se evitar o uso da ferramenta para **policiamento preditivo de indivíduos** (que levanta questões éticas graves), focando em **policiamento preditivo de locais** (que é o foco aqui). O modelo não deve ser usado para justificar o perfilamento racial ou social, mas sim a cobertura objetiva do risco.
* **Reprodutibilidade:** O DBSCAN depende de parâmetros definidos, e o Q-Learning usa um `np.random.seed(42)` para garantir a reprodutibilidade da simulação e das políticas de alocação.
* **Documentação:** A discretização do bairro em uma grade (`grid_size=5`) e a forma como o risco é calculado por célula (contagem simples de crimes) são explicitadas, garantindo que o processo de simulação é transparente.

---

## 5. Visualização de Calor por Bairro e Hora (Função: `analyze_heatmap_data`)

### Storytelling: O Relógio e o Mapa do Crime

Se o comando policial tem uma pergunta-chave: "Quando e onde o crime está mais quente?", essa funcionalidade fornece a resposta. Ela gera os dados brutos para criar um **Mapa de Calor** (*Heatmap*) que cruza o **bairro** com a **hora do dia**.

Esse *insight* é fundamental para o **planejamento de turnos**. Se um bairro tem um pico de furtos entre 10h e 12h e outro entre 17h e 19h, a polícia pode ajustar a presença de viaturas para coincidir com esses horários de risco máximo, ao invés de manter uma distribuição homogênea. É a base para um policiamento **sincronizado com o ritmo da cidade**.

| Detalhes Técnicos     | Descrição                                                                                                                                                |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **O que faz?**        | Calcula a contagem de ocorrências agrupadas por `bairro` e `hora`, gerando os dados para a visualização de um Mapa de Calor.                             |
| **Dados de Entrada**  | `bairro` e `hora`. Filtros opcionais: `tipo_crime`, `dia_semana`, `ano`, `mes`.                                                                          |
| **Modelo/Técnica**    | **Análise Estatística/Agregação (Grouping and Counting)**.                                                                                               |
| **Tipo de ML**        | **Não se aplica**.                                                                                                                                       |
| **Informações Úteis** | É o recurso mais simples e direto, fornecendo uma visão macro da distribuição espaço-temporal do crime para otimizar a distribuição de recursos humanos. |

---

### LGPD/Ética, Reprodutibilidade e Documentação

* **LGPD/Ética:** Ao trabalhar apenas com dados de **contagem** por `bairro/hora`, o risco de exposição de dados pessoais é **mínimo**. A análise é puramente estatística e geográfica.
* **Reprodutibilidade:** A agregação é **determinística**. O resultado é sempre o mesmo para o mesmo conjunto de dados.
* **Documentação:** A função é simples e de fácil compreensão, atuando como um filtro e um agregador direto, não necessitando de documentação complexa de modelos.

---

## 6. Busca de Exemplos por Palavra-Chave (Função: `find_topic_examples`)

### Storytelling: A Prova Concreta dos Tópicos

Depois que a funcionalidade de Análise de Tópicos (LDA) gera as palavras-chave, o analista precisa ver: "Onde estão os crimes reais que sustentam esse tópico?"

Esta funcionalidade é a **conexão final entre o modelo abstrato e a realidade do caso**. Dado um conjunto de *keywords* (ex: `moto`, `celular`, `entrega`), ela vasculha as descrições de `modus_operandi` para encontrar os casos que mais se encaixam nesse perfil, ranqueando-os pela quantidade de palavras-chave encontradas.

Isso fornece ao analista o **exemplo mais puro** do tópico identificado, permitindo que ele valide o resultado do modelo e use o caso real como **evidência** para treinar novos investigadores ou para escrever um relatório.

| Detalhes Técnicos     | Descrição                                                                                                                                   |
| :-------------------- | :------------------------------------------------------------------------------------------------------------------------------------------ |
| **O que faz?**        | Localiza as ocorrências cuja descrição do *modus operandi* mais se assemelha a uma lista de palavras-chave fornecida.                       |
| **Dados de Entrada**  | Lista de `keywords` (palavras-chave), `descricao_modus_operandi`.                                                                           |
| **Modelo/Técnica**    | **Contagem de Palavras (Score)**.                                                                                                           |
| **Tipo de ML**        | **Não se aplica** (É uma análise de texto simples baseada em regras de pontuação).                                                          |
| **Informações Úteis** | O `topic_score` é a contagem de palavras-chave do tópico que aparecem na descrição, facilitando a priorização dos exemplos mais relevantes. |

---

### LGPD/Ética, Reprodutibilidade e Documentação

* **LGPD/Ética:** Os resultados incluem a descrição do `modus_operandi`, uma informação sensível. O acesso a esses exemplos deve ser restrito e utilizado apenas por **pessoal de segurança pública autorizado** e para fins de treinamento/investigação, em conformidade com as políticas de privacidade de dados criminais.
* **Reprodutibilidade:** O cálculo do *score* (contagem de palavras) é **determinístico**.
* **Documentação:** A lógica do *match* é clara: é uma contagem simples de correspondências de palavras, o que a torna robusta e fácil de auditar.
