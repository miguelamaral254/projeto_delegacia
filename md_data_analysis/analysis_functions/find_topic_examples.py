import pandas as pd
from typing import List

def find_topic_examples(df: pd.DataFrame, keywords: List[str], bairro: str = None, limit: int = 5):
    """
    Encontra os melhores exemplos de ocorrências para um tópico,
    pontuando-os pela quantidade de palavras-chave correspondentes.
    """
    df_filtrado = df.copy()

    if bairro:
        df_filtrado = df_filtrado[df_filtrado['bairro'].str.contains(bairro, case=False, na=False)]

    df_filtrado = df_filtrado.dropna(subset=['descricao_modus_operandi'])

    if df_filtrado.empty:
        return []

    # Função para contar quantas palavras-chave aparecem em um texto
    def count_matches(text, keywords_set):
        return sum(1 for word in text.lower().split() if word in keywords_set)

    keywords_set = set(kw.lower() for kw in keywords)

    # Cria uma nova coluna com a pontuação de correspondência
    df_filtrado['topic_score'] = df_filtrado['descricao_modus_operandi'].apply(
        lambda text: count_matches(text, keywords_set)
    )

    # Filtra apenas as ocorrências que correspondem a pelo menos uma palavra-chave
    exemplos_df = df_filtrado[df_filtrado['topic_score'] > 0]

    # Ordena pelos mais relevantes (maior pontuação) e pega os top 5
    top_exemplos = exemplos_df.sort_values(by='topic_score', ascending=False).head(limit)
    
    return top_exemplos[['id_ocorrencia', 'tipo_crime', 'descricao_modus_operandi']].to_dict('records')
