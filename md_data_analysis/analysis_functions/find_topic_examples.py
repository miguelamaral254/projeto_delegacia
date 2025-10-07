import pandas as pd
from typing import List

def find_topic_examples(df: pd.DataFrame, keywords: List[str], bairro: str = None, limit: int = 5):
    df_filtrado = df.copy()
    if bairro:
        df_filtrado = df_filtrado[df_filtrado['bairro'].str.contains(bairro, case=False, na=False)]
    df_filtrado = df_filtrado.dropna(subset=['descricao_modus_operandi'])

    if df_filtrado.empty:
        return []
    def count_matches(text, keywords_set):
        return sum(1 for word in text.lower().split() if word in keywords_set)

    keywords_set = set(kw.lower() for kw in keywords)
    df_filtrado['topic_score'] = df_filtrado['descricao_modus_operandi'].apply(
        lambda text: count_matches(text, keywords_set)
    )
    exemplos_df = df_filtrado[df_filtrado['topic_score'] > 0]
    top_exemplos = exemplos_df.sort_values(by='topic_score', ascending=False).head(limit)
    
    return top_exemplos[['id_ocorrencia', 'tipo_crime', 'descricao_modus_operandi']].to_dict('records')
