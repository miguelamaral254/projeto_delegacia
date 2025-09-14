import pandas as pd

def analyze_top_bairros(df: pd.DataFrame, limit: int = 10):
    top_bairros = df['bairro'].value_counts().head(limit).reset_index()
    top_bairros.columns = ['bairro', 'ocorrencias']
    return top_bairros.to_dict(orient='records')