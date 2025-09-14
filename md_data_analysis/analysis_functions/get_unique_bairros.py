import pandas as pd

def analyze_unique_bairros(df: pd.DataFrame):
    if 'bairro' in df.columns:
        return sorted(df['bairro'].dropna().astype(str).unique().tolist())
    return []