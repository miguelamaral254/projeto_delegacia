import pandas as pd

def analyze_unique_years(df: pd.DataFrame):
    if 'ano' in df.columns:
        return sorted(df['ano'].dropna().unique().tolist())
    return []