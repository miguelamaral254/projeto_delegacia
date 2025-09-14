import pandas as pd

def analyze_unique_crime_types(df: pd.DataFrame):
    if 'tipo_crime' in df.columns:
        return sorted(df['tipo_crime'].unique().tolist())
    return []