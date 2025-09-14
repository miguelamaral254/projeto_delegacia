import pandas as pd

def analyze_all_occurrences(df: pd.DataFrame, tipo_crime: str = None, bairro: str = None):
    df_filtrado = df.copy()
    lat_min, lat_max = -8.3, -7.9
    lon_min, lon_max = -35.1, -34.8
    df_filtrado = df_filtrado[
        (df_filtrado['latitude'] > lat_min) & (df_filtrado['latitude'] < lat_max) &
        (df_filtrado['longitude'] > lon_min) & (df_filtrado['longitude'] < lon_max)
    ]
    if tipo_crime:
        df_filtrado = df_filtrado[df_filtrado['tipo_crime'] == tipo_crime]
    if bairro:
        df_filtrado = df_filtrado[df_filtrado['bairro'] == bairro]
    cols_to_return = ['id_ocorrencia', 'latitude', 'longitude', 'tipo_crime', 'bairro', 'data_ocorrencia']
    return df_filtrado[cols_to_return].to_dict(orient='records')