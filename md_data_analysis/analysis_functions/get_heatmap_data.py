import pandas as pd

def analyze_heatmap_data(df: pd.DataFrame, bairro: str = None, hora: int = None, tipo_crime: str = None, dia_semana: int = None,
                         ano: int = None, mes: int = None):
    df_filtrado = df.copy()
    if bairro:
        df_filtrado = df_filtrado[df_filtrado['bairro'].str.contains(bairro, case=False, na=False)]
    if hora is not None:
        df_filtrado = df_filtrado[df_filtrado['hora'] == hora]
    if tipo_crime:
        df_filtrado = df_filtrado[df_filtrado['tipo_crime'] == tipo_crime]
    if dia_semana is not None:
        df_filtrado = df_filtrado[df_filtrado['dia_semana'] == dia_semana]
    if ano is not None:
        df_filtrado = df_filtrado[df_filtrado['ano'] == ano]
    if mes is not None:
        df_filtrado = df_filtrado[df_filtrado['mes'] == mes]
    heatmap_data = df_filtrado.groupby(['bairro', 'hora']).size().reset_index(name='ocorrencias')
    return heatmap_data.sort_values('ocorrencias', ascending=False).to_dict(orient='records')