import pandas as pd

def analyze_seasonality_data(df: pd.DataFrame, by: str = 'month'):
    if by == 'day_of_week':
        dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
        season_data = df['dia_semana'].map(dias).value_counts().reset_index()
        season_data.columns = ['dia_semana', 'ocorrencias']
    else:
        season_data = df.groupby(['ano', 'mes']).size().reset_index(name='ocorrencias')
        season_data = season_data.sort_values(['ano', 'mes'])
    return season_data.to_dict(orient='records')