import pandas as pd


class DataAnalyzer:
    def __init__(self, file_path: str):
        self.df = pd.read_csv(file_path)
        self.df['data_ocorrencia'] = pd.to_datetime(self.df['data_ocorrencia'])
        self.df['ano'] = self.df['data_ocorrencia'].dt.year
        self.df['mes'] = self.df['data_ocorrencia'].dt.month
        self.df['dia_semana'] = self.df['data_ocorrencia'].dt.dayofweek
        self.df['hora'] = self.df['data_ocorrencia'].dt.hour
        print("Analisador de dados carregado com sucesso.")

    def get_top_bairros(self, limit: int = 10):
        top_bairros = self.df['bairro'].value_counts().head(limit).reset_index()
        top_bairros.columns = ['bairro', 'ocorrencias']
        return top_bairros.to_dict(orient='records')

    def get_heatmap_data(self, bairro: str = None, hora: int = None, tipo_crime: str = None, dia_semana: int = None,
                         ano: int = None, mes: int = None):
        df_filtrado = self.df.copy()

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

        heatmap_data = heatmap_data.sort_values('ocorrencias', ascending=False)

        return heatmap_data.to_dict(orient='records')

    def get_seasonality_data(self, by: str = 'month'):
        if by == 'day_of_week':
            dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
            season_data = self.df['dia_semana'].map(dias).value_counts().reset_index()
            season_data.columns = ['dia_semana', 'ocorrencias']
        else:
            season_data = self.df.groupby(['ano', 'mes']).size().reset_index(name='ocorrencias')
            season_data = season_data.sort_values(['ano', 'mes'])
        return season_data.to_dict(orient='records')

    def get_unique_crime_types(self):
        crime_types = sorted(self.df['tipo_crime'].unique().tolist())
        return crime_types

    def get_unique_bairros(self):
        bairros = sorted(self.df['bairro'].unique().tolist())
        return bairros

    def get_all_occurrences(self, tipo_crime: str = None, bairro: str = None):
        df_filtrado = self.df.copy()
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

    def get_unique_years(self):
        years = sorted(self.df['ano'].unique().tolist())
        return years