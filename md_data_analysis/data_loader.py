import pandas as pd
from functools import lru_cache
from pathlib import Path

# Usamos @lru_cache para garantir que o arquivo seja lido do disco apenas uma vez.
@lru_cache(maxsize=None)
def load_dataframe(file_path: Path) -> pd.DataFrame:
    """
    Carrega o DataFrame a partir de um arquivo CSV e realiza o pré-processamento inicial.
    O resultado é armazenado em cache para evitar leituras repetidas do disco.
    """
    try:
        df = pd.read_csv(file_path)
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
        df['ano'] = df['data_ocorrencia'].dt.year
        df['mes'] = df['data_ocorrencia'].dt.month
        df['dia_semana'] = df['data_ocorrencia'].dt.dayofweek
        df['hora'] = df['data_ocorrencia'].dt.hour
        print("DataFrame carregado e preparado com sucesso.")
        return df
    except FileNotFoundError:
        print(f"Erro: O arquivo de dados não foi encontrado em {file_path}")
        # Retorna um DataFrame vazio para evitar que a aplicação quebre na inicialização
        return pd.DataFrame()
