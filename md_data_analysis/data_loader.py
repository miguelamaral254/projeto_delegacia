import pandas as pd
from pathlib import Path

_df_cache: pd.DataFrame = None

def _load_and_prepare_dataframe(file_path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'], errors='coerce')
        df['ano'] = df['data_ocorrencia'].dt.year
        df['mes'] = df['data_ocorrencia'].dt.month
        df['dia_semana'] = df['data_ocorrencia'].dt.dayofweek
        df['hora'] = df['data_ocorrencia'].dt.hour
        print(f"DataFrame carregado e preparado com sucesso do arquivo: {file_path.name}")
        return df
    except FileNotFoundError:
        print(f"AVISO: Arquivo de dados não encontrado em {file_path}. Um DataFrame vazio será usado.")
        return pd.DataFrame()

def load_initial_dataframe(file_path: Path):
    global _df_cache
    _df_cache = _load_and_prepare_dataframe(file_path)

def reload_dataframe(file_path: Path):
    global _df_cache
    _df_cache = _load_and_prepare_dataframe(file_path)
    print("Cache do DataFrame foi atualizado com sucesso!")

def get_dataframe() -> pd.DataFrame:
    return _df_cache

