# md_data_processing/preprocessor.py
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from md_core.config import NUM_COLS, CAT_COLS, TARGET

def load_data(path):
    """Carrega os dados e realiza a engenharia de features temporais."""
    df = pd.read_csv(path)
    df['data_ocorrencia'] = pd.to_datetime(df['data_ocorrencia'])
    df['ano'] = df['data_ocorrencia'].dt.year
    df['mes'] = df['data_ocorrencia'].dt.month
    df['dia_semana'] = df['data_ocorrencia'].dt.dayofweek
    df['hora'] = df['data_ocorrencia'].dt.hour
    return df

def create_preprocessor():
    """Cria o objeto ColumnTransformer para o pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS)
        ],
        remainder='drop'  # Descarta colunas não especificadas
    )
    return preprocessor