# md_core/config.py
from pathlib import Path

# Define o caminho base do projeto
BASE_DIR = Path(__file__).resolve().parent.parent

# Caminhos para dados e artefatos
DATA_PATH = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"
ARTIFACTS_PATH = BASE_DIR / "artifacts"
MODEL_PATH = ARTIFACTS_PATH / "random_forest_model.joblib"
PREPROCESSOR_PATH = ARTIFACTS_PATH / "preprocessor.joblib"

# Colunas do modelo
TARGET = "tipo_crime"
NUM_COLS = ['quantidade_vitimas', 'quantidade_suspeitos', 'idade_suspeito', 'latitude', 'longitude', 'ano', 'mes', 'dia_semana', 'hora']
CAT_COLS = ['bairro', 'descricao_modus_operandi', 'arma_utilizada', 'sexo_suspeito', 'orgao_responsavel', 'status_investigacao']

# Garante que o diretório de artefatos exista
ARTIFACTS_PATH.mkdir(parents=True, exist_ok=True)