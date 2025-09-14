import sys
from pathlib import Path
from fastapi import APIRouter, Depends
from pydantic import BaseModel
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from md_model.predictor import CrimePredictor 
from md_data_analysis.data_loader import load_dataframe
from md_data_analysis.analysis_functions.predict_hotspots import cluster_hotspots

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PIPELINE_PATH = BASE_DIR / "artifacts" / "lightgbm_model.joblib"
DATA_PATH = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"

router = APIRouter(prefix="/predict", tags=["Predictions"])
predictor = CrimePredictor(model_pipeline_path=MODEL_PIPELINE_PATH)

def get_df():
    return load_dataframe(DATA_PATH)

class OcorrenciaInput(BaseModel):
    bairro: str; descricao_modus_operandi: str; arma_utilizada: str
    sexo_suspeito: str; orgao_responsavel: str; status_investigacao: str
    quantidade_vitimas: int; quantidade_suspeitos: int; idade_suspeito: int
    latitude: float; longitude: float; ano: int; mes: int; dia_semana: int; hora: int

class HotspotInput(BaseModel):
    bairro: str; hora: int; n_hotspots: int = 3

@router.post("")
def predict_crime(ocorrencia: OcorrenciaInput):
    return predictor.predict(ocorrencia.dict())

@router.post("/hotspots")
def predict_crime_hotspots(data: HotspotInput, df: pd.DataFrame = Depends(get_df)):
    return cluster_hotspots(df=df, bairro=data.bairro, hora=data.hora, n_clusters=data.n_hotspots)