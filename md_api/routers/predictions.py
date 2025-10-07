import sys
from pathlib import Path
from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
import pandas as pd
from typing import Optional

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from md_data_analysis.data_loader import get_dataframe
from md_data_analysis.analysis_functions.predict_hotspots import cluster_hotspots
from md_data_analysis.analysis_functions.find_anomalies import find_anomalous_crimes
from md_model.violence_predictor import ViolencePredictor

router = APIRouter(prefix="/predict", tags=["Predictions"])

VIOLENCE_MODEL_PATH = ROOT / "artifacts" / "violence_predictor_model.joblib"
violence_predictor = ViolencePredictor(model_path=VIOLENCE_MODEL_PATH)

def get_df():
    return get_dataframe()

class HotspotInput(BaseModel):
    bairro: str
    hora: int
    tipo_crime: Optional[str] = None

class CrimeInput(BaseModel):
    bairro: str
    descricao_modus_operandi: str
    arma_utilizada: str
    sexo_suspeito: str
    quantidade_vitimas: int
    quantidade_suspeitos: int
    idade_suspeito: int
    latitude: float
    longitude: float
    ano: int
    mes: int
    dia_semana: int
    hora: int
    orgao_responsavel: str
    status_investigacao: str

@router.post("/violence")
def predict_violence(data: CrimeInput):
    """
    Prevê se uma ocorrência tem características de crime violento ou não violento.
    """
    return violence_predictor.predict(data.dict())

@router.post("/hotspots")
def predict_crime_hotspots(data: HotspotInput, df: pd.DataFrame = Depends(get_df)):
    return cluster_hotspots(
        df=df,
        bairro=data.bairro,
        hora=data.hora,
        tipo_crime=data.tipo_crime
    )

@router.get("/anomalies")
def get_anomalous_events(
    n_results: int = Query(20, gt=0, le=100),
    df: pd.DataFrame = Depends(get_df)
):
    return find_anomalous_crimes(df, n_results=n_results)