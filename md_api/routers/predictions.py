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

from md_data_analysis.data_loader import load_dataframe
from md_data_analysis.analysis_functions.predict_hotspots import cluster_hotspots
from md_data_analysis.analysis_functions.find_anomalies import find_anomalous_crimes

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_ocorrencias_delegacia_5.csv"

router = APIRouter(prefix="/predict", tags=["Predictions"])

def get_df():
    return load_dataframe(DATA_PATH)

class HotspotInput(BaseModel):
    bairro: str
    hora: int
    tipo_crime: Optional[str] = None

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

