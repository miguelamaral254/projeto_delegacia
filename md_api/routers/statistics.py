from fastapi import APIRouter, Depends, Query
import pandas as pd
from pathlib import Path
import sys

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from md_data_analysis.data_loader import load_dataframe
from md_data_analysis.analysis_functions.get_top_bairros import analyze_top_bairros
from md_data_analysis.analysis_functions.get_heatmap_data import analyze_heatmap_data
from md_data_analysis.analysis_functions.get_seasonality_data import analyze_seasonality_data
from md_data_analysis.analysis_functions.get_unique_crime_types import analyze_unique_crime_types
from md_data_analysis.analysis_functions.get_unique_bairros import analyze_unique_bairros
from md_data_analysis.analysis_functions.get_unique_years import analyze_unique_years
from md_data_analysis.analysis_functions.analyze_text_topics import analyze_text_topics 

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"

router = APIRouter(prefix="/statistics", tags=["Statistics"])

def get_df():
    return load_dataframe(DATA_PATH)

@router.get("/top-bairros")
def read_top_bairros(limit: int = 10, df: pd.DataFrame = Depends(get_df)):
    return analyze_top_bairros(df, limit)

@router.get("/crime-heatmap-data")
def read_crime_heatmap_data(
    bairro: str = Query(None), hora: int = Query(None), tipo_crime: str = Query(None),
    dia_semana: int = Query(None), ano: int = Query(None), mes: int = Query(None),
    df: pd.DataFrame = Depends(get_df)
):
    return analyze_heatmap_data(df, bairro, hora, tipo_crime, dia_semana, ano, mes)

@router.get("/seasonality")
def read_seasonality_data(by: str = Query('month', enum=['month', 'day_of_week']), df: pd.DataFrame = Depends(get_df)):
    return analyze_seasonality_data(df, by)

@router.get("/unique-crime-types")
def read_unique_crime_types(df: pd.DataFrame = Depends(get_df)):
    return analyze_unique_crime_types(df)

@router.get("/unique-bairros")
def read_unique_bairros(df: pd.DataFrame = Depends(get_df)):
    return analyze_unique_bairros(df)

@router.get("/unique-years")
def read_unique_years(df: pd.DataFrame = Depends(get_df)):
    return analyze_unique_years(df)

@router.get("/modus-operandi-topics")
def read_modus_operandi_topics(
    bairro: str = Query(None),
    n_topics: int = Query(5, gt=1, le=15),
    df: pd.DataFrame = Depends(get_df)
):
    return analyze_text_topics(df, bairro=bairro, n_topics=n_topics)