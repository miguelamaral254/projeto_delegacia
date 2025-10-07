from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel
import pandas as pd
from typing import List

from md_data_analysis.data_loader import get_dataframe
from md_data_analysis.analysis_functions.get_top_bairros import analyze_top_bairros
from md_data_analysis.analysis_functions.get_heatmap_data import analyze_heatmap_data
from md_data_analysis.analysis_functions.get_seasonality_data import analyze_seasonality_data
from md_data_analysis.analysis_functions.get_unique_crime_types import analyze_unique_crime_types
from md_data_analysis.analysis_functions.get_unique_bairros import analyze_unique_bairros
from md_data_analysis.analysis_functions.get_unique_years import analyze_unique_years
from md_data_analysis.analysis_functions.analyze_text_topics import analyze_text_topics
from md_data_analysis.analysis_functions.find_topic_examples import find_topic_examples

router = APIRouter(prefix="/statistics", tags=["Statistics"])

def get_df():
    return get_dataframe()

class TopicKeywords(BaseModel):
    keywords: List[str]
    bairro: str = None

@router.get("/top-bairros")
def read_top_bairros(limit: int = 10, df: pd.DataFrame = Depends(get_df)):
    return analyze_top_bairros(df, limit)

@router.get("/crime-heatmap-data")
def read_crime_heatmap_data(bairro: str = Query(None), hora: int = Query(None), tipo_crime: str = Query(None), dia_semana: int = Query(None), ano: int = Query(None), mes: int = Query(None), df: pd.DataFrame = Depends(get_df)):
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
def read_modus_operandi_topics(bairro: str = Query(None), n_topics: int = Query(5, gt=1, le=15), df: pd.DataFrame = Depends(get_df)):
    return analyze_text_topics(df, bairro=bairro, n_topics=n_topics)

@router.post("/modus-operandi-examples")
def read_modus_operandi_examples(data: TopicKeywords, df: pd.DataFrame = Depends(get_df)):
    return find_topic_examples(df, keywords=data.keywords, bairro=data.bairro)