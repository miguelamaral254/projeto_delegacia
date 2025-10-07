import sys
from pathlib import Path
from fastapi import APIRouter, Depends, Query
import pandas as pd

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# MUDANÇA 1: Importa a função correta
from md_data_analysis.data_loader import get_dataframe
from md_data_analysis.analysis_functions.get_all_occurrences import analyze_all_occurrences

router = APIRouter(tags=["Occurrences"])

# MUDANÇA 2: Usa a nova função para obter o dataframe da memória
def get_df():
    return get_dataframe()

@router.get("/occurrences")
def read_occurrences(
    tipo_crime: str = Query(None), bairro: str = Query(None), df: pd.DataFrame = Depends(get_df)
):
    return analyze_all_occurrences(df, tipo_crime=tipo_crime, bairro=bairro)
