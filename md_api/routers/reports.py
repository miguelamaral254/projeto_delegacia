from fastapi import APIRouter, Depends, HTTPException
import pandas as pd
from pathlib import Path
import sys
import json 


FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from md_data_analysis.data_loader import get_dataframe
from md_data_analysis.analysis_functions.generate_report_data import generate_report_data

router = APIRouter(prefix="/reports", tags=["Reports"])
REPORTS_DIR = Path(__file__).resolve().parent.parent.parent / "reports"
SUMMARY_REPORT_PATH = REPORTS_DIR / "models_summary.json"


def get_df():
    return get_dataframe()

@router.get("/summary")
def get_summary_report_data(df: pd.DataFrame = Depends(get_df)):
    return generate_report_data(df)

@router.get("/models-summary")
def get_models_summary():
    try:
        with open(SUMMARY_REPORT_PATH, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        return [] 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler resumo dos modelos: {str(e)}")


@router.get("/model/{report_name}")
def get_model_report(report_name: str):
    report_path = REPORTS_DIR / report_name
    
    if not report_path.name.endswith(".json") or ".." in report_path.parts:
        raise HTTPException(status_code=400, detail="Nome de relatório inválido.")

    try:
        with open(report_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Arquivo de relatório '{report_name}' não encontrado.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao carregar o relatório: {str(e)}")