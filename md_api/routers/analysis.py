from fastapi import APIRouter, Depends, Query
import pandas as pd
from pathlib import Path
import sys
from typing import Optional
from md_data_analysis.data_loader import load_dataframe
from md_data_analysis.analysis_functions.generate_similarity_network import generate_similarity_network
from md_data_analysis.analysis_functions.simulate_patrol_allocation import simulate_patrol_allocation 

# Configuração de Paths
FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

router = APIRouter(prefix="/analysis", tags=["Complex Analysis"])

def get_df():
    return load_dataframe(Path(ROOT) / "data" / "dataset_ocorrencias_delegacia_5.csv")

@router.get("/similarity-network")
def read_similarity_network(
    bairro: str, 
    tipo_crime: str = Query(None),
    max_nodes: int = Query(100, gt=10, le=200),
    df: pd.DataFrame = Depends(get_df)
):
    return generate_similarity_network(df, bairro=bairro, tipo_crime=tipo_crime, max_nodes=max_nodes)



@router.get("/patrol-simulation")
def run_patrol_simulation(
    bairro: str,
    num_patrols: int = Query(3, gt=0, le=10),
    df: pd.DataFrame = Depends(get_df)
):
    return simulate_patrol_allocation(df, bairro=bairro, num_patrols=num_patrols)