from fastapi import FastAPI, Query
from pydantic import BaseModel
from md_data_analysis.analyzer import DataAnalyzer
from md_model.predictor import CrimePredictor
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"
MODEL_PIPELINE_PATH = BASE_DIR / "artifacts" / "lgbm_model.joblib"

app = FastAPI(
    title="Delegacia 5.0 - API Preditiva de Crimes",
    description="API para análise e predição de ocorrências criminais.",
    version="1.0.0"
)

origins = ["http://localhost:5173"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = DataAnalyzer(file_path=DATA_PATH)
predictor = CrimePredictor(model_pipeline_path=MODEL_PIPELINE_PATH)

class OcorrenciaInput(BaseModel):
    bairro: str
    descricao_modus_operandi: str
    arma_utilizada: str
    sexo_suspeito: str
    orgao_responsavel: str
    status_investigacao: str
    quantidade_vitimas: int
    quantidade_suspeitos: int
    idade_suspeito: int
    latitude: float
    longitude: float
    ano: int
    mes: int
    dia_semana: int
    hora: int

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API Delegacia 5.0. Acesse /docs para a documentação."}

@app.post("/predict")
def predict_crime(ocorrencia: OcorrenciaInput):
    return predictor.predict(ocorrencia.dict())

@app.get("/occurrences")
def get_occurrences(
    tipo_crime: str = Query(None, description="Filtra ocorrências por um tipo de crime específico"),
    bairro: str = Query(None, description="Filtra ocorrências por parte do nome do bairro")
):
    return analyzer.get_all_occurrences(tipo_crime=tipo_crime, bairro=bairro)

@app.get("/statistics/top-bairros")
def get_top_bairros(limit: int = 10):
    return analyzer.get_top_bairros(limit)

@app.get("/statistics/crime-heatmap-data")
def get_crime_heatmap_data(
    bairro: str = Query(None, description="Filtra por parte do nome do bairro"),
    hora: int = Query(None, description="Filtra por uma hora específica", ge=0, le=23),
    tipo_crime: str = Query(None, description="Filtra por um tipo de crime específico"),
    dia_semana: int = Query(None, description="Filtra por um dia da semana (0=Seg, 6=Dom)", ge=0, le=6),
    ano: int = Query(None, description="Filtra por um ano específico"),
    mes: int = Query(None, description="Filtra por um mês específico", ge=1, le=12)
):
    return analyzer.get_heatmap_data(
        bairro=bairro,
        hora=hora,
        tipo_crime=tipo_crime,
        dia_semana=dia_semana,
        ano=ano,
        mes=mes
    )

@app.get("/statistics/seasonality")
def get_seasonality_data(by: str = 'month'):
    return analyzer.get_seasonality_data(by)

@app.get("/statistics/unique-crime-types")
def get_unique_crime_types():
    return analyzer.get_unique_crime_types()

@app.get("/statistics/unique-bairros")
def get_unique_bairros():
    return analyzer.get_unique_bairros()
@app.get("/statistics/unique-years")
def get_unique_years():
    return analyzer.get_unique_years()