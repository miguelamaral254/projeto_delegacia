import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles 

from md_api.routers import statistics, predictions, occurrences, analysis, dataset, reports 
from md_data_analysis.data_loader import load_initial_dataframe

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset_ocorrencias_delegacia_5.csv"

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_initial_dataframe(DATA_PATH)
    yield

app = FastAPI(
    title="Delegacia 5.0 - API Preditiva de Crimes",
    description="API para análise e predição de ocorrências criminais.",
    version="1.0.0",
    lifespan=lifespan
)

origins = ["http://localhost:5173", "http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(statistics.router)
app.include_router(predictions.router)
app.include_router(occurrences.router)
app.include_router(analysis.router)
app.include_router(dataset.router)
app.include_router(reports.router) 
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API Delegacia 5.0. Acesse /docs para a documentação interativa."}