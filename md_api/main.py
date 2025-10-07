import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from md_api.routers import statistics, predictions, occurrences, analysis

FILE = Path(__file__).resolve()
ROOT = FILE.parent.parent  
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

app = FastAPI(
    title="Delegacia 5.0 - API Preditiva de Crimes",
    description="API para análise e predição de ocorrências criminais.",
    version="1.0.0"
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
@app.get("/")
def read_root():
    return {"message": "Bem-vindo à API Delegacia 5.0. Acesse /docs para a documentação interativa."}

