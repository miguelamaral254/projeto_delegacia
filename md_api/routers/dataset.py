from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import subprocess
import sys
import json

from md_data_analysis.data_loader import reload_dataframe

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = ROOT_DIR / "data" / "dataset_ocorrencias_delegacia_5.csv"
STATUS_FILE = ROOT_DIR / "reports" / "training_status.json"
TRAIN_SCRIPT_PATH = ROOT_DIR / "md_training" / "start_train.py"

router = APIRouter(prefix="/dataset", tags=["Dataset Management"])

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Formato de arquivo inválido. Por favor, envie um arquivo .csv.")

    try:
        if STATUS_FILE.exists():
            STATUS_FILE.unlink()
            
        with open(DATA_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar o arquivo: {e}")
    finally:
        file.file.close()

    reload_dataframe(DATA_PATH)
    
    try:
        print("Iniciando o pipeline de treinamento em segundo plano...")
        if not TRAIN_SCRIPT_PATH.exists():
            error_msg = f"ERRO: Script de treino não encontrado em {TRAIN_SCRIPT_PATH}"
            print(error_msg)
            return {"message": f"Dataset '{file.filename}' carregado, mas o script de retreinamento não foi encontrado."}
            
        subprocess.Popen([sys.executable, str(TRAIN_SCRIPT_PATH)])
    except Exception as e:
        error_msg = f"ERRO ao tentar iniciar o pipeline de treinamento: {e}"
        print(error_msg)
        return {"message": f"Dataset '{file.filename}' carregado, mas falha ao iniciar o retreinamento automático."}

    return {"message": f"Dataset '{file.filename}' carregado. O retreinamento dos modelos foi iniciado em segundo plano."}


@router.get("/training-status")
def get_training_status():
    if not STATUS_FILE.exists():
        return {"status": "idle", "message": "Nenhum treinamento em andamento ou iniciado."}
    
    try:
        with open(STATUS_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao ler arquivo de status: {e}")