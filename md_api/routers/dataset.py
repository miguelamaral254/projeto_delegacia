from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import shutil

from md_data_analysis.data_loader import reload_dataframe

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "dataset_ocorrencias_delegacia_5.csv"

router = APIRouter(prefix="/dataset", tags=["Dataset Management"])

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Formato de arquivo inválido. Por favor, envie um arquivo .csv.")

    try:
        with open(DATA_PATH, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao salvar o arquivo: {e}")
    finally:
        file.file.close()

    reload_dataframe(DATA_PATH)

    return {"message": f"Dataset '{file.filename}' carregado com sucesso e análises atualizadas."}