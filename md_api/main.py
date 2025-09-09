# md_api/main.py

from fastapi import FastAPI, HTTPException
from .schemas import OcorrenciaInput, PredictionOutput
from md_model.predictor import predictor  # Esta linha agora funcionará!

app = FastAPI(
    title="Delegacia 5.0 - API Preditiva de Crimes",
    description="API para prever o tipo de crime com base em dados de ocorrência.",
    version="1.0.0"
)


# ... (o resto do arquivo, como o @app.get("/"), pode continuar igual) ...

@app.post("/predict", response_model=PredictionOutput, tags=["Prediction"])
def predict_crime(ocorrencia_data: OcorrenciaInput):
    """
    Recebe os dados de uma ocorrência e retorna o tipo de crime previsto.
    """
    if not predictor.model:
        raise HTTPException(status_code=503, detail="Modelo não está disponível. Treine o modelo primeiro.")

    try:
        # ALTERADO: Converte o objeto Pydantic para um dicionário antes de chamar o predict
        prediction = predictor.predict(ocorrencia_data.dict())
        return PredictionOutput(tipo_crime_predito=prediction)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao processar a predição: {e}")