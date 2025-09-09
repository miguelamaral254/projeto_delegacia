# md_api/schemas.py
from pydantic import BaseModel

class OcorrenciaInput(BaseModel):
    # Features que o usuário precisa enviar para a predição
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
    # Features temporais também são esperadas
    ano: int
    mes: int
    dia_semana: int
    hora: int

class PredictionOutput(BaseModel):
    tipo_crime_predito: str