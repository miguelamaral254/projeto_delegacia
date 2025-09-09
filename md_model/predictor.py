# md_model/predictor.py

import pandas as pd
from joblib import load
from md_core.config import MODEL_PATH, PREPROCESSOR_PATH


# from md_api.schemas import OcorrenciaInput <-- REMOVA ESTA LINHA

class ModelPredictor:
    def __init__(self):
        try:
            print("Carregando artefatos do modelo...")
            self.model = load(MODEL_PATH)
            self.preprocessor = load(PREPROCESSOR_PATH)
            print("Artefatos carregados com sucesso.")
        except FileNotFoundError:
            print("Erro: Arquivos de modelo não encontrados. Execute 'python train.py' primeiro.")
            self.model = None
            self.preprocessor = None

    # ALTERADO: Agora aceita um dicionário (dict) em vez de um objeto Pydantic
    def predict(self, input_data: dict) -> str:
        if not self.model or not self.preprocessor:
            return "Modelo não disponível."

        # ALTERADO: A conversão agora é direto de um dict para DataFrame
        df = pd.DataFrame([input_data])

        # Aplica a mesma transformação dos dados de treino
        transformed_data = self.preprocessor.transform(df)

        # Faz a predição
        prediction = self.model.predict(transformed_data)

        return prediction[0]


# A instância global continua a mesma
predictor = ModelPredictor()