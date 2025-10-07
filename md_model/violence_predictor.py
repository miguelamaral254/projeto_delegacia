import joblib
import pandas as pd
from pathlib import Path

class ViolencePredictor:
    def __init__(self, model_path: Path):
        self.pipeline = joblib.load(model_path)

    def predict(self, input_data: dict):
        input_df = pd.DataFrame([input_data])
        
        prediction = self.pipeline.predict(input_df)[0] 
        prediction_proba = self.pipeline.predict_proba(input_df)[0]

        probabilities = {
            "nao_violento": prediction_proba[0],
            "violento": prediction_proba[1]
        }
        
        return {
            "previsao_violencia": bool(prediction),
            "probabilidades": probabilities
        }