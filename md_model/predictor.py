import joblib
import pandas as pd


class CrimePredictor:
    def __init__(self, model_pipeline_path: str):
        self.pipeline = joblib.load(model_pipeline_path)

    def predict(self, input_data: dict):
        input_df = pd.DataFrame([input_data])

        prediction = self.pipeline.predict(input_df)
        prediction_proba = self.pipeline.predict_proba(input_df)

        classes = self.pipeline.named_steps['classifier'].classes_
        probabilities = dict(zip(classes, prediction_proba[0]))

        return {
            "tipo_crime_predito": prediction[0],
            "probabilidades": probabilities
        }