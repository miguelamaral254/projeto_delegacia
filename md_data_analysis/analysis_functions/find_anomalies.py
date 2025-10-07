import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

def find_anomalous_crimes(df: pd.DataFrame, n_results: int = 20):
    if df.empty:
        return {"message": "DataFrame vazio, não foi possível analisar anomalias.", "anomalies": []}

    features = [
        'bairro', 'arma_utilizada', 'quantidade_vitimas', 
        'quantidade_suspeitos', 'hora', 'dia_semana'
    ]
    
    df_analysis = df.copy()
    for col in features:
        if col not in df_analysis.columns:
            df_analysis[col] = None

    df_analysis_features = df_analysis[features].fillna('N/A')

    if len(df_analysis_features) == 0:
        return {"message": "Não há dados suficientes para análise de anomalias.", "anomalies": []}

    df_encoded = pd.get_dummies(df_analysis_features)
    train_cols = df_encoded.columns
    
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_encoded)
    
    df_predict_features = df[features].fillna('N/A')
    df_predict_encoded = pd.get_dummies(df_predict_features)
    
    df_predict_aligned = df_predict_encoded.reindex(columns=train_cols, fill_value=0)

    df['anomaly_score'] = model.decision_function(df_predict_aligned)
    df['is_anomaly'] = model.predict(df_predict_aligned)

    anomalies = df[df['is_anomaly'] == -1].sort_values(by='anomaly_score').head(n_results)
    
    result = anomalies.where(pd.notna(anomalies), None).to_dict(orient='records')

    return {
        "message": f"{len(result)} anomalias principais encontradas.",
        "anomalies": result
    }