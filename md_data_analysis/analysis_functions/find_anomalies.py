# md_data_analysis/analysis_functions/find_anomalies.py
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder

def find_anomalous_crimes(df: pd.DataFrame, n_results: int = 20):
    """
    Usa o Isolation Forest para encontrar as ocorrências mais anômalas no dataset.
    Foca em features que podem indicar severidade.
    """
    if df.empty:
        return {"message": "DataFrame vazio, não foi possível analisar anomalias.", "anomalies": []}

    # Seleciona features relevantes para identificar anomalias
    # Damos mais peso a características como arma utilizada e quantidade de vítimas
    features = [
        'bairro', 'arma_utilizada', 'quantidade_vitimas', 
        'quantidade_suspeitos', 'hora', 'dia_semana'
    ]
    df_analysis = df[features].copy().dropna()

    if len(df_analysis) == 0:
        return {"message": "Não há dados suficientes após limpeza para análise de anomalias.", "anomalies": []}

    # Prepara os dados para o modelo (One-Hot Encoding para categóricos)
    df_encoded = pd.get_dummies(df_analysis)

    # Cria e treina o modelo Isolation Forest
    # O parâmetro 'contamination' é uma estimativa da proporção de anomalias no dado.
    # Um valor baixo como 0.01 significa que esperamos que 1% dos dados sejam anômalos.
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(df_encoded)

    # Adiciona o score de anomalia e a predição ao DataFrame original
    # Scores mais baixos são mais anômalos
    df['anomaly_score'] = model.decision_function(pd.get_dummies(df[features].fillna('N/A')))
    df['is_anomaly'] = model.predict(pd.get_dummies(df[features].fillna('N/A')))

    # Filtra e ordena para encontrar as ocorrências mais anômalas
    anomalies = df[df['is_anomaly'] == -1].sort_values(by='anomaly_score').head(n_results)
    
    # Formata o resultado para a API
    cols_to_return = [
        'id_ocorrencia', 'tipo_crime', 'bairro', 'data_ocorrencia', 
        'arma_utilizada', 'quantidade_vitimas', 'anomaly_score'
    ]
    result = anomalies[cols_to_return].to_dict(orient='records')

    return {
        "message": f"{len(result)} anomalias principais encontradas.",
        "anomalies": result
    }