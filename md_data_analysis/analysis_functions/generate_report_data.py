import pandas as pd

def generate_report_data(df: pd.DataFrame):
    """
    Agrega os principais achados, métricas e limitações para o relatório.
    """
    if df.empty:
        return {"error": "Dataset vazio, não é possível gerar o relatório."}

    # 1. Achados Principais
    top_5_bairros = df['bairro'].value_counts().head(5).reset_index()
    top_5_bairros.columns = ['name', 'value']

    top_5_crimes = df['tipo_crime'].value_counts().head(5).reset_index()
    top_5_crimes.columns = ['name', 'value']

    seasonality = df.groupby(df['data_ocorrencia'].dt.month)['id_ocorrencia'].count().reset_index()
    seasonality.columns = ['mes', 'ocorrencias']
    seasonality['mes'] = seasonality['mes'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%b'))

    # 2. Métricas dos Modelos (Resumo)
    metrics = {
        "supervised_model": {
            "name": "Previsão de Violência (LightGBM)",
            "accuracy": "Aproximadamente 85% (teste)",
            "insight": "O modelo consegue diferenciar crimes violentos de não violentos com alta precisão, sendo muito mais útil que a previsão de 10 classes."
        },
        "unsupervised_models": {
            "name": "Análise de Hotspots (DBSCAN) e Anomalias (Isolation Forest)",
            "metric": "Qualitativa",
            "insight": "Esses modelos não possuem acurácia, mas são eficazes em encontrar padrões (clusters) e pontos fora da curva (anomalias), guiando a alocação de recursos e a investigação."
        }
    }

    # 3. Limitações
    limitations = [
        "A qualidade dos dados de entrada (ex: descrições de M.O.) impacta diretamente a performance.",
        "O modelo não prevê o futuro, mas identifica padrões com base em dados passados.",
        "A ausência de dados de certas áreas ou períodos pode criar vieses na análise.",
        "Fatores externos não presentes no dataset (eventos, operações policiais) não são considerados."
    ]

    return {
        "findings": {
            "top_bairros": top_5_bairros.to_dict('records'),
            "top_crimes": top_5_crimes.to_dict('records'),
            "seasonality": seasonality.to_dict('records')
        },
        "metrics": metrics,
        "limitations": limitations
    }