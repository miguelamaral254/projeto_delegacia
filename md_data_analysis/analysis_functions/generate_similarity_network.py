import pandas as pd
from typing import Optional

def generate_similarity_network(df: pd.DataFrame, bairro: str, tipo_crime: Optional[str] = None, max_nodes: int = 100):
    df_filtrado = df.copy()

    df_filtrado = df_filtrado[df_filtrado['bairro'].str.contains(bairro, case=False, na=False)]
    if tipo_crime:
        df_filtrado = df_filtrado[df_filtrado['tipo_crime'].str.contains(tipo_crime, case=False, na=False)]

    df_analise = df_filtrado.sort_values(by='data_ocorrencia', ascending=False).head(max_nodes)

    if 'id_ocorrencia' not in df_analise.columns or len(df_analise) < 2:
        return {"message": "Dados insuficientes para gerar a rede.", "nodes": [], "edges": []}

    ocorrencias_dict = df_analise.to_dict('records')
    id_to_data_map = {row['id_ocorrencia']: row for row in ocorrencias_dict}

    nodes = []
    for row in ocorrencias_dict:
        row_clean = {k: v if pd.notna(v) else None for k, v in row.items()}
        nodes.append({
            "id": row_clean['id_ocorrencia'],
            "label": f"{row_clean.get('tipo_crime', 'N/A')} #{row_clean.get('id_ocorrencia', '')}",
            "group": row_clean.get('tipo_crime', 'N/A'),
            **row_clean
        })

    edges = []
    ocorrencias_ids = list(id_to_data_map.keys())
    for i in range(len(ocorrencias_ids)):
        for j in range(i + 1, len(ocorrencias_ids)):
            id1, id2 = ocorrencias_ids[i], ocorrencias_ids[j]
            ocorrencia1, ocorrencia2 = id_to_data_map[id1], id_to_data_map[id2]
            
            score = 0
            if abs(ocorrencia1.get('hora', -99) - ocorrencia2.get('hora', -99)) <= 2:
                score += 1
            if pd.notna(ocorrencia1.get('arma_utilizada')) and ocorrencia1.get('arma_utilizada') == ocorrencia2.get('arma_utilizada'):
                score += 1
            if ocorrencia1.get('dia_semana') == ocorrencia2.get('dia_semana'):
                score += 1

            if score >= 3:
                edges.append({"from": id1, "to": id2, "score": score})

    return {
        "message": f"Rede gerada com {len(nodes)} ocorrências e {len(edges)} conexões.",
        "nodes": nodes,
        "edges": edges
    }