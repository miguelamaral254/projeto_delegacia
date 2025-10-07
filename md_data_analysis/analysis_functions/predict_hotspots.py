import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Optional

def cluster_hotspots(df: pd.DataFrame, bairro: str, hora: int, tipo_crime: Optional[str] = None):
    """
    Encontra hotspots de crimes usando DBSCAN, com filtro opcional por tipo de crime.
    """
    
    df_filtrado = df[(df['bairro'].str.contains(bairro, case=False, na=False)) & (df['hora'] == hora)]

    if tipo_crime:
        df_filtrado = df_filtrado[df_filtrado['tipo_crime'] == tipo_crime]
    
    coords = df_filtrado[['latitude', 'longitude']].dropna()

    if len(coords) < 3: # Reduzido para 3 para ser mais flexível
        return {"message": "Dados insuficientes para encontrar hotspots com os filtros aplicados.", "hotspots": []}

    coords_scaled = StandardScaler().fit_transform(coords)

    # >>>>> CORREÇÃO PRINCIPAL AQUI <<<<<
    # Parâmetros ANTERIORES: eps=0.15, min_samples=4
    # Parâmetros NOVOS (mais flexíveis): Aumentamos o raio de busca (eps) e
    # diminuímos a quantidade mínima de pontos (min_samples).
    db = DBSCAN(eps=0.3, min_samples=3).fit(coords_scaled)
    
    labels = db.labels_
    unique_labels = set(labels)
    
    hotspots = []
    for k in unique_labels:
        if k == -1:
            continue
        
        cluster_points = coords[labels == k]
        
        if not cluster_points.empty:
            centroid = cluster_points.mean().to_dict()
            hotspots.append({
                "lat": centroid['latitude'],
                "lon": centroid['longitude'],
                "ocorrencias_no_hotspot": len(cluster_points)
            })

    return {
        "message": f"{len(hotspots)} hotspots encontrados com DBSCAN.",
        "hotspots": hotspots
    }

