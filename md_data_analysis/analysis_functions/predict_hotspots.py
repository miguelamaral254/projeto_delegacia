import pandas as pd
from sklearn.cluster import KMeans

def cluster_hotspots(df: pd.DataFrame, bairro: str, hora: int, n_clusters: int = 3):
    df_filtrado = df[(df['bairro'] == bairro) & (df['hora'] == hora)].dropna(subset=['latitude', 'longitude'])
    if len(df_filtrado) < n_clusters:
        return {"message": "Dados insuficientes para prever hotspots.", "hotspots": []}
    
    coordenadas = df_filtrado[['latitude', 'longitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    kmeans.fit(coordenadas)
    
    hotspots = [{"lat": lat, "lon": lon} for lat, lon in kmeans.cluster_centers_]
    return {"message": f"{len(hotspots)} hotspots previstos encontrados.", "hotspots": hotspots}