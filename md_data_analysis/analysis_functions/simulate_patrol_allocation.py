import pandas as pd
import numpy as np
from typing import Optional
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def simulate_patrol_allocation(df: pd.DataFrame, bairro: str, num_patrols: int, grid_size: int = 5):
    np.random.seed(42)

    df_bairro = df[df['bairro'].str.contains(bairro, case=False, na=False)].dropna(subset=['latitude', 'longitude'])

    if len(df_bairro) < 10:
        return {"error": "Dados insuficientes para o bairro selecionado."}

    lat_min, lat_max = df_bairro['latitude'].min(), df_bairro['latitude'].max()
    lon_min, lon_max = df_bairro['longitude'].min(), df_bairro['longitude'].max()

    lat_step = (lat_max - lat_min) / grid_size
    lon_step = (lon_max - lon_min) / grid_size

    grid = []
    risk_scores = np.zeros(grid_size * grid_size)
    total_risk = len(df_bairro)

    for i in range(grid_size):
        for j in range(grid_size):
            cell_lat_min = lat_min + i * lat_step
            cell_lat_max = lat_min + (i + 1) * lat_step
            cell_lon_min = lon_min + j * lon_step
            cell_lon_max = lon_min + (j + 1) * lon_step
            cell_crimes = df_bairro[
                (df_bairro['latitude'] >= cell_lat_min) & (df_bairro['latitude'] < cell_lat_max) &
                (df_bairro['longitude'] >= cell_lon_min) & (df_bairro['longitude'] < cell_lon_max)
            ]
            
            cell_index = i * grid_size + j
            risk_scores[cell_index] = len(cell_crimes)
            grid.append({
                "id": cell_index, "bounds": [[cell_lat_min, cell_lon_min], [cell_lat_max, cell_lon_max]],
                "center": [cell_lat_min + lat_step/2, cell_lon_min + lon_step/2], "risk": len(cell_crimes)
            })
    
    num_cells = grid_size * grid_size
    
    random_patrols_indices = np.random.choice(num_cells, num_patrols, replace=False)
    random_risk_covered = np.sum(risk_scores[random_patrols_indices])
    
    heuristic_patrols_indices = np.argsort(risk_scores)[-num_patrols:][::-1]
    heuristic_risk_covered = np.sum(risk_scores[heuristic_patrols_indices])

    q_table = np.zeros((num_cells, num_cells))
    alpha, gamma, epsilon = 0.1, 0.6, 0.1
    for _ in range(1000):
        for state in range(num_cells):
            action = np.argmax(q_table[state]) if np.random.uniform(0, 1) > epsilon else np.random.choice(num_cells)
            reward = risk_scores[action]
            q_table[state, action] += alpha * (reward + gamma * np.max(q_table[action]) - q_table[state, action])

    rl_patrols_indices = np.argsort(np.max(q_table, axis=1))[-num_patrols:][::-1]
    rl_risk_covered = np.sum(risk_scores[rl_patrols_indices])
    
    # --- LÓGICA HÍBRIDA CORRIGIDA PARA O DBSCAN ---
    coords = df_bairro[['latitude', 'longitude']]
    coords_scaled = StandardScaler().fit_transform(coords)
    db = DBSCAN(eps=0.3, min_samples=3).fit(coords_scaled)
    
    hotspot_centroids = []
    for k in set(db.labels_):
        if k != -1:
            cluster_points = coords[db.labels_ == k]
            hotspot_centroids.append(cluster_points.mean().values)

    hotspot_cell_indices = set()
    for lat, lon in hotspot_centroids:
        cell_i = int((lat - lat_min) / lat_step) if lat_step > 0 else 0
        cell_j = int((lon - lon_min) / lon_step) if lon_step > 0 else 0
        cell_i = min(grid_size - 1, max(0, cell_i))
        cell_j = min(grid_size - 1, max(0, cell_j))
        hotspot_cell_indices.add(cell_i * grid_size + cell_j)
    
    dbscan_primary_indices = sorted(list(hotspot_cell_indices), key=lambda i: risk_scores[i], reverse=True)
    dbscan_patrols_indices = dbscan_primary_indices[:num_patrols]
    
    remaining_patrols = num_patrols - len(dbscan_patrols_indices)
    if remaining_patrols > 0:
        all_risky_indices = np.argsort(risk_scores)[::-1]
        fallback_indices = [idx for idx in all_risky_indices if idx not in dbscan_patrols_indices]
        dbscan_patrols_indices.extend(fallback_indices[:remaining_patrols])

    dbscan_risk_covered = np.sum(risk_scores[dbscan_patrols_indices])

    def get_patrol_locations(indices):
        return [grid[i]['center'] for i in indices]

    return {
        "grid": grid, "total_risk": total_risk,
        "policies": {
            "random": {"patrol_locations": get_patrol_locations(random_patrols_indices), "risk_covered": random_risk_covered, "risk_reduction_percent": (random_risk_covered / total_risk) * 100 if total_risk > 0 else 0},
            "heuristic": {"patrol_locations": get_patrol_locations(heuristic_patrols_indices), "risk_covered": heuristic_risk_covered, "risk_reduction_percent": (heuristic_risk_covered / total_risk) * 100 if total_risk > 0 else 0},
            "heuristic_dbscan": {"patrol_locations": get_patrol_locations(dbscan_patrols_indices), "risk_covered": dbscan_risk_covered, "risk_reduction_percent": (dbscan_risk_covered / total_risk) * 100 if total_risk > 0 else 0},
            "rl_q_learning": {"patrol_locations": get_patrol_locations(rl_patrols_indices), "risk_covered": rl_risk_covered, "risk_reduction_percent": (rl_risk_covered / total_risk) * 100 if total_risk > 0 else 0}
        }
    }
