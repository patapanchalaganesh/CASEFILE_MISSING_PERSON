"""
clustering.py
Movement pattern clustering for the CASEFILE missing person project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.neighbors import BallTree
import joblib
import os
from math import radians

# Ethical disclaimer: All case data is FICTIONAL - this is an academic simulation.

def dbscan_stay_points(stay_points_df, eps_km=0.15, min_samples=3):
    """
    Apply DBSCAN on stay point coordinates (lat, lon).
    eps_km is converted to approximate radians.
    """
    if stay_points_df.empty:
        return stay_points_df
    
    # Normalize column names
    stay_points_df = stay_points_df.copy()
    if 'stay_lat' in stay_points_df.columns and 'lat' not in stay_points_df.columns:
        stay_points_df['lat'] = stay_points_df['stay_lat']
        stay_points_df['lon'] = stay_points_df['stay_lon']
    
    # Earth radius in km
    R = 6371.0
    eps_rad = eps_km / R
    
    coords = np.radians(stay_points_df[['lat', 'lon']].values)
    
    dbscan = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm='ball_tree', metric='haversine')
    labels = dbscan.fit_predict(coords)
    
    stay_points_df = stay_points_df.copy()
    stay_points_df['cluster_label'] = labels
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_points = list(labels).count(-1)
    
    print(f"DBSCAN: Found {num_clusters} clusters. Noise points: {noise_points}")
    if num_clusters > 0:
        cluster_sizes = stay_points_df[stay_points_df['cluster_label'] != -1]['cluster_label'].value_counts()
        print(f"Cluster sizes:\n{cluster_sizes}")
        
    return stay_points_df

def kmeans_areas(cluster_centers_df, n_clusters=None):
    """
    Apply K-Means on DBSCAN cluster centers to group into broader areas.
    """
    if cluster_centers_df.empty:
        return cluster_centers_df, None
        
    coords = cluster_centers_df[['lat', 'lon']].values
    
    if n_clusters is None:
        # Elbow method
        max_k = min(25, len(coords) - 1)
        if max_k < 3:
            n_clusters = max(1, len(coords) // 2)
            print(f"Not enough centers for elbow method. Defaulting to k={n_clusters}")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(coords)
        else:
            k_range = range(3, max_k + 1)
            inertias = []
            silhouettes = []
            best_k = 3
            best_score = -1
            
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                labels = kmeans.fit_predict(coords)
                inertias.append(kmeans.inertia_)
                if len(set(labels)) > 1:
                    score = silhouette_score(coords, labels)
                    silhouettes.append(score)
                    if score > best_score:
                        best_score = score
                        best_k = k
                else:
                    silhouettes.append(-1)
                    
            print(f"Elbow method selected k={best_k} (Silhouette: {best_score:.3f})")
            n_clusters = best_k
            plot_elbow(list(k_range), inertias, silhouettes)
            
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(coords)
    
    cluster_centers_df = cluster_centers_df.copy()
    cluster_centers_df['area_id'] = labels
    
    return cluster_centers_df, kmeans

def label_areas_with_pois(area_centers_df, pois_path='data/raw/beijing_pois.csv'):
    """
    For each area center, find nearest POI from the POI dataset.
    """
    if area_centers_df.empty:
        return area_centers_df
        
    area_centers_df = area_centers_df.copy()
    area_centers_df['name'] = 'Unknown Area'
    area_centers_df['category'] = 'Unknown'
    
    if not os.path.exists(pois_path):
        print(f"Warning: POI file {pois_path} not found. Using generic names.")
        area_centers_df['name'] = [f'Area {i}' for i in area_centers_df.index]
        return area_centers_df
        
    try:
        pois_df = pd.read_csv(pois_path)
        # Normalize POI column names
        if 'latitude' in pois_df.columns:
            pois_df = pois_df.rename(columns={'latitude': 'lat', 'longitude': 'lon'})
        if pois_df.empty or 'lat' not in pois_df.columns or 'lon' not in pois_df.columns:
            print("POI file format invalid or empty.")
            return area_centers_df
            
        # Create BallTree for fast nearest neighbor search
        poi_coords = np.radians(pois_df[['lat', 'lon']].values)
        tree = BallTree(poi_coords, metric='haversine')
        
        area_coords = np.radians(area_centers_df[['lat', 'lon']].values)
        distances, indices = tree.query(area_coords, k=1)
        
        for i, idx in enumerate(indices):
            poi_idx = idx[0]
            poi_name = pois_df.iloc[poi_idx].get('name', 'Unknown')
            poi_cat = pois_df.iloc[poi_idx].get('category', 'Unknown')
            
            area_centers_df.loc[area_centers_df.index[i], 'name'] = f"Area near {poi_name}"
            area_centers_df.loc[area_centers_df.index[i], 'category'] = poi_cat
            
    except Exception as e:
        print(f"Error labeling with POIs: {e}")
        
    return area_centers_df

def evaluate_clustering(X, labels):
    """
    Compute clustering evaluation metrics.
    """
    metrics = {}
    if len(set(labels)) > 1:
        metrics['silhouette'] = silhouette_score(X, labels)
        metrics['calinski_harabasz'] = calinski_harabasz_score(X, labels)
        metrics['davies_bouldin'] = davies_bouldin_score(X, labels)
    else:
        metrics['silhouette'] = -1
        metrics['calinski_harabasz'] = -1
        metrics['davies_bouldin'] = -1
    return metrics

def plot_elbow(k_range, inertias, silhouettes):
    """
    Create side-by-side elbow and silhouette plots.
    """
    os.makedirs('reports', exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(k_range, inertias, marker='o')
    ax1.set_title('Elbow Method (Inertia)')
    ax1.set_xlabel('Number of clusters (k)')
    ax1.set_ylabel('Inertia')
    
    ax2.plot(k_range, silhouettes, marker='o', color='green')
    ax2.set_title('Silhouette Scores')
    ax2.set_xlabel('Number of clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    
    plt.tight_layout()
    plt.savefig('reports/clustering_elbow.png')
    plt.close()

def plot_clusters(stay_points_df, cluster_centers_df):
    """
    Scatter plot of stay points colored by cluster.
    """
    os.makedirs('reports', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    
    # Plot noise points
    noise = stay_points_df[stay_points_df['cluster_label'] == -1]
    plt.scatter(noise['lon'], noise['lat'], c='gray', alpha=0.5, s=10, label='Noise')
    
    # Plot clusters
    clusters = stay_points_df[stay_points_df['cluster_label'] != -1]
    plt.scatter(clusters['lon'], clusters['lat'], c=clusters['cluster_label'], cmap='tab20', s=20, label='Clusters')
    
    # Plot centers
    if not cluster_centers_df.empty:
        plt.scatter(cluster_centers_df['lon'], cluster_centers_df['lat'], 
                    c='black', marker='X', s=100, label='Centers')
                    
    plt.title('DBSCAN Clustering of Stay Points')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('reports/clustering_map.png')
    plt.close()

def assign_areas_to_gps(gps_features_df, kmeans_model, area_centers_df):
    """
    For each GPS point, assign to nearest area using the KMeans model.
    """
    if gps_features_df.empty or kmeans_model is None or area_centers_df.empty:
        return gps_features_df
        
    # Handle different column naming conventions
    if 'latitude' in gps_features_df.columns and 'lat' not in gps_features_df.columns:
        lat_col, lon_col = 'latitude', 'longitude'
    else:
        lat_col, lon_col = 'lat', 'lon'
    gps_coords = gps_features_df[[lat_col, lon_col]].values
    area_ids = kmeans_model.predict(gps_coords)
    
    gps_features_df = gps_features_df.copy()
    gps_features_df['area_id'] = area_ids
    
    # Map area_id to area_name
    area_map = dict(zip(area_centers_df['area_id'], area_centers_df['name']))
    gps_features_df['area_name'] = gps_features_df['area_id'].map(area_map)
    
    os.makedirs('data/processed', exist_ok=True)
    gps_features_df.to_csv('data/processed/gps_with_areas.csv', index=False)
    
    return gps_features_df

def clustering_pipeline():
    """
    Run full pipeline: load data, DBSCAN, K-Means, label, evaluate, save.
    """
    print("Starting clustering pipeline...")
    
    # Check if files exist
    stay_points_path = 'data/processed/stay_points.csv'
    gps_features_path = 'data/processed/gps_features.csv'
    
    if not os.path.exists(stay_points_path) or not os.path.exists(gps_features_path):
        print(f"Error: Required data files not found.")
        print(f"Need {stay_points_path} and {gps_features_path}")
        # Creating dummy data for testing purposes
        os.makedirs('data/processed', exist_ok=True)
        print("Generating dummy data for testing...")
        stay_points_df = pd.DataFrame({
            'lat': np.random.uniform(39.4, 40.4, 100),
            'lon': np.random.uniform(115.5, 117.5, 100)
        })
        gps_features_df = pd.DataFrame({
            'lat': np.random.uniform(39.4, 40.4, 500),
            'lon': np.random.uniform(115.5, 117.5, 500)
        })
    else:
        stay_points_df = pd.read_csv(stay_points_path)
        gps_features_df = pd.read_csv(gps_features_path)
        
    # 1. DBSCAN on stay points
    clustered_sp_df = dbscan_stay_points(stay_points_df, eps_km=0.15, min_samples=3)
    
    # Compute cluster centers
    valid_clusters = clustered_sp_df[clustered_sp_df['cluster_label'] != -1]
    if not valid_clusters.empty:
        cluster_centers = valid_clusters.groupby('cluster_label')[['lat', 'lon']].mean().reset_index()
    else:
        cluster_centers = pd.DataFrame(columns=['cluster_label', 'lat', 'lon'])
        
    # 2. K-Means on cluster centers
    print(f"Running K-Means on {len(cluster_centers)} cluster centers...")
    areas_df, kmeans_model = kmeans_areas(cluster_centers)
    
    if areas_df is not None and not areas_df.empty:
        # Compute area centers
        area_centers_df = areas_df.groupby('area_id')[['lat', 'lon']].mean().reset_index()
        
        # 3. Label areas with POIs
        area_centers_df = label_areas_with_pois(area_centers_df)
        
        # 4. Evaluate K-Means clustering
        if kmeans_model is not None and len(cluster_centers) > 3:
            metrics = evaluate_clustering(cluster_centers[['lat', 'lon']].values, areas_df['area_id'].values)
            print("Clustering Metrics:", metrics)
            
        # 6. Plot clusters
        plot_clusters(clustered_sp_df, cluster_centers)
        
        # 7. Assign areas to GPS
        gps_with_areas = assign_areas_to_gps(gps_features_df, kmeans_model, area_centers_df)
        
        # 8. Save models and outputs
        os.makedirs('models', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)
        
        model_data = {
            'dbscan_params': {'eps_km': 0.15, 'min_samples': 3},
            'kmeans_model': kmeans_model
        }
        joblib.dump(model_data, 'models/clustering_model.pkl')
        
        area_centers_df.to_csv('data/processed/area_centers.csv', index=False)
        clustered_sp_df.to_csv('data/processed/stay_points_clustered.csv', index=False)
        
        print(f"Summary: Created {len(cluster_centers)} DBSCAN clusters and {len(area_centers_df)} K-Means areas.")
        print(f"Area Centers:\n{area_centers_df}")
    else:
        print("Not enough clusters found to create areas.")
        
    print("Clustering pipeline completed.")

if __name__ == "__main__":
    clustering_pipeline()
