import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

def prepare_anomaly_features(df):
    """
    Prepare features for anomaly detection.
    """
    features = ['speed_kmh', 'distance_km', 'hour', 'bearing_change', 'time_delta_seconds']
    
    # Check if features exist in df, if not create dummy ones or raise error
    missing = [f for f in features if f not in df.columns]
    if missing:
        print(f"Warning: missing features {missing}")
        for f in missing:
            df[f] = 0.0 # dummy
            
    df_features = df[features].copy()
    
    # Remove NaN/inf values
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna()
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(df_features)
    
    return X, scaler, features, df_features.index

def isolation_forest_detect(X, contamination=0.05):
    """
    Train Isolation Forest and detect anomalies.
    """
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    predictions = model.fit_predict(X)
    scores = model.decision_function(X)
    
    anomalies_count = np.sum(predictions == -1)
    percent = (anomalies_count / len(X)) * 100
    print(f"Isolation Forest: Detected {anomalies_count} anomalies ({percent:.2f}%)")
    
    return predictions, scores, model

def local_outlier_factor_detect(X, n_neighbors=20, contamination=0.05):
    """
    Train LOF model and detect anomalies.
    """
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    predictions = model.fit_predict(X)
    scores = model.negative_outlier_factor_
    
    anomalies_count = np.sum(predictions == -1)
    percent = (anomalies_count / len(X)) * 100
    print(f"Local Outlier Factor: Detected {anomalies_count} anomalies ({percent:.2f}%)")
    
    return predictions, scores

def one_class_svm_detect(X, contamination=0.05):
    """
    Train One-Class SVM and detect anomalies. Samples data if too large.
    """
    if len(X) > 10000:
        idx = np.random.choice(len(X), 10000, replace=False)
        X_sample = X[idx]
    else:
        X_sample = X
        
    model = OneClassSVM(kernel='rbf', nu=contamination)
    model.fit(X_sample)
    
    predictions = model.predict(X)
    scores = model.decision_function(X)
    
    anomalies_count = np.sum(predictions == -1)
    percent = (anomalies_count / len(X)) * 100
    print(f"One-Class SVM: Detected {anomalies_count} anomalies ({percent:.2f}%)")
    
    return predictions, scores

def ensemble_anomaly_detection(df, X, valid_index):
    """
    Run all three methods and create ensemble.
    """
    if_preds, if_scores, if_model = isolation_forest_detect(X)
    lof_preds, lof_scores = local_outlier_factor_detect(X)
    svm_preds, svm_scores = one_class_svm_detect(X)
    
    # Convert predictions (-1 anomaly, 1 normal) to (1 anomaly, 0 normal)
    if_anom = (if_preds == -1).astype(int)
    lof_anom = (lof_preds == -1).astype(int)
    svm_anom = (svm_preds == -1).astype(int)
    
    ensemble_sum = if_anom + lof_anom + svm_anom
    ensemble_anomaly = (ensemble_sum >= 2).astype(int)
    
    # Normalize scores to 0-1 range (higher = more anomalous)
    scaler = MinMaxScaler()
    if_norm = scaler.fit_transform((-if_scores).reshape(-1, 1)).flatten()
    lof_norm = scaler.fit_transform((-lof_scores).reshape(-1, 1)).flatten()
    svm_norm = scaler.fit_transform((-svm_scores).reshape(-1, 1)).flatten()
    
    avg_score = (if_norm + lof_norm + svm_norm) / 3.0
    
    # Create copies of these columns initialized to normal
    df['if_anomaly'] = 0
    df['lof_anomaly'] = 0
    df['svm_anomaly'] = 0
    df['ensemble_anomaly'] = 0
    df['anomaly_score'] = 0.0
    
    # Assign values to the valid index
    df.loc[valid_index, 'if_anomaly'] = if_anom
    df.loc[valid_index, 'lof_anomaly'] = lof_anom
    df.loc[valid_index, 'svm_anomaly'] = svm_anom
    df.loc[valid_index, 'ensemble_anomaly'] = ensemble_anomaly
    df.loc[valid_index, 'anomaly_score'] = avg_score
    
    return df, if_model

def analyze_anomalies(df):
    """
    Statistics on anomalous vs normal movements.
    """
    normal = df[df['ensemble_anomaly'] == 0]
    anomalous = df[df['ensemble_anomaly'] == 1]
    
    features = ['speed_kmh', 'distance_km', 'hour', 'bearing_change']
    
    print("\n--- Anomaly Analysis Summary ---")
    analysis = {}
    for f in features:
        if f in df.columns:
            mean_norm = normal[f].mean() if not normal.empty else 0
            mean_anom = anomalous[f].mean() if not anomalous.empty else 0
            print(f"{f}: Normal Mean = {mean_norm:.2f}, Anomalous Mean = {mean_anom:.2f}")
            analysis[f] = {'normal_mean': mean_norm, 'anomalous_mean': mean_anom}
            
    print("--------------------------------\n")
    return analysis

def plot_anomaly_distribution(df):
    """
    Create 2x2 subplot of anomaly distributions.
    """
    os.makedirs('reports', exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    normal = df[df['ensemble_anomaly'] == 0]
    anomalous = df[df['ensemble_anomaly'] == 1]
    
    if 'speed_kmh' in df.columns:
        axes[0, 0].hist(normal['speed_kmh'].dropna(), bins=30, alpha=0.5, label='Normal', color='blue')
        axes[0, 0].hist(anomalous['speed_kmh'].dropna(), bins=30, alpha=0.5, label='Anomalous', color='red')
        axes[0, 0].set_title('Speed Distribution')
        axes[0, 0].legend()
        
    if 'distance_km' in df.columns:
        axes[0, 1].hist(normal['distance_km'].dropna(), bins=30, alpha=0.5, label='Normal', color='blue')
        axes[0, 1].hist(anomalous['distance_km'].dropna(), bins=30, alpha=0.5, label='Anomalous', color='red')
        axes[0, 1].set_title('Distance Distribution')
        axes[0, 1].legend()
        
    if 'hour' in df.columns:
        axes[1, 0].hist(normal['hour'].dropna(), bins=24, alpha=0.5, label='Normal', color='blue')
        axes[1, 0].hist(anomalous['hour'].dropna(), bins=24, alpha=0.5, label='Anomalous', color='red')
        axes[1, 0].set_title('Temporal Distribution (Hour)')
        axes[1, 0].legend()
        
    if 'anomaly_score' in df.columns:
        axes[1, 1].hist(df['anomaly_score'].dropna(), bins=50, color='purple')
        axes[1, 1].set_title('Anomaly Score Distribution')
        
    plt.tight_layout()
    plt.savefig('reports/anomaly_distributions.png')
    plt.close()

def plot_anomaly_map(df):
    """
    Scatter plot of lat/lon for normal and anomalous points.
    """
    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("Latitude or Longitude columns missing. Cannot plot map.")
        return
        
    os.makedirs('reports', exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    normal = df[df['ensemble_anomaly'] == 0]
    anomalous = df[df['ensemble_anomaly'] == 1]
    
    plt.scatter(normal['lon'], normal['lat'], s=10, c='blue', alpha=0.5, label='Normal')
    plt.scatter(anomalous['lon'], anomalous['lat'], s=20, c='red', alpha=0.8, label='Anomalous')
    
    plt.title('Anomaly Map')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/anomaly_map.png')
    plt.close()

def anomaly_pipeline():
    """
    Run the full anomaly detection pipeline.
    """
    print("Starting anomaly detection pipeline...")
    
    # Load data
    data_path = 'data/processed/gps_features.csv'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        return
        
    df = pd.read_csv(data_path)
    
    # Prepare features
    X, scaler, feature_names, valid_index = prepare_anomaly_features(df)
    
    # Run ensemble
    # Pass valid_index so we only update rows that had valid features
    df, if_model = ensemble_anomaly_detection(df, X, valid_index)
    
    # Analyze and plot
    analyze_anomalies(df)
    plot_anomaly_distribution(df)
    plot_anomaly_map(df)
    
    # Save models and results
    os.makedirs('models', exist_ok=True)
    joblib.dump(if_model, 'models/anomaly_model.pkl')
    
    os.makedirs('data/processed', exist_ok=True)
    df.to_csv('data/processed/gps_anomalies.csv', index=False)
    
    print("\nPipeline complete.")
    print("Models saved to models/")
    print("Results saved to data/processed/")
    print("Reports saved to reports/")
    print("\nNote: Anomalous movement does not indicate suspicious or criminal behavior. It simply means the movement pattern differs from the persons typical behavior.")

if __name__ == "__main__":
    anomaly_pipeline()
