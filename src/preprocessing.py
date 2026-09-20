"""
Preprocessing module for CASEFILE: AI-Powered Missing Person Investigation System.
Handles data cleaning, feature engineering, and preparation of GPS trajectories.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw GPS CSV file.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        Pandas DataFrame containing raw data.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} rows.")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate rows from the dataset.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        DataFrame with duplicates removed.
    """
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicate rows.")
    return df

def remove_invalid_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove invalid coordinates and points outside the Beijing bounding box.
    Beijing bounding box: lat 39.4-40.5, lon 115.5-117.5.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Filtered DataFrame.
    """
    initial_len = len(df)
    
    # Valid global coords
    df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
    df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
    
    # Beijing bounding box
    df = df[(df['latitude'] >= 39.4) & (df['latitude'] <= 40.5)]
    df = df[(df['longitude'] >= 115.5) & (df['longitude'] <= 117.5)]
    
    print(f"Removed {initial_len - len(df)} invalid/out-of-bounds coordinate rows.")
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    - Drop rows with NaN in lat/lon/timestamp
    - Fill NaN in altitude with 0
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned DataFrame.
    """
    initial_len = len(df)
    
    # Drop rows with NaN in critical columns
    df = df.dropna(subset=['latitude', 'longitude', 'timestamp'])
    
    # Fill altitude
    if 'altitude' in df.columns:
        df['altitude'] = df['altitude'].fillna(0)
        
    print(f"Removed {initial_len - len(df)} rows with missing critical values.")
    return df

def convert_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp to datetime object and sort by user_id and timestamp.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Sorted DataFrame with datetime timestamps.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['user_id', 'timestamp']).reset_index(drop=True)
    print("Timestamps converted and sorted by user_id and timestamp.")
    return df

def haversine_distance(lat1: pd.Series, lon1: pd.Series, lat2: pd.Series, lon2: pd.Series) -> pd.Series:
    """
    Vectorized Haversine formula to calculate the distance between two points on the Earth.
    
    Args:
        lat1, lon1: Latitude and longitude of point 1 (in degrees).
        lat2, lon2: Latitude and longitude of point 2 (in degrees).
        
    Returns:
        Distance in kilometers.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371 # Radius of earth in kilometers.
    return c * r

def remove_speed_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate speed between consecutive points and remove points where speed > 200 km/h.
    
    Args:
        df: Input DataFrame sorted by user_id and timestamp.
        
    Returns:
        DataFrame without speed outliers.
    """
    initial_len = len(df)
    
    # Calculate previous point's coordinates and timestamp for the same user
    df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    
    # Calculate distance (km) and time difference (hours)
    distance_km = haversine_distance(df['prev_lat'], df['prev_lon'], df['latitude'], df['longitude'])
    time_diff_hours = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds() / 3600.0
    
    # Calculate speed (km/h)
    speed_kmh = np.where(time_diff_hours > 0, distance_km / time_diff_hours, 0)
    
    df['speed_kmh'] = speed_kmh
    valid_speed_mask = (df['speed_kmh'] <= 200) | (df['speed_kmh'].isna())
    
    df = df[valid_speed_mask].copy()
    
    # Drop temporary columns
    df = df.drop(columns=['prev_lat', 'prev_lon', 'prev_timestamp', 'speed_kmh'])
    
    print(f"Removed {initial_len - len(df)} rows with speed > 200 km/h.")
    return df

def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features: hour, day_of_week, month, is_weekend, time_period.
    
    Args:
        df: Input DataFrame with datetime timestamp.
        
    Returns:
        DataFrame with new time features.
    """
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek # 0=Monday, 6=Sunday
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'] >= 5
    
    # Define time periods
    bins = [0, 6, 12, 18, 24]
    labels = ['night', 'morning', 'afternoon', 'evening']
    df['time_period'] = pd.cut(df['hour'], bins=bins, labels=labels, right=False)
    
    print("Created time features (hour, day_of_week, month, is_weekend, time_period).")
    return df

def normalize_features(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply StandardScaler to specified numerical columns and save scaler.
    
    Args:
        df: Input DataFrame.
        columns: List of column names to normalize.
        
    Returns:
        DataFrame with normalized columns.
    """
    scaler = StandardScaler()
    
    cols_to_scale = [col for col in columns if col in df.columns]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        # Save scaler
        os.makedirs('models', exist_ok=True)
        scaler_path = os.path.join('models', 'scaler.pkl')
        joblib.dump(scaler, scaler_path)
        print(f"Normalized columns {cols_to_scale} and saved scaler to {scaler_path}.")
    else:
        print("No matching columns found for normalization.")
        
    return df

def preprocess_pipeline(input_path: str, output_path: str) -> None:
    """
    Run all preprocessing steps in sequence and save cleaned data.
    
    Args:
        input_path: Path to the raw GPS data CSV.
        output_path: Path to save the cleaned data.
    """
    print("Starting preprocessing pipeline...")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return
        
    df = load_raw_data(input_path)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = remove_invalid_coordinates(df)
    df = convert_timestamps(df)
    df = remove_speed_outliers(df)
    df = create_time_features(df)
    # Note: We intentionally do NOT normalize lat/lon/altitude here.
    # Normalization destroys geospatial meaning needed for clustering, mapping, and distance calculations.
    # Normalization/scaling is applied as needed within individual ML modules.
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Saved preprocessed data to {output_path}. Final shape: {df.shape}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    preprocess_pipeline('data/raw/gps_trajectories.csv', 'data/processed/gps_cleaned.csv')
