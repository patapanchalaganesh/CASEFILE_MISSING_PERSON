"""
Feature engineering module for the CASEFILE: AI-Powered Missing Person Investigation System.
Derives movement-related features from cleaned GPS data.
"""

import pandas as pd
import numpy as np
import os
import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def haversine_distance(lat1, lon1, lat2, lon2):
    """Vectorized Haversine distance in km."""
    R = 6371.0
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def compute_point_features(df):
    """Compute features for consecutive GPS points."""
    logging.info("Computing point features...")
    df = df.copy()
    df.sort_values(by=['user_id', 'timestamp'], inplace=True)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Shifted columns
    df['prev_lat'] = df.groupby('user_id')['latitude'].shift(1)
    df['prev_lon'] = df.groupby('user_id')['longitude'].shift(1)
    df['prev_timestamp'] = df.groupby('user_id')['timestamp'].shift(1)
    
    df['distance_km'] = haversine_distance(df['prev_lat'], df['prev_lon'], df['latitude'], df['longitude'])
    df['time_delta_seconds'] = (df['timestamp'] - df['prev_timestamp']).dt.total_seconds()
    
    # Speed
    df['speed_kmh'] = np.where(df['time_delta_seconds'] > 0, 
                               (df['distance_km'] / (df['time_delta_seconds'] / 3600.0)), 0)
    
    # Acceleration
    df['prev_speed'] = df.groupby('user_id')['speed_kmh'].shift(1)
    df['acceleration'] = np.where(df['time_delta_seconds'] > 0,
                                  (df['speed_kmh'] - df['prev_speed']) / df['time_delta_seconds'], 0)
    
    # Bearing
    lat1 = np.radians(df['prev_lat'])
    lat2 = np.radians(df['latitude'])
    diff_long = np.radians(df['longitude'] - df['prev_lon'])
    
    x = np.sin(diff_long) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - (np.sin(lat1) * np.cos(lat2) * np.cos(diff_long))
    
    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    df['bearing'] = (initial_bearing + 360) % 360
    
    df['prev_bearing'] = df.groupby('user_id')['bearing'].shift(1)
    df['bearing_change'] = np.abs(df['bearing'] - df['prev_bearing'])
    df['bearing_change'] = np.where(df['bearing_change'] > 180, 360 - df['bearing_change'], df['bearing_change'])
    
    df.drop(columns=['prev_lat', 'prev_lon', 'prev_timestamp', 'prev_speed', 'prev_bearing'], inplace=True)
    
    return df

def compute_trajectory_features(df):
    """Group consecutive points into trajectories and compute features."""
    logging.info("Computing trajectory features...")
    df = df.copy()
    
    # Gap > 20 mins = new trajectory
    df['gap_gt_20m'] = (df['time_delta_seconds'] > 1200).astype(int)
    # First point for each user is a new trajectory
    df['new_user'] = (df['user_id'] != df['user_id'].shift(1)).astype(int)
    df['traj_start'] = df['gap_gt_20m'] | df['new_user']
    df['trajectory_id'] = df.groupby('user_id')['traj_start'].cumsum()
    df['trajectory_id'] = df['user_id'].astype(str) + '_' + df['trajectory_id'].astype(str)
    
    def traj_stats(group):
        if len(group) < 2:
            return None
        
        start_pt = group.iloc[0]
        end_pt = group.iloc[-1]
        
        total_distance_km = group['distance_km'].sum()
        duration_minutes = (end_pt['timestamp'] - start_pt['timestamp']).total_seconds() / 60.0
        
        avg_speed_kmh = total_distance_km / (duration_minutes / 60.0) if duration_minutes > 0 else 0
        max_speed_kmh = group['speed_kmh'].max()
        
        straight_line_distance = haversine_distance(start_pt['latitude'], start_pt['longitude'], 
                                                    end_pt['latitude'], end_pt['longitude'])
        
        sinuosity = total_distance_km / straight_line_distance if straight_line_distance > 0 else 1.0
        
        return pd.Series({
            'trajectory_id': group.name,
            'user_id': start_pt['user_id'],
            'total_distance_km': total_distance_km,
            'duration_minutes': duration_minutes,
            'average_speed_kmh': avg_speed_kmh,
            'max_speed_kmh': max_speed_kmh,
            'start_lat': start_pt['latitude'],
            'start_lon': start_pt['longitude'],
            'end_lat': end_pt['latitude'],
            'end_lon': end_pt['longitude'],
            'straight_line_distance': straight_line_distance,
            'sinuosity': sinuosity
        })
        
    summary = df.groupby('trajectory_id').apply(traj_stats, include_groups=False).dropna().reset_index(drop=True)
    return df, summary

def compute_stay_points(df, distance_threshold=0.1, time_threshold=300):
    """Identify stay points."""
    logging.info("Computing stay points...")
    stay_points = []
    
    for user_id, group in df.groupby('user_id'):
        pts = group.to_dict('records')
        i = 0
        while i < len(pts):
            j = i + 1
            while j < len(pts):
                dist = haversine_distance(pts[i]['latitude'], pts[i]['longitude'], pts[j]['latitude'], pts[j]['longitude'])
                if dist > distance_threshold:
                    break
                j += 1
            
            # Sub-trajectory from i to j-1
            time_diff = (pts[j-1]['timestamp'] - pts[i]['timestamp']).total_seconds()
            if time_diff >= time_threshold:
                sub_pts = pts[i:j]
                lats = [p['latitude'] for p in sub_pts]
                lons = [p['longitude'] for p in sub_pts]
                
                stay_points.append({
                    'user_id': user_id,
                    'stay_lat': np.mean(lats),
                    'stay_lon': np.mean(lons),
                    'arrival_time': pts[i]['timestamp'],
                    'departure_time': pts[j-1]['timestamp'],
                    'dwell_time_minutes': time_diff / 60.0
                })
                i = j
            else:
                i += 1
                
    return pd.DataFrame(stay_points)

def compute_user_profiles(df, stay_points_df, trajectories_df):
    """Aggregate user profiles."""
    logging.info("Computing user profiles...")
    profiles = []
    
    for user_id, group in df.groupby('user_id'):
        user_trajs = trajectories_df[trajectories_df['user_id'] == user_id]
        user_stays = stay_points_df[stay_points_df['user_id'] == user_id] if not stay_points_df.empty else pd.DataFrame()
        
        total_trajectories = len(user_trajs)
        total_distance_km = user_trajs['total_distance_km'].sum()
        total_points = len(group)
        
        days = (group['timestamp'].max() - group['timestamp'].min()).days
        if days == 0: days = 1
        
        avg_daily_distance_km = total_distance_km / days
        avg_speed_kmh = user_trajs['average_speed_kmh'].mean() if not user_trajs.empty else 0
        max_speed_kmh = user_trajs['max_speed_kmh'].max() if not user_trajs.empty else 0
        avg_trajectory_duration_min = user_trajs['duration_minutes'].mean() if not user_trajs.empty else 0
        
        active_hours = group['timestamp'].dt.hour.value_counts().head(3).index.tolist()
        
        # Stays
        if len(user_stays) > 0:
            user_stays = user_stays.copy()
            user_stays['rounded_lat'] = user_stays['stay_lat'].round(3)
            user_stays['rounded_lon'] = user_stays['stay_lon'].round(3)
            user_stays['loc_str'] = user_stays['rounded_lat'].astype(str) + "_" + user_stays['rounded_lon'].astype(str)
            
            num_unique_stay_points = user_stays['loc_str'].nunique()
            most_visited = user_stays['loc_str'].mode().iloc[0]
            most_visited_lat = user_stays[user_stays['loc_str'] == most_visited]['stay_lat'].iloc[0]
            most_visited_lon = user_stays[user_stays['loc_str'] == most_visited]['stay_lon'].iloc[0]
            
            # Home logic
            user_stays['hour'] = user_stays['arrival_time'].dt.hour
            night_stays = user_stays[(user_stays['hour'] >= 22) | (user_stays['hour'] <= 6)]
            if len(night_stays) > 0:
                home_loc = night_stays['loc_str'].mode().iloc[0]
                home_lat = night_stays[night_stays['loc_str'] == home_loc]['stay_lat'].iloc[0]
                home_lon = night_stays[night_stays['loc_str'] == home_loc]['stay_lon'].iloc[0]
            else:
                home_lat, home_lon = np.nan, np.nan
                
            # Work logic
            day_stays = user_stays[(user_stays['hour'] >= 9) & (user_stays['hour'] <= 17) & (user_stays['arrival_time'].dt.dayofweek < 5)]
            if len(day_stays) > 0:
                work_loc = day_stays['loc_str'].mode().iloc[0]
                work_lat = day_stays[day_stays['loc_str'] == work_loc]['stay_lat'].iloc[0]
                work_lon = day_stays[day_stays['loc_str'] == work_loc]['stay_lon'].iloc[0]
            else:
                work_lat, work_lon = np.nan, np.nan
                
        else:
            num_unique_stay_points = 0
            most_visited_lat, most_visited_lon = np.nan, np.nan
            home_lat, home_lon = np.nan, np.nan
            work_lat, work_lon = np.nan, np.nan
            
        profiles.append({
            'user_id': user_id,
            'total_trajectories': total_trajectories,
            'total_distance_km': total_distance_km,
            'total_points': total_points,
            'avg_daily_distance_km': avg_daily_distance_km,
            'avg_speed_kmh': avg_speed_kmh,
            'max_speed_kmh': max_speed_kmh,
            'num_unique_stay_points': num_unique_stay_points,
            'most_visited_lat': most_visited_lat,
            'most_visited_lon': most_visited_lon,
            'avg_trajectory_duration_min': avg_trajectory_duration_min,
            'active_hours': str(active_hours),
            'home_lat': home_lat,
            'home_lon': home_lon,
            'work_lat': work_lat,
            'work_lon': work_lon
        })
        
    return pd.DataFrame(profiles)

def compute_location_visit_frequency(stay_points_df):
    """Compute frequency of visits to grid locations."""
    logging.info("Computing location visit frequencies...")
    if len(stay_points_df) == 0:
        return pd.DataFrame()
        
    sp = stay_points_df.copy()
    sp['grid_lat'] = sp['stay_lat'].round(3)
    sp['grid_lon'] = sp['stay_lon'].round(3)
    sp['grid_cell'] = sp['grid_lat'].astype(str) + '_' + sp['grid_lon'].astype(str)
    
    freq = []
    for (user_id, grid_cell), group in sp.groupby(['user_id', 'grid_cell']):
        freq.append({
            'user_id': user_id,
            'grid_cell': grid_cell,
            'grid_lat': group['grid_lat'].iloc[0],
            'grid_lon': group['grid_lon'].iloc[0],
            'visit_frequency': len(group),
            'avg_dwell_time_minutes': group['dwell_time_minutes'].mean(),
            'most_common_hour': group['arrival_time'].dt.hour.mode().iloc[0] if not group['arrival_time'].dt.hour.mode().empty else np.nan,
            'most_common_day': group['arrival_time'].dt.dayofweek.mode().iloc[0] if not group['arrival_time'].dt.dayofweek.mode().empty else np.nan
        })
        
    return pd.DataFrame(freq)

def spatial_binning(df, grid_size=0.005):
    """Assign GPS points to spatial grid cells."""
    logging.info("Applying spatial binning...")
    df = df.copy()
    
    df['grid_center_lat'] = (df['latitude'] // grid_size) * grid_size + (grid_size / 2)
    df['grid_center_lon'] = (df['longitude'] // grid_size) * grid_size + (grid_size / 2)
    
    df['grid_cell_id'] = 'grid_' + df['grid_center_lat'].astype(str) + '_' + df['grid_center_lon'].astype(str)
    
    # Area ID label encoding
    unique_cells = df['grid_cell_id'].unique()
    cell_to_id = {cell: i for i, cell in enumerate(unique_cells)}
    df['area_id'] = df['grid_cell_id'].map(cell_to_id)
    
    return df

def feature_pipeline(input_path, data_dir):
    """Run all feature computation and save outputs."""
    logging.info(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df = compute_point_features(df)
    df, trajectories_summary = compute_trajectory_features(df)
    stay_points = compute_stay_points(df)
    user_profiles = compute_user_profiles(df, stay_points, trajectories_summary)
    location_freq = compute_location_visit_frequency(stay_points)
    df = spatial_binning(df)
    
    logging.info("Saving outputs...")
    os.makedirs(data_dir, exist_ok=True)
    
    df.to_csv(os.path.join(data_dir, 'gps_features.csv'), index=False)
    trajectories_summary.to_csv(os.path.join(data_dir, 'trajectory_summary.csv'), index=False)
    stay_points.to_csv(os.path.join(data_dir, 'stay_points.csv'), index=False)
    user_profiles.to_csv(os.path.join(data_dir, 'user_profiles.csv'), index=False)
    location_freq.to_csv(os.path.join(data_dir, 'location_frequencies.csv'), index=False)
    
    logging.info("Feature engineering pipeline completed.")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, 'data', 'processed', 'gps_cleaned.csv')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    
    if os.path.exists(input_file):
        feature_pipeline(input_file, processed_dir)
    else:
        logging.error(f"Input file {input_file} not found. Ensure the data exists.")
