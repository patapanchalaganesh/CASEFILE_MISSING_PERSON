"""
Module: data_collection.py
Description: Handles data collection for the CASEFILE project. Generates synthetic GPS trajectory 
data and POIs, and includes a fallback parser for the GeoLife dataset.
"""

import os
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Constants for synthetic generation
BEIJING_LAT_MIN, BEIJING_LAT_MAX = 39.85, 40.05
BEIJING_LON_MIN, BEIJING_LON_MAX = 116.20, 116.55
SEED = 42
NUM_USERS = 10
DAYS_PER_USER_MIN, DAYS_PER_USER_MAX = 30, 60
START_DATE = datetime(2023, 1, 1)

np.random.seed(SEED)
random.seed(SEED)

def get_random_location():
    """Return a random latitude and longitude within the Beijing bounding box."""
    lat = np.random.uniform(BEIJING_LAT_MIN, BEIJING_LAT_MAX)
    lon = np.random.uniform(BEIJING_LON_MIN, BEIJING_LON_MAX)
    return lat, lon

def generate_user_profile(user_id):
    """Generate a home, work, and frequent locations for a given user."""
    home = get_random_location()
    
    # Work: 2-8 km from home. Roughly 1 deg lat = 111 km. 
    # So 2-8 km is approx 0.018 to 0.072 degrees
    dist = np.random.uniform(0.018, 0.072)
    angle = np.random.uniform(0, 2 * np.pi)
    work = (home[0] + dist * np.cos(angle), home[1] + dist * np.sin(angle))
    
    frequent = [get_random_location() for _ in range(np.random.randint(3, 6))]
    occasional = [get_random_location() for _ in range(np.random.randint(1, 3))]
    
    return {
        'user_id': user_id,
        'home': home,
        'work': work,
        'frequent': frequent,
        'occasional': occasional
    }

def add_noise(location, noise_level_meters=10):
    """Add Gaussian noise to a location to simulate GPS inaccuracy."""
    # 1 deg ~ 111,000 meters. 10 meters ~ 0.00009 degrees
    noise_deg = noise_level_meters / 111000.0
    lat_noise = np.random.normal(0, noise_deg)
    lon_noise = np.random.normal(0, noise_deg)
    return location[0] + lat_noise, location[1] + lon_noise

def generate_route(start_loc, end_loc, start_time, speed_kmh=20, interval_sec=15):
    """Generate a sequence of GPS points interpolating between start and end locations."""
    # Distance in km
    dist_lat = (end_loc[0] - start_loc[0]) * 111
    dist_lon = (end_loc[1] - start_loc[1]) * 111 * np.cos(np.radians(start_loc[0]))
    dist_km = np.sqrt(dist_lat**2 + dist_lon**2)
    
    duration_hours = dist_km / speed_kmh
    duration_sec = duration_hours * 3600
    
    num_points = int(duration_sec / interval_sec)
    if num_points < 2:
        num_points = 2
        
    points = []
    for i in range(num_points):
        frac = i / (num_points - 1)
        lat = start_loc[0] + (end_loc[0] - start_loc[0]) * frac
        lon = start_loc[1] + (end_loc[1] - start_loc[1]) * frac
        
        lat, lon = add_noise((lat, lon), 10)
        current_time = start_time + timedelta(seconds=i * interval_sec)
        
        points.append({
            'latitude': lat,
            'longitude': lon,
            'altitude': np.random.normal(50, 5),
            'timestamp': current_time
        })
        
    return points, current_time

def generate_stay(location, start_time, duration_minutes, interval_sec=60):
    """Generate GPS points simulating a user staying at a specific location."""
    points = []
    num_points = int((duration_minutes * 60) / interval_sec)
    for i in range(max(1, num_points)):
        lat, lon = add_noise(location, 15)
        current_time = start_time + timedelta(seconds=i * interval_sec)
        points.append({
            'latitude': lat,
            'longitude': lon,
            'altitude': np.random.normal(50, 5),
            'timestamp': current_time
        })
    return points, start_time + timedelta(minutes=duration_minutes)

def generate_synthetic_gps_data(output_dir='data/raw'):
    """Generate the full synthetic GPS dataset for all users and days."""
    print("Generating synthetic GPS trajectories for 10 users...")
    os.makedirs(output_dir, exist_ok=True)
    
    all_records = []
    
    for uid in range(1, NUM_USERS + 1):
        profile = generate_user_profile(uid)
        num_days = np.random.randint(DAYS_PER_USER_MIN, DAYS_PER_USER_MAX + 1)
        
        for day in range(num_days):
            current_date = START_DATE + timedelta(days=day)
            is_weekend = current_date.weekday() >= 5
            
            # Start at home
            start_hour = np.random.normal(8, 0.5)
            current_time = current_date + timedelta(hours=start_hour)
            
            # Home stay (from midnight to morning)
            stay_points, current_time = generate_stay(
                profile['home'], 
                current_date, 
                start_hour * 60
            )
            for p in stay_points: p['user_id'] = uid
            all_records.extend(stay_points)
            
            if not is_weekend:
                # Go to work
                route_points, current_time = generate_route(profile['home'], profile['work'], current_time, speed_kmh=np.random.uniform(20, 40))
                for p in route_points: p['user_id'] = uid
                all_records.extend(route_points)
                
                # Work morning
                stay_points, current_time = generate_stay(profile['work'], current_time, duration_minutes=np.random.normal(240, 30))
                for p in stay_points: p['user_id'] = uid
                all_records.extend(stay_points)
                
                # Lunch
                lunch_spot = random.choice(profile['frequent'])
                route_points, current_time = generate_route(profile['work'], lunch_spot, current_time, speed_kmh=np.random.uniform(5, 20))
                for p in route_points: p['user_id'] = uid
                all_records.extend(route_points)
                
                stay_points, current_time = generate_stay(lunch_spot, current_time, duration_minutes=np.random.normal(60, 15))
                for p in stay_points: p['user_id'] = uid
                all_records.extend(stay_points)
                
                # Back to work
                route_points, current_time = generate_route(lunch_spot, profile['work'], current_time, speed_kmh=np.random.uniform(5, 20))
                for p in route_points: p['user_id'] = uid
                all_records.extend(route_points)
                
                # Work afternoon
                stay_points, current_time = generate_stay(profile['work'], current_time, duration_minutes=np.random.normal(240, 30))
                for p in stay_points: p['user_id'] = uid
                all_records.extend(stay_points)
                
                # Errands occasionally
                if random.random() < 0.3:
                    errand_spot = random.choice(profile['frequent'] + profile['occasional'])
                    route_points, current_time = generate_route(profile['work'], errand_spot, current_time)
                    for p in route_points: p['user_id'] = uid
                    all_records.extend(route_points)
                    
                    stay_points, current_time = generate_stay(errand_spot, current_time, duration_minutes=np.random.normal(45, 20))
                    for p in stay_points: p['user_id'] = uid
                    all_records.extend(stay_points)
                    
                    # Back home
                    route_points, current_time = generate_route(errand_spot, profile['home'], current_time)
                    for p in route_points: p['user_id'] = uid
                    all_records.extend(route_points)
                else:
                    # Straight home
                    route_points, current_time = generate_route(profile['work'], profile['home'], current_time)
                    for p in route_points: p['user_id'] = uid
                    all_records.extend(route_points)
                    
            else:
                # Weekend pattern
                if random.random() < 0.7:
                    # Go out
                    dest = random.choice(profile['frequent'] + profile['occasional'])
                    route_points, current_time = generate_route(profile['home'], dest, current_time)
                    for p in route_points: p['user_id'] = uid
                    all_records.extend(route_points)
                    
                    stay_points, current_time = generate_stay(dest, current_time, duration_minutes=np.random.normal(180, 60))
                    for p in stay_points: p['user_id'] = uid
                    all_records.extend(stay_points)
                    
                    # Back home
                    route_points, current_time = generate_route(dest, profile['home'], current_time)
                    for p in route_points: p['user_id'] = uid
                    all_records.extend(route_points)
            
            # Evening stay
            end_of_day = current_date + timedelta(days=1)
            remaining_minutes = (end_of_day - current_time).total_seconds() / 60
            if remaining_minutes > 0:
                stay_points, _ = generate_stay(profile['home'], current_time, duration_minutes=remaining_minutes)
                for p in stay_points: p['user_id'] = uid
                all_records.extend(stay_points)

    df = pd.DataFrame(all_records)
    output_path = os.path.join(output_dir, 'gps_trajectories.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} synthetic GPS points to {output_path}")

def parse_geolife_plt(plt_file, user_id):
    """Parse a single GeoLife .plt file."""
    try:
        df = pd.read_csv(plt_file, skiprows=6, header=None,
                         names=['latitude', 'longitude', 'zero', 'altitude', 'days_since_1900', 'date', 'time'])
        df['user_id'] = user_id
        df['timestamp'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df = df.drop(columns=['zero', 'days_since_1900', 'date', 'time'])
        return df
    except Exception as e:
        print(f"Error parsing {plt_file}: {e}")
        return pd.DataFrame()

def parse_geolife_dataset(geolife_dir, output_dir='data/raw'):
    """Parse the entire GeoLife dataset directory structure."""
    print(f"Parsing GeoLife dataset from {geolife_dir}...")
    dfs = []
    for user_id in os.listdir(geolife_dir):
        user_dir = os.path.join(geolife_dir, user_id, 'Trajectory')
        if os.path.isdir(user_dir):
            for plt_file in os.listdir(user_dir):
                if plt_file.endswith('.plt'):
                    plt_path = os.path.join(user_dir, plt_file)
                    df = parse_geolife_plt(plt_path, user_id)
                    dfs.append(df)
    
    if dfs:
        combined_df = pd.concat(dfs, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'geolife_gps.csv')
        combined_df.to_csv(output_path, index=False)
        print(f"Saved {len(combined_df)} GeoLife GPS points to {output_path}")
    else:
        print("No GeoLife PLT files found.")

def generate_pois(output_dir='data/raw'):
    """Generate a synthetic dataset of Points of Interest (POIs) in Beijing."""
    print("Generating synthetic POIs...")
    os.makedirs(output_dir, exist_ok=True)
    
    categories = ['restaurant', 'hospital', 'park', 'school', 'bus_stop', 'train_station', 'market', 'residential', 'office']
    num_pois = np.random.randint(50, 101)
    
    pois = []
    for i in range(num_pois):
        lat, lon = get_random_location()
        cat = random.choice(categories)
        pois.append({
            'poi_id': f"POI_{i+1:03d}",
            'name': f"Beijing {cat.capitalize()} {i+1}",
            'category': cat,
            'latitude': lat,
            'longitude': lon
        })
        
    df = pd.DataFrame(pois)
    output_path = os.path.join(output_dir, 'beijing_pois.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} synthetic POIs to {output_path}")

def generate_data_sources_report(output_dir='reports'):
    """Generate a markdown report documenting the data sources."""
    os.makedirs(output_dir, exist_ok=True)
    content = """# Data Sources Report

## 1. Synthetic GPS Trajectories
Generated synthetic trajectories for 10 users in Beijing mimicking daily commuting patterns with noise.
Saved to `data/raw/gps_trajectories.csv`.

## 2. Points of Interest (POIs)
Synthetic points of interest in Beijing for map features and context.
Saved to `data/raw/beijing_pois.csv`.

## 3. GeoLife Dataset (Optional)
Fallback parsing provided for the Microsoft GeoLife GPS trajectory dataset.
"""
    output_path = os.path.join(output_dir, 'data_sources.md')
    with open(output_path, 'w') as f:
        f.write(content)
    print(f"Generated report at {output_path}")

if __name__ == "__main__":
    # Change to project directory to ensure relative paths work
    base_dir = r"c:\CASEFILE_MISSING_PERSON"
    if os.path.exists(base_dir):
        os.chdir(base_dir)
        
    generate_synthetic_gps_data()
    generate_pois()
    generate_data_sources_report()
    
    print("Data collection completed successfully.")
