"""
case_generator.py
Generates synthetic missing-person case records for the CASEFILE project.

DISCLAIMER: All identities, cases, and descriptions are completely FICTIONAL
and are used for academic simulation purposes only.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

np.random.seed(42)


def _nearest_area(lat, lon, area_centers_df):
    """Find the nearest area to a given lat/lon."""
    if area_centers_df.empty:
        return 0, "Area_0"
    dists = np.sqrt(
        (area_centers_df['lat'] - lat) ** 2 + (area_centers_df['lon'] - lon) ** 2
    )
    idx = dists.idxmin()
    row = area_centers_df.loc[idx]
    return int(row['area_id']), row['name']


def generate_cases(user_profiles_df, stay_points_df, area_centers_df, n_cases=100):
    """
    Create fictional missing-person cases based on actual movement data.
    """
    cases = []
    users = user_profiles_df['user_id'].unique()

    # Determine lat/lon column names in stay points
    lat_col = 'stay_lat' if 'stay_lat' in stay_points_df.columns else 'lat'
    lon_col = 'stay_lon' if 'stay_lon' in stay_points_df.columns else 'lon'

    # Sort stay points by arrival time
    if 'arrival_time' in stay_points_df.columns:
        stay_points_df = stay_points_df.sort_values(['user_id', 'arrival_time']).reset_index(drop=True)

    weather_options = ['Clear', 'Cloudy', 'Rain', 'Fog', 'Snow']
    weather_weights = [0.4, 0.3, 0.15, 0.1, 0.05]

    age_groups = ['Under 18', '18-25', '26-35', '36-50', '51-65', 'Over 65']
    genders = ['Male', 'Female']
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    day_weights = [0.16, 0.16, 0.16, 0.16, 0.16, 0.10, 0.10]

    # Pre-filter users who have at least 3 stay points (need prev, current, next)
    valid_users = [u for u in users if len(stay_points_df[stay_points_df['user_id'] == u]) >= 3]
    if not valid_users:
        valid_users = users.tolist()

    area_names = area_centers_df['name'].tolist() if 'name' in area_centers_df.columns else [f"Area_{i}" for i in range(len(area_centers_df))]

    for i in range(1, n_cases + 1):
        case_id = f'MP-2026-{i:03d}'
        user_id = np.random.choice(valid_users)

        user_profile = user_profiles_df[user_profiles_df['user_id'] == user_id].iloc[0]
        user_pts = stay_points_df[stay_points_df['user_id'] == user_id].reset_index(drop=True)

        if len(user_pts) >= 3:
            idx = np.random.randint(1, len(user_pts) - 1)
            pt = user_pts.iloc[idx]
            prev_pt = user_pts.iloc[idx - 1]
            next_pt = user_pts.iloc[idx + 1] if idx + 1 < len(user_pts) else user_pts.iloc[0]
        elif len(user_pts) >= 1:
            pt = user_pts.iloc[0]
            prev_pt = pt
            next_pt = pt
        else:
            continue

        last_lat = pt[lat_col]
        last_lon = pt[lon_col]

        # Determine areas using nearest-area lookup
        _, usual_area = _nearest_area(user_profile.get('most_visited_lat', last_lat),
                                       user_profile.get('most_visited_lon', last_lon),
                                       area_centers_df)
        _, prev_area = _nearest_area(prev_pt[lat_col], prev_pt[lon_col], area_centers_df)
        _, target_area = _nearest_area(next_pt[lat_col], next_pt[lon_col], area_centers_df)

        # Weighted hour towards evening
        hours = list(range(24))
        hour_weights = [1]*6 + [2]*2 + [3]*4 + [3]*4 + [5]*4 + [3]*4
        hour_weights = [w / sum(hour_weights) for w in hour_weights]
        hour = int(np.random.choice(hours, p=hour_weights))
        minute = np.random.randint(0, 60)
        last_seen_time = f"{hour:02d}:{minute:02d}"

        case = {
            'Case_ID': case_id,
            'Person_ID': user_id,
            'Age_Group': np.random.choice(age_groups),
            'Gender': np.random.choice(genders),
            'Last_Latitude': round(last_lat, 6),
            'Last_Longitude': round(last_lon, 6),
            'Last_Seen_Time': last_seen_time,
            'Day': np.random.choice(days, p=day_weights),
            'Weather': np.random.choice(weather_options, p=weather_weights),
            'Usual_Area': usual_area,
            'Average_Distance': round(user_profile.get('avg_daily_distance_km', 5.0), 2),
            'Average_Speed': round(user_profile.get('avg_speed_kmh', 5.0), 2),
            'Previous_Area': prev_area,
            'Time_Since_Last_Seen': int(np.random.randint(1, 73)),
            'Target_Area': target_area,
        }
        cases.append(case)

    return pd.DataFrame(cases)


def generate_investigation_case(user_profiles_df, stay_points_df, area_centers_df):
    """
    Create ONE detailed investigation case for final demonstration.
    """
    np.random.seed(17)  # Different seed for variety
    base_case_df = generate_cases(user_profiles_df, stay_points_df, area_centers_df, n_cases=1)
    np.random.seed(42)  # Reset

    case_dict = base_case_df.iloc[0].to_dict()
    case_dict['Case_ID'] = 'MP-2026-017'
    case_dict['Age_Group'] = '18-25'
    case_dict['Last_Seen_Time'] = '18:45'
    case_dict['Day'] = 'Friday'
    case_dict['Weather'] = 'Rain'
    case_dict['description'] = (
        "Subject was last seen leaving their workplace in the evening during rain. "
        "Colleagues reported nothing unusual, but subject did not return home. "
        "Subject's belongings were found near their usual transit stop."
    )
    case_dict['investigating_officer'] = 'Det. J. Chen'
    case_dict['report_date'] = datetime.now().strftime('%Y-%m-%d')

    return pd.DataFrame([case_dict])


def encode_case_features(cases_df):
    """Label encode categorical features and save encoders."""
    cat_cols = ['Age_Group', 'Gender', 'Day', 'Weather', 'Usual_Area', 'Previous_Area', 'Target_Area']
    encoders = {}
    encoded_df = cases_df.copy()

    os.makedirs('models', exist_ok=True)

    for col in cat_cols:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoders[col] = le

    joblib.dump(encoders, 'models/case_encoders.pkl')
    return encoded_df


def case_generation_pipeline():
    """Load data, generate cases, encode, and save outputs."""
    print("Loading processed data...")
    try:
        user_profiles_df = pd.read_csv('data/processed/user_profiles.csv')
        stay_points_df = pd.read_csv('data/processed/stay_points_clustered.csv')
        area_centers_df = pd.read_csv('data/processed/area_centers.csv')
    except FileNotFoundError as e:
        print(f"Error: Missing required data file - {e}")
        print("Please run clustering.py first.")
        return

    print(f"  Users: {len(user_profiles_df)}, Stay points: {len(stay_points_df)}, Areas: {len(area_centers_df)}")

    print("Generating 100 training cases...")
    cases_df = generate_cases(user_profiles_df, stay_points_df, area_centers_df, n_cases=100)

    print("Generating investigation case MP-2026-017...")
    inv_case_df = generate_investigation_case(user_profiles_df, stay_points_df, area_centers_df)

    print("Encoding case features...")
    encoded_cases_df = encode_case_features(cases_df)

    os.makedirs('data/synthetic', exist_ok=True)
    cases_df.to_csv('data/synthetic/cases.csv', index=False)
    encoded_cases_df.to_csv('data/synthetic/cases_encoded.csv', index=False)
    inv_case_df.to_csv('data/synthetic/investigation_case.csv', index=False)

    print(f"\nGenerated {len(cases_df)} training cases.")
    print(f"Target area distribution:\n{cases_df['Target_Area'].value_counts()}")
    print(f"\nInvestigation case: {inv_case_df.iloc[0]['Case_ID']}")
    print("Pipeline complete.")


if __name__ == "__main__":
    case_generation_pipeline()
