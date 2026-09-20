import os
import math
import pandas as pd
import numpy as np

def compute_ml_prediction_score(area_probabilities):
    """
    Input: dict of {area_id: probability} from the location prediction model
    Normalize to 0-100 scale
    Return dict of {area_id: score}
    """
    if not area_probabilities:
        return {}
    
    max_prob = max(area_probabilities.values()) if area_probabilities else 1.0
    if max_prob == 0:
        max_prob = 1.0
        
    scores = {area_id: (prob / max_prob) * 100.0 for area_id, prob in area_probabilities.items()}
    return scores

def compute_visit_frequency_score(area_ids, location_frequencies_df):
    """
    Based on historical visit counts for each area
    Normalize to 0-100 (most visited = 100)
    Return dict of {area_id: score}
    """
    scores = {}
    if location_frequencies_df.empty or 'visit_count' not in location_frequencies_df.columns:
        return {area_id: 0.0 for area_id in area_ids}
        
    max_count = location_frequencies_df['visit_count'].max()
    if max_count == 0:
        max_count = 1
        
    for area_id in area_ids:
        # Assuming df has 'area_id' column or index is area_id
        if 'area_id' in location_frequencies_df.columns:
            area_data = location_frequencies_df[location_frequencies_df['area_id'] == area_id]
            count = area_data['visit_count'].iloc[0] if not area_data.empty else 0
        elif area_id in location_frequencies_df.index:
            count = location_frequencies_df.loc[area_id, 'visit_count']
        else:
            count = 0
            
        scores[area_id] = (count / max_count) * 100.0
        
    return scores

def compute_route_similarity_score(area_ids, predicted_routes, transition_matrix):
    """
    Areas appearing in top predicted routes get higher scores
    Weighted by position in route (earlier = higher)
    Normalize to 0-100
    Return dict of {area_id: score}
    """
    scores = {area_id: 0.0 for area_id in area_ids}
    if not predicted_routes:
        return scores
        
    # Assign score based on presence in predicted routes, e.g., sum of (1 / (index + 1))
    raw_scores = {area_id: 0.0 for area_id in area_ids}
    
    for route in predicted_routes:
        for idx, area_id in enumerate(route):
            if area_id in raw_scores:
                raw_scores[area_id] += 1.0 / (idx + 1)
                
    max_raw = max(raw_scores.values()) if raw_scores else 1.0
    if max_raw == 0:
        max_raw = 1.0
        
    for area_id, val in raw_scores.items():
        scores[area_id] = (val / max_raw) * 100.0
        
    return scores

def compute_distance_relevance_score(area_centers, last_known_lat, last_known_lon, avg_distance_km):
    """
    Areas closer to last known location score higher
    Penalize areas beyond 2x average historical distance
    Normalize to 0-100
    Return dict of {area_id: score}
    """
    # area_centers: dict {area_id: (lat, lon)}
    scores = {}
    
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat / 2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c
        
    max_dist = avg_distance_km * 2
    if max_dist == 0:
        max_dist = 1.0
        
    for area_id, (lat, lon) in area_centers.items():
        dist = haversine(last_known_lat, last_known_lon, lat, lon)
        if dist > max_dist:
            score = 0.0
        else:
            score = max(0.0, 100.0 * (1.0 - (dist / max_dist)))
        scores[area_id] = score
        
    return scores

def compute_time_relevance_score(area_ids, current_hour, current_day, location_frequencies_df):
    """
    Areas typically visited at similar time/day score higher
    Return dict of {area_id: score}
    """
    scores = {area_id: 0.0 for area_id in area_ids}
    # Placeholder logic - in a real scenario, we'd query location_frequencies_df for hour/day match
    # Since specific df schema isn't fully defined, we provide a mock implementation
    for area_id in area_ids:
        # Mock score
        scores[area_id] = np.random.uniform(20.0, 80.0)
    return scores

def compute_anomaly_score(area_ids, anomaly_data_df):
    """
    Areas with anomalous movements score higher (unusual behavior suggests deviation)
    Normalize to 0-100
    Return dict of {area_id: score}
    """
    scores = {area_id: 0.0 for area_id in area_ids}
    if anomaly_data_df.empty or 'anomaly_score' not in anomaly_data_df.columns:
        return scores
        
    max_anomaly = anomaly_data_df['anomaly_score'].max()
    if max_anomaly == 0:
        max_anomaly = 1.0
        
    for area_id in area_ids:
        if 'area_id' in anomaly_data_df.columns:
            area_data = anomaly_data_df[anomaly_data_df['area_id'] == area_id]
            val = area_data['anomaly_score'].iloc[0] if not area_data.empty else 0.0
        elif area_id in anomaly_data_df.index:
            val = anomaly_data_df.loc[area_id, 'anomaly_score']
        else:
            val = 0.0
            
        scores[area_id] = (val / max_anomaly) * 100.0
        
    return scores

def compute_search_priority(area_ids, weights=None, ml_prediction=None, visit_frequency=None, 
                            route_similarity=None, distance_relevance=None, 
                            time_relevance=None, anomaly_evidence=None):
    """
    Compute weighted sum for each area and assign priority labels
    Return DataFrame
    """
    if weights is None:
        weights = {
            'ml_prediction': 0.30,
            'visit_frequency': 0.20,
            'route_similarity': 0.15,
            'distance_relevance': 0.15,
            'time_relevance': 0.10,
            'anomaly_evidence': 0.10
        }
        
    data = []
    for area_id in area_ids:
        ml = ml_prediction.get(area_id, 0.0) if ml_prediction else 0.0
        vf = visit_frequency.get(area_id, 0.0) if visit_frequency else 0.0
        rs = route_similarity.get(area_id, 0.0) if route_similarity else 0.0
        dr = distance_relevance.get(area_id, 0.0) if distance_relevance else 0.0
        tr = time_relevance.get(area_id, 0.0) if time_relevance else 0.0
        ae = anomaly_evidence.get(area_id, 0.0) if anomaly_evidence else 0.0
        
        composite_score = (
            ml * weights['ml_prediction'] +
            vf * weights['visit_frequency'] +
            rs * weights['route_similarity'] +
            dr * weights['distance_relevance'] +
            tr * weights['time_relevance'] +
            ae * weights['anomaly_evidence']
        )
        
        if composite_score <= 30:
            priority = 'Low'
        elif composite_score <= 60:
            priority = 'Medium'
        elif composite_score <= 80:
            priority = 'High'
        else:
            priority = 'Very High'
            
        data.append({
            'area_id': area_id,
            'area_name': f"Area_{area_id}",
            'composite_score': composite_score,
            'priority': priority,
            'ml_score': ml,
            'visit_score': vf,
            'route_score': rs,
            'distance_score': dr,
            'time_score': tr,
            'anomaly_score': ae
        })
        
    df = pd.DataFrame(data)
    df = df.sort_values(by='composite_score', ascending=False).reset_index(drop=True)
    return df

def search_priority_pipeline(case_data, area_centers, location_freqs, predicted_routes, transition_matrix, anomaly_data, prediction_probs):
    print("Running Search Priority Pipeline...")
    area_ids = list(area_centers.keys())
    
    ml_scores = compute_ml_prediction_score(prediction_probs)
    vf_scores = compute_visit_frequency_score(area_ids, location_freqs)
    rs_scores = compute_route_similarity_score(area_ids, predicted_routes, transition_matrix)
    
    last_known_lat = case_data.get('last_known_lat', 39.9)
    last_known_lon = case_data.get('last_known_lon', 116.4)
    avg_distance_km = case_data.get('avg_distance_km', 5.0)
    
    dr_scores = compute_distance_relevance_score(area_centers, last_known_lat, last_known_lon, avg_distance_km)
    
    current_hour = case_data.get('current_hour', 12)
    current_day = case_data.get('current_day', 0)
    tr_scores = compute_time_relevance_score(area_ids, current_hour, current_day, location_freqs)
    
    ae_scores = compute_anomaly_score(area_ids, anomaly_data)
    
    priority_df = compute_search_priority(
        area_ids,
        ml_prediction=ml_scores,
        visit_frequency=vf_scores,
        route_similarity=rs_scores,
        distance_relevance=dr_scores,
        time_relevance=tr_scores,
        anomaly_evidence=ae_scores
    )
    
    os.makedirs('data/processed', exist_ok=True)
    priority_df.to_csv('data/processed/search_priority.csv', index=False)
    
    print("\n--- Top Search Priorities ---")
    print(priority_df.head())
    
    return priority_df

if __name__ == "__main__":
    # Mock data for testing
    mock_case_data = {
        'last_known_lat': 39.95,
        'last_known_lon': 116.45,
        'avg_distance_km': 10.0,
        'current_hour': 18,
        'current_day': 4
    }
    mock_area_centers = {
        1: (39.90, 116.40),
        2: (39.96, 116.46),
        3: (40.00, 116.50)
    }
    mock_location_freqs = pd.DataFrame({
        'area_id': [1, 2, 3],
        'visit_count': [50, 10, 5]
    })
    mock_predicted_routes = [[2, 3], [1, 2]]
    mock_transition_matrix = {}
    mock_anomaly_data = pd.DataFrame({
        'area_id': [1, 2, 3],
        'anomaly_score': [0.1, 0.9, 0.2]
    })
    mock_prediction_probs = {
        1: 0.1,
        2: 0.8,
        3: 0.3
    }
    
    search_priority_pipeline(
        mock_case_data, 
        mock_area_centers, 
        mock_location_freqs, 
        mock_predicted_routes, 
        mock_transition_matrix, 
        mock_anomaly_data, 
        mock_prediction_probs
    )
