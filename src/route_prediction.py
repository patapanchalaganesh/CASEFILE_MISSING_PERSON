"""
Markov Chain route prediction module for CASEFILE: AI-Powered Missing Person Investigation System.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib.patches import FancyArrowPatch

def build_transition_sequences(gps_df):
    """
    Extract area transition sequences from GPS data.
    """
    sequences = {}
    for user_id, group in gps_df.groupby('user_id'):
        sorted_group = group.sort_values('timestamp')
        areas = sorted_group['area_id'].tolist()
        
        # Keep consecutive changes only
        transitions = []
        if areas:
            transitions.append(areas[0])
            for a in areas[1:]:
                if a != transitions[-1] and pd.notnull(a):
                    transitions.append(a)
        sequences[user_id] = transitions
    return sequences

def build_transition_matrix(sequences, n_areas=None):
    """
    Build transition probability matrix.
    """
    unique_areas = set()
    for seq in sequences.values():
        unique_areas.update([int(a) for a in seq if pd.notnull(a)])
    
    unique_areas = sorted(list(unique_areas))
    area_to_idx = {area: idx for idx, area in enumerate(unique_areas)}
    idx_to_area = {idx: area for area, idx in area_to_idx.items()}
    
    N = len(unique_areas)
    if n_areas and n_areas > N:
        N = n_areas
        
    counts = np.zeros((N, N))
    
    for seq in sequences.values():
        for i in range(len(seq) - 1):
            src = int(seq[i])
            dst = int(seq[i+1])
            if src in area_to_idx and dst in area_to_idx:
                counts[area_to_idx[src], area_to_idx[dst]] += 1
            
    matrix = np.zeros((N, N))
    for i in range(N):
        row_sum = np.sum(counts[i, :])
        if row_sum > 0:
            matrix[i, :] = counts[i, :] / row_sum
        else:
            matrix[i, :] = 1.0 / N
            
    return matrix, area_to_idx, idx_to_area

def predict_route(transition_matrix, start_area_idx, n_steps=5, idx_to_area=None, area_names=None):
    """
    Predict most probable sequence.
    """
    current_idx = start_area_idx
    route = []
    cumulative_prob = 1.0
    
    for step in range(n_steps):
        probs = transition_matrix[current_idx, :]
        next_idx = np.argmax(probs)
        step_prob = probs[next_idx]
        cumulative_prob *= step_prob
        
        area_id = idx_to_area[next_idx] if idx_to_area else next_idx
        area_name = area_names.get(area_id, str(area_id)) if area_names else str(area_id)
        
        route.append({
            'step': step + 1,
            'area_id': area_id,
            'area_name': area_name,
            'step_probability': float(step_prob),
            'cumulative_probability': float(cumulative_prob)
        })
        current_idx = next_idx
        
    return route

def predict_top_routes(transition_matrix, start_area_idx, n_steps=4, top_k=3, idx_to_area=None, area_names=None):
    """
    Beam search for top K routes.
    """
    # Beam contains tuples of (current_idx, cumulative_prob, path)
    beam = [(start_area_idx, 1.0, [])]
    
    for step in range(n_steps):
        new_beam = []
        for current_idx, cum_prob, path in beam:
            probs = transition_matrix[current_idx, :]
            # Get top k for this step to expand
            top_indices = np.argsort(probs)[-top_k:]
            
            for next_idx in top_indices:
                step_prob = probs[next_idx]
                new_cum_prob = cum_prob * step_prob
                
                area_id = idx_to_area[next_idx] if idx_to_area else next_idx
                area_name = area_names.get(area_id, str(area_id)) if area_names else str(area_id)
                
                new_step = {
                    'step': step + 1,
                    'area_id': area_id,
                    'area_name': area_name,
                    'step_probability': float(step_prob),
                    'cumulative_probability': float(new_cum_prob)
                }
                new_path = path + [new_step]
                new_beam.append((next_idx, new_cum_prob, new_path))
        
        # Sort by cumulative prob and keep top_k
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:top_k]
        
    return [path for _, _, path in beam]

def compute_route_similarity(predicted_route, historical_routes):
    """
    Compare predicted route with actual historical routes.
    """
    pred_areas = [step['area_id'] for step in predicted_route]
    pred_set = set(pred_areas)
    
    similarities = []
    
    for hist_route in historical_routes:
        hist_set = set(hist_route)
        
        # Jaccard
        intersection = len(pred_set.intersection(hist_set))
        union = len(pred_set.union(hist_set))
        jaccard = intersection / union if union > 0 else 0
        
        # LCS
        m, n = len(pred_areas), len(hist_route)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred_areas[i-1] == hist_route[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        seq_match = lcs / max(m, n) if max(m, n) > 0 else 0
        
        similarities.append({
            'jaccard': jaccard,
            'seq_match': seq_match
        })
        
    return similarities

def plot_transition_matrix(matrix, idx_to_area, area_names=None):
    """
    Heatmap of transition probabilities.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Transition Probability')
    
    labels = [area_names.get(idx_to_area[i], str(idx_to_area[i])) if area_names else str(idx_to_area[i]) 
              for i in range(len(idx_to_area))]
    
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    
    plt.xlabel('Destination Area')
    plt.ylabel('Source Area')
    plt.title('Area Transition Probabilities')
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/transition_matrix.png')
    plt.close()

def plot_transition_graph(matrix, idx_to_area, area_centers, area_names=None, threshold=0.05):
    """
    Network-style visualization using matplotlib.
    """
    plt.figure(figsize=(12, 10))
    ax = plt.gca()
    
    # Extract positions
    pos = {}
    for i, area_id in idx_to_area.items():
        # find area in area_centers
        row = area_centers[area_centers['area_id'] == area_id]
        if not row.empty:
            lon_col = 'lon' if 'lon' in row.columns else 'longitude'
            lat_col = 'lat' if 'lat' in row.columns else 'latitude'
            pos[i] = (row.iloc[0][lon_col], row.iloc[0][lat_col])
        else:
            # fallback
            pos[i] = (np.random.rand(), np.random.rand())
            
    # Draw edges
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            prob = matrix[i, j]
            if prob > threshold and i != j:
                x1, y1 = pos[i]
                x2, y2 = pos[j]
                
                # Draw arrow
                arrow = FancyArrowPatch((x1, y1), (x2, y2), 
                                        arrowstyle='->', 
                                        mutation_scale=20,
                                        lw=prob * 5, 
                                        color='blue', 
                                        alpha=0.6,
                                        connectionstyle="arc3,rad=0.1")
                ax.add_patch(arrow)
                
                # Label
                mx, my = (x1 + x2)/2, (y1 + y2)/2
                plt.text(mx, my, f'{prob:.2f}', fontsize=8, color='darkblue')

    # Draw nodes
    for i, (x, y) in pos.items():
        plt.scatter(x, y, s=500, c='red', zorder=5)
        label = area_names.get(idx_to_area[i], str(idx_to_area[i])) if area_names else str(idx_to_area[i])
        plt.text(x, y, label, fontsize=10, ha='center', va='center', color='white', fontweight='bold', zorder=6)

    plt.title('Area Transition Graph (Probability > {})'.format(threshold))
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/transition_graph.png')
    plt.close()

def route_prediction_pipeline():
    """
    Pipeline to build Markov chain and predict routes.
    """
    print("Starting route prediction pipeline...")
    
    # 1. Load data
    try:
        gps_df = pd.read_csv('data/processed/gps_with_areas.csv')
        area_centers = pd.read_csv('data/processed/area_centers.csv')
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please run area clustering first.")
        # Create dummy data for testing if files don't exist
        print("Creating dummy data for demonstration...")
        os.makedirs('data/processed', exist_ok=True)
        gps_df = pd.DataFrame({
            'user_id': [1, 1, 1, 1, 1, 2, 2, 2],
            'timestamp': ['2023-01-01 10:00', '2023-01-01 10:10', '2023-01-01 10:30', '2023-01-01 11:00', '2023-01-01 11:30',
                          '2023-01-01 10:00', '2023-01-01 10:20', '2023-01-01 10:40'],
            'area_id': [1, 1, 2, 3, 1, 2, 3, 2]
        })
        area_centers = pd.DataFrame({
            'area_id': [1, 2, 3],
            'latitude': [39.9, 39.95, 40.0],
            'longitude': [116.3, 116.35, 116.4]
        })
        gps_df.to_csv('data/processed/gps_with_areas.csv', index=False)
        area_centers.to_csv('data/processed/area_centers.csv', index=False)

    # 2. Build sequences
    print("Building transition sequences...")
    sequences = build_transition_sequences(gps_df)
    
    # 3. Build transition matrix
    print("Building transition matrix...")
    matrix, area_to_idx, idx_to_area = build_transition_matrix(sequences, n_areas=None)
    
    # 4. Predict routes
    print("Predicting top routes...")
    start_area = list(area_to_idx.keys())[0] if area_to_idx else 0
    start_idx = area_to_idx.get(start_area, 0)
    
    top_routes = predict_top_routes(matrix, start_idx, n_steps=4, top_k=3, idx_to_area=idx_to_area)
    
    # Print summary
    print(f"Top 3 routes starting from Area {start_area}:")
    for i, route in enumerate(top_routes):
        prob = route[-1]['cumulative_probability'] if route else 0
        path = " -> ".join([str(step['area_id']) for step in route])
        print(f"Route {i+1} (Prob: {prob:.4f}): Area {start_area} -> {path}")
        
    # 5. Compute similarities
    print("Computing similarities with historical sequences...")
    historical = list(sequences.values())
    if top_routes and historical:
        sims = compute_route_similarity(top_routes[0], historical)
        avg_jaccard = sum([s['jaccard'] for s in sims]) / len(sims) if sims else 0
        avg_seq = sum([s['seq_match'] for s in sims]) / len(sims) if sims else 0
        print(f"Top route avg Jaccard similarity to history: {avg_jaccard:.4f}")
        print(f"Top route avg Sequence match to history: {avg_seq:.4f}")
    
    # 6. Plotting
    print("Generating plots...")
    plot_transition_matrix(matrix, idx_to_area)
    plot_transition_graph(matrix, idx_to_area, area_centers)
    
    # 7. Save outputs
    print("Saving outputs...")
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    joblib.dump({
        'matrix': matrix,
        'area_to_idx': area_to_idx,
        'idx_to_area': idx_to_area
    }, 'models/transition_matrix.pkl')
    
    # Flatten top_routes for CSV
    routes_df_data = []
    for i, route in enumerate(top_routes):
        for step in route:
            row = {'route_rank': i+1, **step}
            routes_df_data.append(row)
            
    pd.DataFrame(routes_df_data).to_csv('data/processed/predicted_routes.csv', index=False)
    print("Pipeline complete.")

if __name__ == '__main__':
    route_prediction_pipeline()
