import os
import pandas as pd
import numpy as np
import folium
from folium import plugins
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for file paths
DATA_DIR = "data/processed"
REPORTS_DIR = "reports"

def create_investigation_map(case_data=None, area_centers=None, stay_points=None, 
                             predicted_areas=None, predicted_routes=None, 
                             anomaly_points=None, search_priority=None, map_center=None):
    """
    Create a comprehensive Folium map with multiple layers.
    """
    if map_center is None:
        if case_data is not None and not case_data.empty and 'last_known_lat' in case_data.columns:
            map_center = [case_data['last_known_lat'].iloc[0], case_data['last_known_lon'].iloc[0]]
        else:
            map_center = [39.9, 116.4] # Default Beijing

    ESRI_TILES = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
    m = folium.Map(location=map_center, zoom_start=12, tiles=ESRI_TILES, attr='Esri')

    # b) Last Known Location
    if case_data is not None and not case_data.empty:
        fg_lkl = folium.FeatureGroup(name='Last Known Location')
        for _, row in case_data.iterrows():
            lat = row.get('last_known_lat', map_center[0])
            lon = row.get('last_known_lon', map_center[1])
            time = row.get('last_seen_time', 'Unknown')
            case_id = row.get('case_id', 'Unknown')
            
            html = f"<b>Case ID:</b> {case_id}<br><b>Last Seen:</b> {time}<br><b>Coords:</b> {lat:.4f}, {lon:.4f}"
            iframe = folium.IFrame(html, width=200, height=100)
            popup = folium.Popup(iframe, max_width=200)
            
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=folium.Icon(color='red', icon='exclamation-sign')
            ).add_to(fg_lkl)
        fg_lkl.add_to(m)

    # c) Frequently Visited Areas layer
    if area_centers is not None and not area_centers.empty:
        fg_frequent = folium.FeatureGroup(name='Frequent Locations')
        for _, row in area_centers.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            area_id = row.get('area_id', row.get('cluster_id', 'Unknown'))
            visit_count = row.get('visit_count', row.get('num_points', 10))
            avg_dwell = row.get('avg_dwell_time', 'Unknown')
            
            html = f"<b>Area ID:</b> {area_id}<br><b>Visits:</b> {visit_count}<br><b>Avg Dwell:</b> {avg_dwell}"
            iframe = folium.IFrame(html, width=200, height=100)
            popup = folium.Popup(iframe, max_width=200)
            
            radius = min(max(int(float(visit_count)) * 10, 50), 500)
            folium.Circle(
                location=[lat, lon],
                radius=radius,
                popup=popup,
                color='blue',
                fill=True,
                fill_opacity=0.4
            ).add_to(fg_frequent)
        fg_frequent.add_to(m)

    # d) Predicted Areas layer
    if predicted_areas is not None and not predicted_areas.empty:
        fg_pred = folium.FeatureGroup(name='Predicted Areas')
        for i, row in predicted_areas.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            rank = row.get('rank', i+1)
            area_id = row.get('area_id', 'Unknown')
            prob = row.get('probability', 0.5)
            priority = row.get('priority_level', 'High')
            
            html = f"<b>Rank:</b> {rank}<br><b>Area:</b> {area_id}<br><b>Probability:</b> {prob:.2f}<br><b>Priority:</b> {priority}"
            iframe = folium.IFrame(html, width=200, height=120)
            popup = folium.Popup(iframe, max_width=200)
            
            icon = plugins.BeautifyIcon(
                number=int(rank),
                border_color='green',
                text_color='green',
                inner_icon_style='margin-top:0;'
            )
            
            folium.Marker(
                location=[lat, lon],
                popup=popup,
                icon=icon
            ).add_to(fg_pred)
            
            # Circle overlay
            folium.Circle(
                location=[lat, lon],
                radius=prob * 1000,
                color='green',
                fill=True,
                fill_opacity=0.2
            ).add_to(fg_pred)
        fg_pred.add_to(m)

    # e) Probable Route layer
    if predicted_routes is not None and not predicted_routes.empty:
        fg_route = folium.FeatureGroup(name='Probable Route')
        
        if 'step' in predicted_routes.columns:
            predicted_routes = predicted_routes.sort_values('step')
        
        # If routes have area_id but no lat/lon, merge with area_centers
        if 'latitude' not in predicted_routes.columns and 'area_id' in predicted_routes.columns:
            if area_centers is not None and not area_centers.empty:
                predicted_routes = predicted_routes.merge(
                    area_centers[['area_id', 'latitude', 'longitude']],
                    on='area_id', how='left'
                )
        
        if 'latitude' in predicted_routes.columns and 'longitude' in predicted_routes.columns:
            route_coords = predicted_routes[['latitude', 'longitude']].dropna().values.tolist()
            
            if len(route_coords) > 1:
                try:
                    plugins.AntPath(
                        locations=route_coords,
                        color='orange',
                        weight=5,
                        opacity=0.7,
                        delay=1000
                    ).add_to(fg_route)
                except Exception:
                    folium.PolyLine(
                        locations=route_coords,
                        color='orange',
                        weight=5,
                        opacity=0.7
                    ).add_to(fg_route)
                    
                for _, row in predicted_routes.iterrows():
                    if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                        lat = row['latitude']
                        lon = row['longitude']
                        prob = row.get('step_probability', row.get('probability', 0.0))
                        name = row.get('area_name', '')
                        html = f"<b>{name}</b><br><b>Step Prob:</b> {prob:.2f}"
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=5,
                            color='orange',
                            fill=True,
                            popup=folium.Popup(html, max_width=150)
                        ).add_to(fg_route)
                
        fg_route.add_to(m)

    # f) Anomalous Locations layer
    if anomaly_points is not None and not anomaly_points.empty:
        fg_anom = folium.FeatureGroup(name='Anomalies')
        for _, row in anomaly_points.iterrows():
            if row.get('anomaly', -1) == -1:
                lat = row['latitude']
                lon = row['longitude']
                score = row.get('anomaly_score', 'Unknown')
                speed = row.get('speed', 'Unknown')
                time = row.get('datetime', 'Unknown')
                
                html = f"<b>Score:</b> {score}<br><b>Speed:</b> {speed}<br><b>Time:</b> {time}"
                iframe = folium.IFrame(html, width=200, height=100)
                popup = folium.Popup(iframe, max_width=200)
                
                folium.RegularPolygonMarker(
                    location=[lat, lon],
                    popup=popup,
                    number_of_sides=3,
                    radius=8,
                    color='red',
                    fill_color='red'
                ).add_to(fg_anom)
        fg_anom.add_to(m)

    # g) Search Priority Heatmap layer
    if search_priority is not None and not search_priority.empty:
        fg_priority = folium.FeatureGroup(name='Search Priority')
        
        priority_colors = {
            'Very High': 'darkred',
            'High': 'orange',
            'Medium': 'yellow',
            'Low': 'lightgreen'
        }
        
        for _, row in search_priority.iterrows():
            lat = row['latitude']
            lon = row['longitude']
            area_id = row.get('area_id', 'Unknown')
            score = row.get('composite_score', 0)
            level = row.get('priority', row.get('priority_level', 'Medium'))
            factors = row.get('factors', 'None')
            
            color = priority_colors.get(level, 'gray')
            
            html = f"<b>Area:</b> {area_id}<br><b>Score:</b> {float(score):.2f}<br><b>Priority:</b> {level}<br><b>Factors:</b> {factors}"
            iframe = folium.IFrame(html, width=200, height=120)
            popup = folium.Popup(iframe, max_width=200)
            
            folium.CircleMarker(
                location=[lat, lon],
                radius=10,
                popup=popup,
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(fg_priority)
        fg_priority.add_to(m)

    # h) Layer Control
    folium.LayerControl().add_to(m)
    
    # i) Legend
    legend_html = '''
     <div style="position: fixed; 
     bottom: 50px; left: 50px; width: 220px; height: 260px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color: white; padding: 10px;">
     <b>Investigation Legend</b><br>
     <i class="glyphicon glyphicon-exclamation-sign" style="color:red"></i> Last Known Location<br>
     <i class="fa fa-circle" style="color:blue"></i> Frequent Areas<br>
     <i class="fa fa-map-marker" style="color:green"></i> Predicted Areas<br>
     <i style="color:orange">—</i> Probable Route<br>
     <i class="fa fa-play fa-rotate-270" style="color:red"></i> Anomalies<br>
     <i class="fa fa-circle" style="color:darkred"></i> Very High Priority<br>
     <i class="fa fa-circle" style="color:orange"></i> High Priority<br>
     <i class="fa fa-circle" style="color:yellow"></i> Medium Priority<br>
     <i class="fa fa-circle" style="color:lightgreen"></i> Low Priority<br>
     </div>
     '''
    m.get_root().html.add_child(folium.Element(legend_html))

    return m

def create_movement_heatmap(gps_df):
    """
    Create a Folium HeatMap of all GPS points.
    """
    if gps_df is None or gps_df.empty:
        logger.warning("No GPS data provided for heatmap.")
        return None
        
    center_lat = gps_df['latitude'].median()
    center_lon = gps_df['longitude'].median()
    
    ESRI_TILES = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles=ESRI_TILES, attr='Esri')
    
    heat_data = gps_df[['latitude', 'longitude']].values.tolist()
    plugins.HeatMap(heat_data, radius=10, blur=15, max_zoom=1).add_to(m)
    
    return m

def create_cluster_map(stay_points_clustered, area_centers):
    """
    Simpler map showing just clusters and area centers
    """
    if stay_points_clustered is None or stay_points_clustered.empty:
        logger.warning("No clustered stay points provided.")
        return None
        
    center_lat = stay_points_clustered['latitude'].median()
    center_lon = stay_points_clustered['longitude'].median()
    
    ESRI_TILES = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}'
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=ESRI_TILES, attr='Esri')
    
    # Generate colors
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'white', 'pink', 'lightblue', 'lightgreen', 'gray', 'black', 'lightgray']
    
    # Plot points
    for _, row in stay_points_clustered.iterrows():
        cluster_id = int(row.get('cluster_id', -1))
        if cluster_id >= 0:
            color = colors[cluster_id % len(colors)]
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=color,
                fill=True
            ).add_to(m)
            
    # Plot centers
    if area_centers is not None and not area_centers.empty:
        for _, row in area_centers.iterrows():
            cluster_id = int(row.get('cluster_id', -1))
            lat = row['latitude']
            lon = row['longitude']
            
            folium.Marker(
                location=[lat, lon],
                popup=f"Cluster {cluster_id} Center",
                icon=folium.Icon(color='black', icon='star')
            ).add_to(m)
            
    return m

def save_investigation_map(map_obj, output_path='reports/investigation_map.html'):
    """
    Save map to HTML file
    """
    if map_obj is None:
        logger.warning(f"Map object is None, cannot save to {output_path}")
        return
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    map_obj.save(output_path)
    logger.info(f"Map saved to {output_path}")

def load_csv_safe(filepath):
    """Safely load a CSV file, returning None if not found."""
    try:
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            logger.warning(f"File not found: {filepath}")
            return None
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None

def normalize_coords(df):
    """Normalize coordinate column names to latitude/longitude."""
    if df is None:
        return None
    df = df.copy()
    rename_map = {}
    
    # Prefer 'lat'/'lon' over 'stay_lat'/'stay_lon'
    has_lat = 'lat' in df.columns
    has_stay_lat = 'stay_lat' in df.columns
    
    if 'latitude' not in df.columns:
        if has_lat:
            rename_map['lat'] = 'latitude'
        elif has_stay_lat:
            rename_map['stay_lat'] = 'latitude'
    
    if 'longitude' not in df.columns:
        if 'lon' in df.columns:
            rename_map['lon'] = 'longitude'
        elif 'stay_lon' in df.columns:
            rename_map['stay_lon'] = 'longitude'
    
    if 'Last_Latitude' in df.columns and 'last_known_lat' not in df.columns:
        rename_map['Last_Latitude'] = 'last_known_lat'
    if 'Last_Longitude' in df.columns and 'last_known_lon' not in df.columns:
        rename_map['Last_Longitude'] = 'last_known_lon'
    if 'Last_Seen_Time' in df.columns and 'last_seen_time' not in df.columns:
        rename_map['Last_Seen_Time'] = 'last_seen_time'
    if 'Case_ID' in df.columns and 'case_id' not in df.columns:
        rename_map['Case_ID'] = 'case_id'
    if 'cluster_label' in df.columns and 'cluster_id' not in df.columns:
        rename_map['cluster_label'] = 'cluster_id'
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def mapping_pipeline():
    """
    Load all required data files and create maps.
    Optimized to avoid loading full 250MB+ files.
    """
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load small data files with column normalization
    logger.info("Loading data files for mapping...")
    area_centers = normalize_coords(load_csv_safe(os.path.join(DATA_DIR, 'area_centers.csv')))
    stay_points_clustered = normalize_coords(load_csv_safe(os.path.join(DATA_DIR, 'stay_points_clustered.csv')))
    search_priority = normalize_coords(load_csv_safe(os.path.join(DATA_DIR, 'search_priority.csv')))
    predicted_routes = normalize_coords(load_csv_safe(os.path.join(DATA_DIR, 'predicted_routes.csv')))
    
    # For anomalies, only load a sample (the full file is 250MB+)
    gps_anomalies = None
    anomaly_path = os.path.join(DATA_DIR, 'gps_anomalies.csv')
    if os.path.exists(anomaly_path):
        try:
            logger.info("Loading anomaly sample (first 50K rows)...")
            df_anom = pd.read_csv(anomaly_path, nrows=50000)
            # Only keep anomalous points for mapping
            if 'ensemble_anomaly' in df_anom.columns:
                gps_anomalies = normalize_coords(df_anom[df_anom['ensemble_anomaly'] == True].head(500))
            elif 'anomaly' in df_anom.columns:
                gps_anomalies = normalize_coords(df_anom[df_anom['anomaly'] == -1].head(500))
            else:
                gps_anomalies = normalize_coords(df_anom.head(100))
            logger.info(f"Loaded {len(gps_anomalies)} anomaly points for mapping.")
        except Exception as e:
            logger.warning(f"Error loading anomalies: {e}")
    
    # Merge area coordinates into search_priority and predicted_areas
    def merge_area_coords(df):
        """Merge area_centers lat/lon into a DataFrame that has area_id."""
        if df is None or area_centers is None:
            return df
        if 'latitude' not in df.columns and 'area_id' in df.columns:
            df = df.merge(
                area_centers[['area_id', 'latitude', 'longitude']],
                on='area_id', how='left'
            )
        return df
    
    search_priority = merge_area_coords(search_priority)
    predicted_areas = None
    if search_priority is not None and area_centers is not None:
        predicted_areas = search_priority.copy()
        if 'latitude' not in predicted_areas.columns and 'area_id' in predicted_areas.columns:
            predicted_areas = predicted_areas.merge(
                area_centers[['area_id', 'latitude', 'longitude']],
                on='area_id', how='left'
            )
        if 'probability' not in predicted_areas.columns:
            predicted_areas['probability'] = predicted_areas.get('composite_score', 50) / 100.0
        if 'priority_level' not in predicted_areas.columns:
            predicted_areas['priority_level'] = predicted_areas.get('priority', 'Medium')
        if 'rank' not in predicted_areas.columns:
            predicted_areas['rank'] = range(1, len(predicted_areas) + 1)
    
    investigation_case = normalize_coords(load_csv_safe('data/synthetic/investigation_case.csv'))
    
    # 1. Investigation Map
    logger.info("Creating comprehensive investigation map...")
    inv_map = create_investigation_map(
        case_data=investigation_case,
        area_centers=area_centers,
        stay_points=stay_points_clustered,
        predicted_areas=predicted_areas,
        predicted_routes=predicted_routes,
        anomaly_points=gps_anomalies,
        search_priority=search_priority
    )
    save_investigation_map(inv_map, os.path.join(REPORTS_DIR, 'investigation_map.html'))
    
    # 2. Movement Heatmap — use stay_points (small) instead of full GPS features (236MB)
    gps_data = stay_points_clustered
    if gps_data is not None:
        logger.info("Creating movement heatmap from stay points...")
        heat_map = create_movement_heatmap(gps_data)
        save_investigation_map(heat_map, os.path.join(REPORTS_DIR, 'movement_heatmap.html'))
        
    # 3. Cluster Map
    if stay_points_clustered is not None and area_centers is not None:
        logger.info("Creating cluster map...")
        cluster_map = create_cluster_map(stay_points_clustered, area_centers)
        save_investigation_map(cluster_map, os.path.join(REPORTS_DIR, 'cluster_map.html'))
        
    logger.info("Mapping pipeline completed successfully.")

if __name__ == "__main__":
    mapping_pipeline()
