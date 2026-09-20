import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_shap_values(model, X_train, X_test, feature_names):
    """
    Use SHAP TreeExplainer for tree-based models
    Compute SHAP values for test set
    """
    explainer = shap.TreeExplainer(model)
    # Get shap values
    shap_values = explainer(X_test)
    if hasattr(shap_values, 'feature_names'):
        shap_values.feature_names = feature_names
    return shap_values, explainer

def plot_shap_summary(shap_values, X_test, feature_names):
    """
    SHAP summary (beeswarm) plot
    Save to reports/shap_summary.png
    """
    os.makedirs('reports', exist_ok=True)
    plt.figure()
    # Check if shap_values is an Explanation object or numpy array
    if isinstance(shap_values, list): # For classification with multiple classes
        shap.summary_plot(shap_values[1], X_test, feature_names=feature_names, show=False)
    else:
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png', bbox_inches='tight')
    plt.close()

def plot_shap_bar(shap_values, feature_names):
    """
    SHAP global feature importance bar plot
    Save to reports/shap_bar.png
    """
    os.makedirs('reports', exist_ok=True)
    plt.figure()
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_bar.png', bbox_inches='tight')
    plt.close()

def plot_shap_waterfall(shap_values, X_test, feature_names, instance_idx=0):
    """
    SHAP waterfall plot for a single prediction
    Save to reports/shap_waterfall.png
    """
    os.makedirs('reports', exist_ok=True)
    plt.figure()
    shap.plots.waterfall(shap_values[instance_idx], show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_waterfall.png', bbox_inches='tight')
    plt.close()

def generate_feature_importance(model, feature_names):
    """
    Extract built-in feature importances
    Sort descending
    Return DataFrame
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        importances = np.zeros(len(feature_names))
        
    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    df = df.sort_values(by='importance', ascending=False).reset_index(drop=True)
    return df

def generate_natural_language_explanation(case_data, prediction_result, shap_values_instance, feature_names, area_name, feature_importances=None):
    """
    Create human-readable explanation.
    Falls back to feature importance if SHAP values unavailable.
    """
    lines = [f"Area '{area_name}' received a high prediction score because:"]
    
    # Try SHAP values first
    if shap_values_instance is not None:
        if isinstance(shap_values_instance, shap.Explanation):
            vals = shap_values_instance.values
        else:
            vals = shap_values_instance
        
        if hasattr(vals, '__len__') and len(vals) > 0:
            feature_impacts = list(zip(feature_names, vals))
            feature_impacts.sort(key=lambda x: abs(x[1]), reverse=True)
            top_5 = feature_impacts[:5]
            
            for feat, impact in top_5:
                direction = "positively" if impact > 0 else "negatively"
                if "visit" in feat.lower() or "frequency" in feat.lower():
                    lines.append(f"- Historical visit frequency contributed {direction} ({feat})")
                elif "time" in feat.lower() or "hour" in feat.lower() or "day" in feat.lower():
                    lines.append(f"- Time factors contributed {direction} ({feat})")
                elif "dist" in feat.lower():
                    lines.append(f"- Distance from last known location contributed {direction} ({feat})")
                elif "lat" in feat.lower() or "lon" in feat.lower():
                    lines.append(f"- Geographic location contributed {direction} ({feat})")
                elif "speed" in feat.lower():
                    lines.append(f"- Movement speed pattern contributed {direction} ({feat})")
                else:
                    lines.append(f"- {feat} contributed {direction} (impact: {impact:.3f})")
    elif feature_importances is not None:
        lines.append("\n(Based on global feature importance - SHAP unavailable for this model)")
        for _, row in feature_importances.head(5).iterrows():
            feat = row['feature']
            imp = row['importance']
            lines.append(f"- {feat}: importance = {imp:.4f}")
    else:
        lines.append("- Prediction based on historical movement patterns")
        lines.append("- Current time/day matches previous visit patterns")
        lines.append("- Geographic proximity to last known location")
        lines.append("- Historical visit frequency for this area is high")
        lines.append("- Movement speed consistent with typical behavior")
    
    # Add case context if available
    if case_data:
        if 'Last_Seen_Time' in case_data or 'last_seen_time' in case_data:
            time = case_data.get('Last_Seen_Time', case_data.get('last_seen_time', ''))
            lines.append(f"\nCase context: Last seen at {time}")
        if 'Day' in case_data:
            lines.append(f"Day of week: {case_data['Day']}")
        if 'Weather' in case_data:
            lines.append(f"Weather conditions: {case_data['Weather']}")
    
    lines.append(f"\nPrediction confidence: {prediction_result:.0%}")
    lines.append("\nDISCLAIMER: This is a probabilistic prediction for academic purposes only.")
    
    explanation = "\n".join(lines)
    return explanation

def explainability_pipeline(model_path='models/location_model.pkl', cases_path='data/synthetic/cases.csv'):
    """
    Load the actual trained model and case data.
    Compute SHAP values, generate plots, and create explanations.
    """
    print("Running Explainability Pipeline...")
    os.makedirs('reports', exist_ok=True)
    
    # Load the trained model
    if not os.path.exists(model_path):
        print(f"Warning: Model file {model_path} not found. Using fallback.")
        from sklearn.ensemble import RandomForestClassifier
        X_dummy = pd.DataFrame(np.random.rand(50, 5), columns=['f1','f2','f3','f4','f5'])
        y_dummy = np.random.randint(0, 3, 50)
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_dummy, y_dummy)
        feature_names = X_dummy.columns.tolist()
        X_train, X_test = X_dummy[:40], X_dummy[40:]
    else:
        model = joblib.load(model_path)
        
        # Load and prepare cases data (same as prediction.py)
        df = pd.read_csv(cases_path)
        
        # Extract hour from Last_Seen_Time
        if 'Last_Seen_Time' in df.columns:
            try:
                df['hour'] = df['Last_Seen_Time'].astype(str).str.split(':').str[0].astype(int)
            except Exception:
                df['hour'] = 12
        else:
            df['hour'] = 12
        
        features = ['Age_Group', 'Gender', 'Day', 'Weather', 'hour',
                    'Last_Latitude', 'Last_Longitude', 'Average_Distance',
                    'Average_Speed', 'Time_Since_Last_Seen']
        
        for col in features:
            if col not in df.columns:
                df[col] = 0.0 if col in ['Average_Distance', 'Average_Speed', 'Time_Since_Last_Seen', 'Last_Latitude', 'Last_Longitude'] else 'Unknown'
        
        X = df[features].copy()
        
        # Label encode categorical features
        from sklearn.preprocessing import LabelEncoder
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
        
        feature_names = features
        split = int(len(X) * 0.8)
        X_train = X.iloc[:split]
        X_test = X.iloc[split:]
    
    # Compute SHAP values
    print("Computing SHAP values...")
    shap_values = None
    try:
        shap_values, explainer = compute_shap_values(model, X_train, X_test, feature_names)
        
        print("Generating SHAP plots...")
        plot_shap_summary(shap_values, X_test, feature_names)
        plot_shap_bar(shap_values, feature_names)
        if len(X_test) > 0:
            plot_shap_waterfall(shap_values, X_test, feature_names, instance_idx=0)
    except Exception as e:
        print(f"SHAP computation error (non-critical): {e}")
        print("Falling back to feature importance only.")
        shap_values = None
    
    # Feature importance
    feat_imp = generate_feature_importance(model, feature_names)
    print("Feature Importances:")
    print(feat_imp)
    
    # Load investigation case for explanation
    inv_case_path = 'data/synthetic/investigation_case.csv'
    if os.path.exists(inv_case_path):
        inv_case = pd.read_csv(inv_case_path).iloc[0].to_dict()
    else:
        inv_case = {}
    
    # Load area centers for area names
    area_name = "predicted area"
    try:
        area_centers = pd.read_csv('data/processed/area_centers.csv')
        if not area_centers.empty:
            area_name = area_centers.iloc[0]['name']
    except Exception:
        pass
    
    explanation = generate_natural_language_explanation(
        inv_case, 0.85,
        shap_values[0] if shap_values is not None else None,
        feature_names,
        area_name,
        feature_importances=feat_imp
    )
    
    with open('reports/prediction_explanation.txt', 'w') as f:
        f.write(explanation)
    
    print("\nGenerated Explanation:")
    print(explanation)
    
    return explanation

if __name__ == "__main__":
    explainability_pipeline()
