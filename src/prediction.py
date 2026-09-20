import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from xgboost import XGBClassifier

def prepare_prediction_data(cases_path='data/synthetic/cases.csv'):
    """
    Load and prepare data for prediction models.
    """
    df = pd.read_csv(cases_path)
    
    # Extract hour from Last_Seen_Time (format: "HH:MM")
    if 'Last_Seen_Time' in df.columns:
        try:
            df['hour'] = df['Last_Seen_Time'].astype(str).str.split(':').str[0].astype(int)
        except Exception:
            df['hour'] = 12  # fallback
    else:
        df['hour'] = 12
        
    features = ['Age_Group', 'Gender', 'Day', 'Weather', 'hour', 
                'Last_Latitude', 'Last_Longitude', 'Average_Distance', 
                'Average_Speed', 'Time_Since_Last_Seen']
    
    # Ensure all required features are present
    for col in features:
        if col not in df.columns:
            if col in ['Average_Distance', 'Average_Speed', 'Time_Since_Last_Seen']:
                df[col] = 0.0
            elif col in ['Last_Latitude', 'Last_Longitude']:
                df[col] = 39.9
            else:
                df[col] = 'Unknown'
                
    X = df[features].copy()
    y = df['Target_Area'].copy()
    
    label_encoders = {}
    
    # Label encode categorical features (handle StringDtype in Python 3.14)
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
            
    # Encode target
    le_y = LabelEncoder()
    y_encoded = le_y.fit_transform(y.astype(str))
    label_encoders['Target_Area'] = le_y
    
    # Handle stratification - if too few samples per class, don't stratify
    unique, counts = np.unique(y_encoded, return_counts=True)
    min_count = counts.min()
    stratify_param = y_encoded if min_count >= 2 else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, stratify=stratify_param, random_state=42
    )
    
    return X_train, X_test, y_train, y_test, label_encoders, features

def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train):
    model = XGBClassifier(n_estimators=200, max_depth=8, learning_rate=0.1, 
                          random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    model = GradientBoostingClassifier(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train):
    model = KNeighborsClassifier(n_neighbors=7)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # Calculate top-k accuracy
    classes = model.classes_
    top_1_acc = 0
    top_3_acc = 0
    top_5_acc = 0
    
    for i in range(len(y_test)):
        true_class = y_test[i]
        probs = y_prob[i]
        top_k_indices = np.argsort(probs)[::-1]
        
        if len(top_k_indices) > 0 and top_k_indices[0] == true_class:
            top_1_acc += 1
        if true_class in top_k_indices[:3]:
            top_3_acc += 1
        if true_class in top_k_indices[:5]:
            top_5_acc += 1
            
    n = len(y_test)
    
    return {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'top_1_accuracy': top_1_acc / n,
        'top_3_accuracy': top_3_acc / n,
        'top_5_accuracy': top_5_acc / n,
        'y_pred': y_pred
    }

def compare_models(results_dict):
    metrics = []
    for name, res in results_dict.items():
        metrics.append({
            'Model': name,
            'Accuracy': res['accuracy'],
            'Precision': res['precision'],
            'Recall': res['recall'],
            'F1': res['f1'],
            'Top-1 Acc': res['top_1_accuracy'],
            'Top-3 Acc': res['top_3_accuracy'],
            'Top-5 Acc': res['top_5_accuracy']
        })
        
    df = pd.DataFrame(metrics)
    
    print("\n--- Model Comparison ---")
    print(df.to_string(index=False))
    
    os.makedirs('reports', exist_ok=True)
    df.to_csv('reports/model_comparison.csv', index=False)
    
    best_model = df.loc[df['F1'].idxmax()]['Model']
    return best_model

def plot_confusion_matrix(y_test, y_pred, model_name, class_names=None):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig(f'reports/confusion_matrix_{model_name}.png')
    plt.close()

def plot_feature_importance(model, feature_names, model_name):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title(f'Feature Importances - {model_name}')
        plt.bar(range(len(importances)), importances[indices], align='center')
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.tight_layout()
        
        os.makedirs('reports', exist_ok=True)
        plt.savefig(f'reports/feature_importance_{model_name}.png')
        plt.close()

def predict_location(model, case_features, label_encoders, top_n=5):
    features_list = ['Age_Group', 'Gender', 'Day', 'Weather', 'hour', 
                'Last_Latitude', 'Last_Longitude', 'Average_Distance', 
                'Average_Speed', 'Time_Since_Last_Seen']
                
    X_input = []
    for f in features_list:
        val = case_features.get(f, 0)
        if f in label_encoders and f != 'Target_Area':
            try:
                if str(val) in label_encoders[f].classes_:
                    val = label_encoders[f].transform([str(val)])[0]
                else:
                    val = 0
            except:
                val = 0
        X_input.append(val)
        
    X_input = np.array(X_input).reshape(1, -1)
    
    probs = model.predict_proba(X_input)[0]
    top_indices = np.argsort(probs)[::-1][:top_n]
    
    target_encoder = label_encoders.get('Target_Area')
    
    results = []
    for i, idx in enumerate(top_indices):
        prob = probs[idx]
        area_name = target_encoder.inverse_transform([idx])[0] if target_encoder else str(idx)
        
        if prob > 0.30:
            priority = 'Very High'
        elif prob > 0.20:
            priority = 'High'
        elif prob > 0.10:
            priority = 'Medium'
        else:
            priority = 'Low'
            
        results.append({
            'area': area_name,
            'probability': float(prob),
            'rank': i + 1,
            'priority': priority
        })
        
    return results

def prediction_pipeline():
    print("Starting location prediction pipeline...")
    
    try:
        X_train, X_test, y_train, y_test, encoders, feature_names = prepare_prediction_data('data/synthetic/cases.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return
        
    print("Training models...")
    models = {
        'RandomForest': train_random_forest(X_train, y_train),
        'XGBoost': train_xgboost(X_train, y_train),
        'GradientBoosting': train_gradient_boosting(X_train, y_train),
        'KNN': train_knn(X_train, y_train)
    }
    
    print("Evaluating models...")
    results = {}
    for name, model in models.items():
        res = evaluate_model(model, X_test, y_test, name)
        results[name] = res
        
    best_name = compare_models(results)
    best_model = models[best_name]
    print(f"\nBest model selected: {best_name}")
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/location_model.pkl')
    joblib.dump(encoders, 'models/prediction_encoders.pkl')
    print("Saved best model to models/location_model.pkl")
    
    target_encoder = encoders.get('Target_Area')
    class_names = target_encoder.classes_ if target_encoder else None
    
    plot_confusion_matrix(y_test, results[best_name]['y_pred'], best_name, class_names)
    plot_feature_importance(best_model, feature_names, best_name)
    print("Generated evaluation plots in reports/")
    
    print("Pipeline completed successfully.")

if __name__ == "__main__":
    prediction_pipeline()
