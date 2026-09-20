# CASEFILE: An AI-Powered Missing Person Investigation and Probable Location Prediction System Using Advanced Machine Learning

## Abstract

This report presents CASEFILE, an AI-powered investigation-support system designed to analyze historical GPS movement patterns and predict probable geographical areas for fictional missing-person cases. The system integrates multiple advanced machine learning techniques including DBSCAN clustering, K-Means clustering, Isolation Forest anomaly detection, XGBoost classification, and Markov Chain route prediction to generate ranked lists of search-priority areas. All case data is synthetic and fictional. The system is an academic simulation and must not be used for real-world missing person investigations.

## 1. Introduction

When a person goes missing, investigators typically have access to information such as the last known location, time of disappearance, historical movement patterns, and frequently visited locations. Manually analyzing large volumes of movement data is time-consuming and prone to oversight. This project develops an ML-based investigation-support system that automates the analysis of historical movement patterns and generates ranked predictions of probable locations.

The core question addressed: **"Based on the available historical and contextual evidence, which locations should be investigated first?"**

## 2. Problem Statement

The system addresses the challenge of processing and analyzing large-scale GPS trajectory data to:
- Identify frequently visited locations from historical movement data
- Detect normal vs. anomalous movement behavior
- Predict probable geographical areas where a missing person might be found
- Rank locations by search priority to optimize investigation resource allocation

## 3. Objectives

1. Collect and preprocess GPS trajectory data
2. Identify frequently visited geographical locations using clustering
3. Understand normal movement patterns
4. Detect unusual movements using anomaly detection
5. Predict probable geographical areas using supervised ML
6. Predict probable movement routes using Markov Chains
7. Rank locations by composite search-priority score
8. Visualize predictions on interactive maps
9. Provide explanations for ML predictions using SHAP
10. Develop a complete interactive dashboard

## 4. Literature Review

### 4.1 GPS Trajectory Analysis
GPS trajectory mining has been extensively studied in transportation research and urban computing. Zheng et al. (2008) demonstrated that GPS trajectories can reveal meaningful movement patterns and frequently visited locations through spatial clustering.

### 4.2 Clustering for Location Identification
DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is particularly suited for GPS data as it handles arbitrary cluster shapes and identifies noise points. K-Means provides complementary area-level grouping for broader spatial analysis.

### 4.3 Anomaly Detection in Movement Data
Isolation Forest (Liu et al., 2008) efficiently identifies anomalous movements by isolating observations through random partitioning. It is well-suited for high-dimensional GPS features.

### 4.4 Location Prediction
XGBoost (Chen & Guestrin, 2016) has demonstrated strong performance in classification tasks with tabular data. It handles mixed feature types and provides built-in feature importance measures.

### 4.5 Route Prediction
Markov Chain models capture transition probabilities between locations, enabling prediction of probable movement sequences from a given starting location.

## 5. Dataset Description

### 5.1 Synthetic GPS Trajectory Data
- **Source**: Algorithmically generated (based on GeoLife dataset methodology)
- **Coverage**: Beijing, China metropolitan area
- **Users**: 10 simulated individuals
- **Duration**: 30-60 days per user
- **Total Points**: ~900,000 GPS records
- **Features**: user_id, latitude, longitude, altitude, timestamp

### 5.2 Points of Interest (POI) Data
- **Source**: Synthetically generated based on Beijing geography
- **Categories**: Restaurant, hospital, park, school, bus stop, train station, market, residential, office
- **Total POIs**: 51

### 5.3 Synthetic Case Records
- **Total Cases**: 100 training cases + 1 investigation case
- **Fields**: Case_ID, Person_ID, Age_Group, Gender, Last_Latitude, Last_Longitude, Last_Seen_Time, Day, Weather, Usual_Area, Average_Distance, Average_Speed, Previous_Area, Time_Since_Last_Seen, Target_Area

## 6. Data Preprocessing

### 6.1 Data Cleaning Steps
1. **Duplicate Removal**: Exact duplicate GPS records removed
2. **Invalid Coordinate Removal**: Points outside valid GPS ranges filtered
3. **Beijing Bounding Box Filter**: Points outside the study area removed
4. **Speed Outlier Removal**: Points implying speeds >200 km/h removed (GPS jumps)
5. **Missing Value Handling**: NaN coordinates dropped, altitude NaNs filled with 0

### 6.2 Time Feature Extraction
- Hour (0-23), Day of week (0-6), Month, Weekend flag
- Time period: Night (0-5), Morning (6-11), Afternoon (12-17), Evening (18-23)

## 7. Feature Engineering

### 7.1 Point-Level Features
- Distance to previous point (Haversine formula)
- Speed (km/h), acceleration
- Bearing and bearing change

### 7.2 Trajectory-Level Features
- Total distance, duration, average/max speed
- Sinuosity (path curvature)

### 7.3 Stay Points
- Locations where user stayed within 100m for >5 minutes
- Dwell time computed for each stay point

### 7.4 User Profiles
- Home location (most common nighttime stay point)
- Work location (most common weekday daytime stay point)
- Average daily distance, typical active hours

### 7.5 Spatial Binning
- 500m × 500m grid cells for area classification
- Area IDs assigned for ML prediction target

## 8. Methodology

### 8.1 Movement Clustering (Module 4)
**DBSCAN** was applied to stay points with eps=0.15 km (≈0.00135°) and min_samples=3 to identify frequently visited location clusters. The identified clusters were then grouped into broader areas using **K-Means** with the optimal k determined via silhouette analysis.

### 8.2 Anomaly Detection (Module 5)
Three methods were ensembled:
1. **Isolation Forest** (n_estimators=200, contamination=0.05)
2. **Local Outlier Factor** (n_neighbors=20)
3. **One-Class SVM** (RBF kernel, nu=0.05)

A movement is flagged as anomalous if ≥2 of 3 methods agree.

### 8.3 Location Prediction (Module 6)
Four classifiers were compared:
1. **Random Forest** (n_estimators=200, max_depth=15)
2. **XGBoost** (n_estimators=200, max_depth=8, lr=0.1)
3. **Gradient Boosting** (n_estimators=150, max_depth=6)
4. **K-Nearest Neighbors** (k=7)

Features: hour, day, weekend, last coordinates, average speed/distance, previous area, time since last seen, age group, weather.

### 8.4 Route Prediction (Module 7)
A **Markov Chain** transition probability matrix was built from historical area transition sequences. Beam search identifies top-K probable routes from the last known area.

### 8.5 Search Priority Score (Module 8)
Composite scoring with configurable weights:
| Factor | Weight |
|--------|--------|
| ML prediction probability | 30% |
| Historical visit frequency | 20% |
| Route similarity | 15% |
| Distance relevance | 15% |
| Time relevance | 10% |
| Anomaly evidence | 10% |

### 8.6 Explainability (Module 9)
SHAP TreeExplainer generates global and local feature importance explanations for XGBoost predictions.

## 9. Model Training

Models were trained using 80/20 train-test split with stratified sampling. Hyperparameters were selected based on domain knowledge and preliminary experiments.

## 10. Model Evaluation

### 10.1 Classification Metrics
| Model | Accuracy | Precision | Recall | F1 | Top-3 Acc | Top-5 Acc |
|-------|----------|-----------|--------|-----|-----------|-----------|
| Random Forest | - | - | - | - | - | - |
| XGBoost | - | - | - | - | - | - |
| Gradient Boosting | - | - | - | - | - | - |
| KNN | - | - | - | - | - | - |

*(Values to be populated from actual pipeline run)*

### 10.2 Anomaly Detection Evaluation
- Total anomalies detected
- Anomaly rate
- Distribution comparison (normal vs. anomalous)
- Threshold sensitivity analysis

### 10.3 Clustering Evaluation
- Silhouette Score
- Calinski-Harabasz Index
- Davies-Bouldin Index

## 11. Results

### 11.1 Final Case Investigation: MP-2026-017
- **Top Predicted Areas**: *(populated from pipeline)*
- **Detected Anomalies**: *(populated from pipeline)*
- **Probable Route**: *(populated from pipeline)*
- **Search Priority Ranking**: *(populated from pipeline)*

## 12. Explainability

SHAP analysis reveals the key features driving location predictions. Feature importance rankings and individual prediction explanations provide transparency into why specific areas receive high prediction scores.

## 13. Limitations

1. **Synthetic Data**: Results are based on algorithmically generated movement patterns that may not capture the full complexity of real human behavior.
2. **Geographic Scope**: Limited to Beijing, China; patterns may not transfer to other cities.
3. **Temporal Scope**: 30-60 days of data per user may be insufficient for robust pattern learning.
4. **Model Assumptions**: Markov property assumes memoryless transitions; real movement depends on longer history.
5. **Feature Limitations**: Weather, social context, and personal circumstances are simplified.
6. **Scale**: 10 synthetic users is insufficient for population-level generalization.
7. **Anomaly Interpretation**: Anomalies indicate deviation from patterns, not suspicious activity.

## 14. Ethical Considerations

1. **Fictional Data**: All identities, cases, and scenarios are entirely fictional.
2. **No PII**: No personally identifiable information is used.
3. **Probabilistic Nature**: All predictions are probabilistic; they do not prove a person's location.
4. **Non-accusatory**: Anomalous movement is NOT indicative of criminal behavior.
5. **Model Bias**: Synthetic data may embed generation biases; real deployment would require bias auditing.
6. **Not for Real Use**: This system is an academic simulation and must not be used for real-world decisions.
7. **False Positives**: The system will produce false predictions; over-reliance on ML output without human judgment is dangerous.
8. **Privacy**: Real deployment of such systems would raise significant privacy concerns.

## 15. Future Scope

1. Integration with real (anonymized) GPS datasets
2. Deep learning models (LSTMs, Transformers) for sequence prediction
3. Graph Neural Networks for spatial relationship modeling
4. Real-time streaming data processing
5. Multi-modal data fusion (CCTV, phone records, social media)
6. Reinforcement learning for adaptive search strategies
7. Federated learning for privacy-preserving model training
8. Transfer learning across different cities

## 16. Conclusion

This project demonstrates a complete AI-powered investigation-support system that integrates multiple machine learning techniques for analyzing movement patterns and predicting probable locations. The system successfully combines unsupervised learning (DBSCAN, K-Means), anomaly detection (Isolation Forest ensemble), supervised classification (XGBoost), sequence prediction (Markov Chains), and explainability (SHAP) into a unified investigation dashboard.

The composite search-priority scoring mechanism provides investigators with a ranked, explainable list of areas to prioritize, while the interactive map offers spatial context for decision-making. All predictions are presented with probability scores and explanations, maintaining transparency and supporting informed human judgment.

## 17. References

1. Zheng, Y., Li, Q., Chen, Y., Xie, X., & Ma, W. Y. (2008). Understanding mobility based on GPS data. In Proceedings of the 10th international conference on Ubiquitous computing.
2. Liu, F. T., Ting, K. M., & Zhou, Z. H. (2008). Isolation forest. In 2008 eighth IEEE international conference on data mining.
3. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD international conference.
4. Ester, M., Kriegel, H. P., Sander, J., & Xu, X. (1996). A density-based algorithm for discovering clusters in large spatial databases with noise. In KDD.
5. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. In Advances in neural information processing systems.
6. Microsoft Research. GeoLife GPS Trajectory Dataset. https://www.microsoft.com/en-us/research/publication/geolife-gps-trajectory-dataset-user-guide/
