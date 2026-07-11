# train_models.py
"""
Training script for FloodSense project.
- Loads engineered features from `floodsense_features_engineered.xlsx`.
- Trains a binary classifier for flood event detection.
- Trains a multi-class Logistic Regression pipeline for flood severity.
- Serializes models to `flood_event_binary_model.pkl` and `flood_severity_lr_pipeline.pkl` using joblib.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Load engineered features
features_path = "floodsense_features_engineered.xlsx"
df = pd.read_excel(features_path)

# Assume the engineered dataset includes the following columns (based on notebooks):
# - target_flood_event (binary 0/1)
# - flood_risk_score (continuous) which we will bin into Low/Medium/High for severity classification
# - feature columns used for modeling (e.g., precip_3day_avg, soil_moisture_trend, water_area_pct_change, etc.)

# Binary classification target
binary_target = "flood_event"

# Severity target creation (based on flood_risk_score)
severity_target = "flood_severity"
# Binning thresholds (example: 0-0.33 Low, 0.33-0.66 Medium, >0.66 High)
score_col = "flood_risk_score"
df[severity_target] = pd.cut(
    df[score_col],
    bins=[-float('inf'), 0.33, 0.66, float('inf')],
    labels=["Low", "Medium", "High"]
)

# Feature columns (exclude target columns and raw score column)
exclude_cols = [binary_target, severity_target, score_col]
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols]

# ---- Binary Flood Event Model ----
X_bin = X.copy()
y_bin = df[binary_target]
X_train, X_test, y_train, y_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin)

binary_clf = LogisticRegression(max_iter=500, class_weight='balanced')
binary_clf.fit(X_train, y_train)

# Save binary model
joblib.dump(binary_clf, "flood_event_binary_model.pkl")

print("Binary flood event model trained and saved to flood_event_binary_model.pkl")
print("Binary classification report:\n", classification_report(y_test, binary_clf.predict(X_test)))

# ---- Flood Severity Multi-class Model ----
X_sev = X.copy()
y_sev = df[severity_target]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_sev, y_sev, test_size=0.2, random_state=42, stratify=y_sev)

severity_pipeline = Pipeline([ ("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=500, class_weight='balanced')) ])

severity_pipeline.fit(X_train_s, y_train_s)

# Save severity pipeline
joblib.dump(severity_pipeline, "flood_severity_lr_pipeline.pkl")

print("Severity logistic regression pipeline trained and saved to flood_severity_lr_pipeline.pkl")
print("Severity classification report:\n", classification_report(y_test_s, severity_pipeline.predict(X_test_s)))
