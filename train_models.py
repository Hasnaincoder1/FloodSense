# train_models.py
"""
FloodSense 
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── 1. Load data
print("Loading data...")
features_path = "floodsense_features_engineered.xlsx"
df = pd.read_excel(features_path)
print(f"  Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"  Dropped {before - len(df)} duplicate rows → {len(df)} remaining")

# ── 3. Remove leakage columns 
# These columns are derived from or equal to the target — never use as features
LEAKAGE_COLS = [
    "flood_risk_score",
    "water_area_pct_change_raw",
    "flood_severity",
    "flood_rate_7d",
    "flood_rate_14d",
    "days_since_flood",
]
BINARY_TARGET = "flood_event"
SEVERITY_TARGET = "flood_severity"
SCORE_COL = "flood_risk_score"

# ── 4. Create severity target BEFORE dropping score column 
if SCORE_COL in df.columns:
    df[SEVERITY_TARGET] = pd.cut(
        df[SCORE_COL],
        bins=[-float("inf"), 0.33, 0.66, float("inf")],
        labels=["Low", "Medium", "High"],
    )
    print(f"  Severity distribution:\n{df[SEVERITY_TARGET].value_counts()}")
else:
    print(f"  WARNING: '{SCORE_COL}' not found — severity model will be skipped")
    df[SEVERITY_TARGET] = None

# ── 5. Define feature columns 
exclude = set(LEAKAGE_COLS + [BINARY_TARGET, SEVERITY_TARGET])
feature_cols = [c for c in df.columns if c not in exclude]
print(f"  Using {len(feature_cols)} features: {feature_cols[:8]} ...")

# ── 6. Clean features
X = df[feature_cols].copy()

# Replace inf with NaN then fill with column median
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True))

# Keep only numeric columns
X = X.select_dtypes(include=np.number)
print(f"  Final feature count after numeric filter: {X.shape[1]}")

# ── 7. Binary flood event model 
print("\n── Training binary flood event model ──")
y_bin = df[BINARY_TARGET]

# Verify target
print(f"  Target distribution:\n{y_bin.value_counts()}")
if y_bin.nunique() < 2:
    print("  ERROR: Target has only one class — cannot train. Check your data.")
    exit(1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_bin,
    test_size=0.2,
    random_state=42,
    stratify=y_bin,
)

# Full pipeline: scaler + classifier
# Saving as pipeline preserves feature names for the app
binary_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",   # handles 32/68 imbalance
        C=0.1,                     # slight regularisation
        solver="lbfgs",
        random_state=42,
    )),
])

binary_pipeline.fit(X_train, y_train)

# Evaluate
y_pred = binary_pipeline.predict(X_test)
y_proba = binary_pipeline.predict_proba(X_test)[:, 1]

print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Flood", "Flood"]))

print("  Confusion Matrix (rows=Actual, cols=Predicted):")
cm = confusion_matrix(y_test, y_pred)
print(f"              Predicted No Flood  Predicted Flood")
print(f"  Actual No Flood:    {cm[0][0]:>6}           {cm[0][1]:>6}")
print(f"  Actual Flood:       {cm[1][0]:>6}           {cm[1][1]:>6}")

# Sanity check — model must predict at least some floods
floods_predicted = cm[1][1] + cm[0][1]
if floods_predicted == 0:
    print("\n  WARNING: Model is predicting NO floods at all.")
    print("  Check if flood_event column is correct and balanced.")
else:
    print(f"\n  ✓ Model correctly predicted {cm[1][1]} floods in test set.")

# Save
joblib.dump(binary_pipeline, "flood_event_binary_model.pkl")
print("  Saved → flood_event_binary_model.pkl")

# ── 8. Severity multi-class model 
print("\n── Training severity classification model ──")
y_sev = df[SEVERITY_TARGET]

if y_sev.isna().all() or y_sev.nunique() < 2:
    print("  Skipping severity model — insufficient target data.")
else:
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y_sev.astype(str),
        test_size=0.2,
        random_state=42,
        stratify=y_sev,
    )

    severity_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])

    severity_pipeline.fit(X_train_s, y_train_s)

    print("\n  Severity Classification Report:")
    print(classification_report(y_test_s, severity_pipeline.predict(X_test_s)))

    joblib.dump(severity_pipeline, "flood_severity_lr_pipeline.pkl")
    print("  Saved → flood_severity_lr_pipeline.pkl")

print("\n✓ Training complete.")
print("  Replace the old .pkl files in your FLOOD folder with the new ones.")
print("  Then restart Streamlit: python -m streamlit run app.py")
