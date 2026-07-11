# FloodSense Streamlit App

Interactive flood-risk early warning interface for the FloodSense challenge.

## What this app includes

- **Home page** with challenge context and safety notes.
- **Data Exploration** page for quick table and chart review (when data file exists).
- **Risk Check** page with district/date/rainfall/soil/water input and:
  - color-coded risk badge (Low/Medium/High/Critical)
  - probability gauge for immediate visual confidence
  - bilingual English/Urdu risk label
  - confidence percent
  - estimated population at risk
  - recommended action per risk level
- **Batch Upload** page for CSV-based multi-row checks.
- **Performance** page with notebook-reported performance summary plus:
  - logistic-regression feature importance chart
  - predicted-risk distribution histogram
  - confusion matrix visualization
  - ROC curve with AUC

## Run locally

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start app:
   - `streamlit run app.py`

## Expected project files

The app supports missing files gracefully, but full functionality needs these files in the project root:

- `flood_event_binary_model.pkl`
- `flood_severity_lr_pipeline.pkl`
- `floodsense_features_engineered.xlsx` (or CSV equivalent)
- optional NDMA inputs: `ndma_flood_impact_2022_features_engineered.xlsx` (also accepts `ndma_flood_impact_2022.csv` and `ndma_flood_impact_2022_FINAL_CLEAN.xlsx`)

## Data and notebook alignment

- Target follows `flood_event` from provided docs/notebooks.
- Risk band mapping follows challenge rules:
  - Low: `<25%`
  - Medium: `25–50%`
  - High: `50–75%`
  - Critical: `>75%`
- App handles missing/invalid input with:
  - `Insufficient data — manual assessment recommended`

## Notes

- `hackathon_notes.txt` and `data_dictionary.txt` are retained as requirement references.
- `train_models.py` and notebooks are available for retraining/saving assets if missing.
