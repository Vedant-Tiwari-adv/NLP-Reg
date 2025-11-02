import joblib
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

# -----------------------
# Load artifact
# -----------------------
artifact_path = "xgb_price_model_artifact.joblib"
artifact = joblib.load(artifact_path)

model = artifact['model']
feature_cols = artifact['feature_columns']

# -----------------------
# Extract feature importance
# -----------------------
# XGBoost stores importance by feature index; map it back to names
booster = model.get_booster()
importance_dict = booster.get_score(importance_type='gain')  # can use 'weight', 'gain', or 'cover'

# Map to feature names in original order
importance_df = pd.DataFrame([
    {'feature': feature, 'importance': importance_dict.get(feature, 0.0)}
    for feature in feature_cols
])

# Sort descending
importance_df = importance_df.sort_values(by='importance', ascending=False).reset_index(drop=True)

# -----------------------
# Plot
# -----------------------
plt.figure(figsize=(10, 8))
plt.barh(importance_df['feature'][:30][::-1], importance_df['importance'][:30][::-1])
plt.xlabel("Feature Importance (Gain)")
plt.ylabel("Feature")
plt.title("Top 30 Feature Importances — XGBoost Model")
plt.tight_layout()
plt.show()

# Optional: Save to CSV
importance_df.to_csv("xgb_feature_importance.csv", index=False)
print("Feature importances saved to xgb_feature_importance.csv")
