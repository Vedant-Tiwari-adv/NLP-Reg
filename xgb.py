# xgb_pipeline_full.py
import numpy as np
import pandas as pd
import time
import warnings
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import xgboost as xgb

warnings.filterwarnings('ignore')

# -----------------------
# Metrics
# -----------------------
def smape(y_true, y_pred):
    return 100.0 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# -----------------------
# Data preparation
# -----------------------
def prepare_features(df):
    df = df.copy()
    if 'price' in df.columns:
        df.drop(columns=['price'], inplace=True)
    assert 'log_price' in df.columns, "DataFrame must contain 'log_price' column"
    
    txt_cols = [c for c in df.columns if c.startswith('txt_pca_')]
    img_cols = [c for c in df.columns if c.startswith('img_pca_')]
    numeric_cols = [c for c in df.columns if c not in txt_cols + img_cols + ['log_price','sample_id']]
    feature_cols = txt_cols + img_cols + numeric_cols

    X = df[feature_cols].astype(np.float32)
    y = df['log_price'].astype(np.float32).values
    return X, y, txt_cols, img_cols, numeric_cols

# -----------------------
# Main pipeline
# -----------------------
def run_xgb_pipeline(df, test_size=0.2, random_state=42, n_iter_search=5, cv=3, n_jobs=1):
    t0 = time.time()
    
    X, y_log, txt_cols, img_cols, numeric_cols = prepare_features(df)
    
    # Scale numeric features if present
    if numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    else:
        scaler = None

    # Train-test split
    X_train, X_test, y_train_log, y_test_log = train_test_split(
        X, y_log, test_size=test_size, random_state=random_state
    )

    # Detect GPU if available
    try:
        n_gpus = xgb.rabit.get_num_workers()
    except:
        n_gpus = 0

    tree_method = 'gpu_hist' if n_gpus > 0 else 'hist'
    predictor = 'gpu_predictor' if n_gpus > 0 else 'cpu_predictor'
    print(f"Using tree_method={tree_method}, predictor={predictor}")

    # XGB Regressor
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method=tree_method,
        predictor=predictor,
        eval_metric='rmse',
        random_state=random_state,
        verbosity=0
    )

    # Hyperparameter search (Randomized)
    param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.8],
        'reg_alpha': [0, 0.1],
        'reg_lambda': [1, 2],
        'min_child_weight': [1, 3]
    }

    rand_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring='neg_mean_squared_error',
        cv=cv,
        verbose=1,
        random_state=random_state,
        n_jobs=n_jobs,
        return_train_score=False
    )

    print("Starting hyperparameter search...")
    rand_search.fit(X_train, y_train_log)
    model = rand_search.best_estimator_
    print("Best params:", rand_search.best_params_)
    print("Best CV score (neg MSE):", rand_search.best_score_)

    # Predict
    preds_log = model.predict(X_test)
    preds_price = np.exp(preds_log)
    targets_price = np.exp(y_test_log)

    # Metrics
    val_smape = smape(targets_price, preds_price)
    val_rmse = rmse(targets_price, preds_price)
    val_r2 = r2_score(targets_price, preds_price)

    print("\nEvaluation on test set (price space):")
    print(f"SMAPE: {val_smape:.4f}")
    print(f"RMSE:  {val_rmse:.4f}")
    print(f"R²:    {val_r2:.4f}")

    # Save CSV for plotting (remove extreme 2.5% tails)
    df_plot = pd.DataFrame({'actual_price': targets_price, 'predicted_price': preds_price})
    lower = df_plot.quantile(0.025)
    upper = df_plot.quantile(0.975)
    df_plot = df_plot[(df_plot.actual_price.between(lower.actual_price, upper.actual_price)) &
                      (df_plot.predicted_price.between(lower.predicted_price, upper.predicted_price))]
    df_plot.to_csv("xgb_plot_data.csv", index=False)
    print("Saved plot data to xgb_plot_data.csv")

    # Save model & artifacts
    artifact = {
        'model': model,
        'scaler_numeric': scaler,
        'txt_cols': txt_cols,
        'img_cols': img_cols,
        'numeric_cols': numeric_cols,
        'feature_columns': X.columns.tolist()
    }
    joblib.dump(artifact, 'xgb_price_model_artifact.joblib')
    print("Saved model & artifacts to xgb_price_model_artifact.joblib")
    print(f"Total time: {time.time() - t0:.1f}s")

    return {
        'model': model,
        'scaler': scaler,
        'metrics': {'SMAPE': val_smape, 'RMSE': val_rmse, 'R2': val_r2},
        'preds_price': preds_price,
        'targets_price': targets_price,
        'feature_columns': X.columns.tolist(),
        'search_cv': rand_search
    }

# -----------------------
# Run if script
# -----------------------
if __name__ == "__main__":
    data_path = r"C:\Personal\Educational\Projects\NLP-Reg\df_shoovitencoded.npz"
    npz = np.load(data_path, allow_pickle=True)
    df = pd.DataFrame({k: npz[k] for k in npz.files})

    results = run_xgb_pipeline(
        df,
        test_size=0.2,
        random_state=42,
        n_iter_search=5,  # small search to save memory & CPU
        cv=3,
        n_jobs=1
    )

    print("Metrics:", results['metrics'])
