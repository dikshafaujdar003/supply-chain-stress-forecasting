import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import shap
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from loguru import logger
from pathlib import Path

FEATURE_COLS = [
    "rolling_vol_30", "vol_spike", "cum_return_20d", "drawdown",
    "price_below_50ma", "price_below_200ma",
    "supply_chain_pressure", "shipping_costs", "import_prices",
    "pmi_manufacturing", "producer_prices",
    "avg_sentiment", "negative_ratio", "article_count",
    "betweenness", "pagerank", "in_degree", "out_degree", "tier",
]

MODEL_PATH = Path("data") / "processed" / "stress_model.pkl"

def train_stress_classifier(df: pd.DataFrame):
    mlflow.set_experiment("supply_chain_stress")
    available = [c for c in FEATURE_COLS if c in df.columns]
    X, y = df[available].fillna(0), df["stress_label"]

    tscv = TimeSeriesSplit(n_splits=5)
    candidates = {
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05,
                                  max_depth=6, eval_metric="logloss", random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200,
                                                        learning_rate=0.05,
                                                        max_depth=5, random_state=42),
    }

    best_model, best_auc = None, 0
    for name, model in candidates.items():
        with mlflow.start_run(run_name=name):
            aucs = []
            for train_idx, val_idx in tscv.split(X):
                scaler = StandardScaler()
                X_tr  = scaler.fit_transform(X.iloc[train_idx])
                X_val = scaler.transform(X.iloc[val_idx])
                model.fit(X_tr, y.iloc[train_idx])
                auc = roc_auc_score(y.iloc[val_idx], model.predict_proba(X_val)[:, 1])
                aucs.append(auc)

            mean_auc = np.mean(aucs)
            mlflow.log_metric("mean_cv_auc", mean_auc)
            mlflow.log_param("model", name)

            scaler_final = StandardScaler()
            X_all = scaler_final.fit_transform(X)
            model.fit(X_all, y)
            mlflow.sklearn.log_model(model, "model")
            logger.info(f"{name} AUC: {mean_auc:.4f}")

            if mean_auc > best_auc:
                best_auc  = mean_auc
                best_model = (model, scaler_final, name)

    model, scaler, name = best_model
    explainer   = shap.Explainer(model, scaler.transform(X))
    shap_values = explainer(scaler.transform(X))

    joblib.dump({"model": model, "scaler": scaler, "features": list(X.columns)},
                MODEL_PATH)
    logger.info(f"Saved: {name} (AUC={best_auc:.4f})")
    return model, scaler, shap_values

def predict_current_stress(df: pd.DataFrame) -> pd.DataFrame:
    bundle   = joblib.load(MODEL_PATH)
    model, scaler, features = bundle["model"], bundle["scaler"], bundle["features"]
    latest   = df.groupby("ticker").last().reset_index()
    available = [c for c in features if c in latest.columns]
    X        = scaler.transform(latest[available].fillna(0))
    latest["stress_probability"] = model.predict_proba(X)[:, 1]
    latest["stress_label_pred"]  = model.predict(X)
    return latest[["ticker", "stress_probability", "stress_label_pred"]]