"""Train XGBoost churn model, log to MLflow, save pkl artifact."""
import os
import json
import pickle
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, classification_report
)
from sklearn.model_selection import StratifiedKFold

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT   = os.getenv("MLFLOW_EXPERIMENT", "xgb-churn-compression")
DATA_DIR     = os.getenv("DATA_DIR", "/data")
MODEL_OUT    = os.getenv("MODEL_OUT", "/artifacts/model.pkl")

PARAMS = {
    "n_estimators":     int(os.getenv("N_ESTIMATORS",  "300")),
    "max_depth":        int(os.getenv("MAX_DEPTH",      "6")),
    "learning_rate":  float(os.getenv("LEARNING_RATE",  "0.05")),
    "subsample":      float(os.getenv("SUBSAMPLE",      "0.8")),
    "colsample_bytree": float(os.getenv("COLSAMPLE",    "0.8")),
    "use_label_encoder": False,
    "eval_metric":      "logloss",
    "tree_method":      "hist",
    "random_state":     42,
}

FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls", "payment_delay_days",
    "contract_type", "internet_service", "online_security", "tech_support",
]


def load_data():
    train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    test  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    X_tr, y_tr = train[FEATURE_COLS].values, train["churn"].values
    X_te, y_te = test[FEATURE_COLS].values,  test["churn"].values
    return X_tr, y_tr, X_te, y_te


def cross_val_metrics(X, y, params, n_splits=5):
    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, va_idx in skf.split(X, y):
        m = xgb.XGBClassifier(**params)
        m.fit(X[tr_idx], y[tr_idx], eval_set=[(X[va_idx], y[va_idx])], verbose=False)
        aucs.append(roc_auc_score(y[va_idx], m.predict_proba(X[va_idx])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X_tr, y_tr, X_te, y_te = load_data()

    with mlflow.start_run(run_name="xgb-baseline") as run:
        mlflow.log_params({k: v for k, v in PARAMS.items() if k not in ("use_label_encoder",)})
        mlflow.log_param("train_rows", len(X_tr))
        mlflow.log_param("test_rows",  len(X_te))

        cv_mean, cv_std = cross_val_metrics(X_tr, y_tr, PARAMS)
        mlflow.log_metric("cv_auc_mean", cv_mean)
        mlflow.log_metric("cv_auc_std",  cv_std)
        print(f"CV AUC: {cv_mean:.4f} ± {cv_std:.4f}")

        model = xgb.XGBClassifier(**PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=50)

        y_pred      = model.predict(X_te)
        y_prob      = model.predict_proba(X_te)[:, 1]
        acc         = accuracy_score(y_te, y_pred)
        auc         = roc_auc_score(y_te, y_prob)
        f1          = f1_score(y_te, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_auc",      auc)
        mlflow.log_metric("test_f1",       f1)
        print(f"Test  ACC={acc:.4f}  AUC={auc:.4f}  F1={f1:.4f}")
        print(classification_report(y_te, y_pred, target_names=["stay", "churn"]))

        # Save pkl
        os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
        with open(MODEL_OUT, "wb") as f:
            pickle.dump(model, f)
        mlflow.log_artifact(MODEL_OUT, artifact_path="model")

        # Save feature names + baseline distribution for drift
        meta = {
            "feature_cols": FEATURE_COLS,
            "run_id":        run.info.run_id,
            "test_accuracy": acc,
            "test_auc":      auc,
            "test_f1":       f1,
        }
        meta_path = MODEL_OUT.replace(".pkl", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        mlflow.log_artifact(meta_path, artifact_path="model")

        # Save train feature stats for PSI baseline
        stats = pd.DataFrame(X_tr, columns=FEATURE_COLS).describe().to_dict()
        stats_path = MODEL_OUT.replace(".pkl", "_train_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        mlflow.log_artifact(stats_path, artifact_path="model")

        print(f"Run ID: {run.info.run_id}")
        print(f"Model saved: {MODEL_OUT}")


if __name__ == "__main__":
    main()
