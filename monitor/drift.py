"""Feature drift detection using Population Stability Index (PSI).

Compares incoming prediction-time features against train distribution.
Runs as a scheduled job or triggered post-deploy.
"""
import os
import json
import numpy as np
import pandas as pd

TRAIN_STATS_PATH = os.getenv("TRAIN_STATS_PATH", "artifacts/model_train_stats.json")
INCOMING_CSV     = os.getenv("INCOMING_CSV",     "data/test.csv")
DRIFT_REPORT_OUT = os.getenv("DRIFT_REPORT_OUT", "artifacts/drift_report.json")
PSI_WARN         = float(os.getenv("PSI_WARN",  "0.1"))   # moderate drift
PSI_ALERT        = float(os.getenv("PSI_ALERT", "0.2"))   # severe drift
N_BINS           = int(os.getenv("N_BINS", "10"))

FEATURE_COLS = [
    "tenure_months", "monthly_charges", "total_charges",
    "num_products", "support_calls", "payment_delay_days",
    "contract_type", "internet_service", "online_security", "tech_support",
]


def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """PSI = sum((actual% - expected%) * ln(actual% / expected%))"""
    breakpoints = np.percentile(expected, np.linspace(0, 100, n_bins + 1))
    breakpoints  = np.unique(breakpoints)  # remove duplicates for low-cardinality features

    exp_counts  = np.histogram(expected, bins=breakpoints)[0]
    act_counts  = np.histogram(actual,   bins=breakpoints)[0]

    exp_pct = exp_counts / len(expected)
    act_pct = act_counts / len(actual)

    # Clip to avoid log(0)
    exp_pct = np.clip(exp_pct, 1e-6, None)
    act_pct = np.clip(act_pct, 1e-6, None)

    psi = np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))
    return float(psi)


def load_train_distribution():
    """Load saved train feature stats from training step."""
    if not os.path.exists(TRAIN_STATS_PATH):
        raise FileNotFoundError(f"Train stats not found: {TRAIN_STATS_PATH}")
    with open(TRAIN_STATS_PATH) as f:
        stats = json.load(f)
    return stats


def reconstruct_train_sample(stats: dict, n: int = 5000) -> pd.DataFrame:
    """Approximate train distribution from saved describe() stats using normal sampling."""
    rng  = np.random.default_rng(0)
    rows = {}
    for col in FEATURE_COLS:
        if col in stats:
            mean = stats[col].get("mean", 0)
            std  = stats[col].get("std",  1)
            rows[col] = rng.normal(mean, std, n)
    return pd.DataFrame(rows)


def main():
    incoming = pd.read_csv(INCOMING_CSV)
    stats    = load_train_distribution()
    train_df = reconstruct_train_sample(stats)

    report   = {"features": {}, "summary": {}}
    psi_vals = []

    for col in FEATURE_COLS:
        if col not in incoming.columns or col not in train_df.columns:
            continue

        psi = compute_psi(
            train_df[col].values,
            incoming[col].values,
            n_bins=N_BINS,
        )
        psi_vals.append(psi)

        status = "ok"
        if psi >= PSI_ALERT:
            status = "alert"
        elif psi >= PSI_WARN:
            status = "warn"

        report["features"][col] = {
            "psi":    round(psi, 4),
            "status": status,
        }

    overall_psi = float(np.mean(psi_vals)) if psi_vals else 0.0
    alert_cols  = [c for c, v in report["features"].items() if v["status"] == "alert"]
    warn_cols   = [c for c, v in report["features"].items() if v["status"] == "warn"]

    report["summary"] = {
        "mean_psi":       round(overall_psi, 4),
        "alert_features": alert_cols,
        "warn_features":  warn_cols,
        "overall_status": "alert" if alert_cols else ("warn" if warn_cols else "ok"),
        "n_incoming":     len(incoming),
    }

    os.makedirs(os.path.dirname(DRIFT_REPORT_OUT), exist_ok=True)
    with open(DRIFT_REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))

    status = report["summary"]["overall_status"]
    print(f"\nDrift status: {status.upper()}")
    if alert_cols:
        print(f"ALERT features: {alert_cols}")
    if warn_cols:
        print(f"WARN  features: {warn_cols}")


if __name__ == "__main__":
    main()
