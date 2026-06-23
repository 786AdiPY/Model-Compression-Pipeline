"""Push best compressed model to MLflow Model Registry after gate passes."""
import os
import sys
import json
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI    = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT      = os.getenv("MLFLOW_EXPERIMENT",   "xgb-churn-compression")
RESULTS_PATH    = os.getenv("RESULTS_PATH",         "/artifacts/benchmark_results.json")
GATE_REPORT     = os.getenv("GATE_REPORT",          "/artifacts/gate_report.json")
MODEL_PKL       = os.getenv("MODEL_PKL",            "/artifacts/model.pkl")
MODEL_ONNX      = os.getenv("MODEL_ONNX",           "/artifacts/model_fp32.onnx")
MODEL_TRT       = os.getenv("MODEL_TRT",            "/artifacts/model_int8.trt")
REGISTRY_NAME   = os.getenv("REGISTRY_NAME",        "churn-detector")

PRIORITY = ["trt_int8", "onnx_fp32", "pkl_xgboost", "trt_int8_fallback_onnx"]


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pick_best(results: list, gate: dict) -> dict:
    """Pick fastest model that passed the gate."""
    passed_names = {c["model"] for c in gate.get("checks", []) if c["passed"]}
    passed_names.add(gate.get("baseline", ""))  # baseline always passes

    for priority_model in PRIORITY:
        for r in results:
            if r["model"] == priority_model and r["model"] in passed_names:
                return r
    # Fallback: pick baseline
    return next(r for r in results if r["model"] == gate.get("baseline"))


def register_model(best: dict, results: list, gate: dict):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()

    # Find the latest run in experiment
    exp = client.get_experiment_by_name(EXPERIMENT)
    if exp is None:
        print(f"Experiment '{EXPERIMENT}' not found.")
        sys.exit(1)

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print("No runs found.")
        sys.exit(1)

    run_id = runs[0].info.run_id

    with mlflow.start_run(run_id=run_id):
        # Log final benchmark + gate artifacts
        mlflow.log_artifact(RESULTS_PATH, artifact_path="benchmark")
        mlflow.log_artifact(GATE_REPORT,  artifact_path="benchmark")

        # Log compression metadata
        mlflow.log_metric("best_model_latency_ms", best["latency_ms"])
        mlflow.log_metric("best_model_speedup",    best.get("speedup_vs_pkl", 1.0))
        mlflow.log_metric("best_model_accuracy",   best["accuracy"])
        mlflow.log_metric("best_model_auc",        best["auc"])
        mlflow.set_tag("best_compressed_model", best["model"])

    # Register model artifact
    artifact_uri = f"runs:/{run_id}/model"

    model_version = mlflow.register_model(
        model_uri=artifact_uri,
        name=REGISTRY_NAME,
    )

    # Add description tags
    client.update_model_version(
        name=REGISTRY_NAME,
        version=model_version.version,
        description=(
            f"Best compressed model: {best['model']}. "
            f"Latency={best['latency_ms']:.2f}ms, "
            f"Accuracy={best['accuracy']:.4f}, "
            f"Speedup={best.get('speedup_vs_pkl', 1.0):.2f}x vs pkl."
        ),
    )

    # Transition to Production
    client.transition_model_version_stage(
        name=REGISTRY_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True,
    )

    print(f"\nRegistered '{REGISTRY_NAME}' v{model_version.version} → Production")
    print(f"Best model: {best['model']}")
    print(f"Run ID: {run_id}")


def main():
    results = load_json(RESULTS_PATH)
    gate    = load_json(GATE_REPORT)

    if not gate.get("passed"):
        print("Gate did not pass — aborting registry push.")
        sys.exit(1)

    best = pick_best(results, gate)
    print(f"Selected best model: {best['model']}  "
          f"(latency={best['latency_ms']}ms, acc={best['accuracy']})")

    register_model(best, results, gate)


if __name__ == "__main__":
    main()
