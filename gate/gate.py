"""Quality gate: block deploy if accuracy drop vs pkl baseline exceeds threshold."""
import os
import sys
import json

RESULTS_PATH    = os.getenv("RESULTS_PATH",    "artifacts/benchmark_results.json")
MAX_ACC_DROP    = float(os.getenv("MAX_ACC_DROP",    "0.01"))   # 1 %
MAX_AUC_DROP    = float(os.getenv("MAX_AUC_DROP",    "0.01"))   # 1 %
GATE_REPORT_OUT = os.getenv("GATE_REPORT_OUT", "artifacts/gate_report.json")


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def run_gate(results: list) -> dict:
    baseline = next((r for r in results if r["model"] == "pkl_xgboost"), None)
    if baseline is None:
        return {"passed": False, "reason": "pkl baseline not found in results"}

    report = {"baseline": baseline["model"], "checks": [], "passed": True}

    for r in results:
        if r["model"] == baseline["model"]:
            continue

        acc_drop = baseline["accuracy"] - r["accuracy"]
        auc_drop = baseline["auc"]      - r["auc"]
        passed   = acc_drop <= MAX_ACC_DROP and auc_drop <= MAX_AUC_DROP

        check = {
            "model":        r["model"],
            "acc_drop":     round(acc_drop, 4),
            "auc_drop":     round(auc_drop, 4),
            "acc_threshold": MAX_ACC_DROP,
            "auc_threshold": MAX_AUC_DROP,
            "passed":        passed,
            "latency_ms":   r["latency_ms"],
            "speedup":      r.get("speedup_vs_pkl", 1.0),
        }
        report["checks"].append(check)

        if not passed:
            report["passed"] = False
            check["fail_reason"] = []
            if acc_drop > MAX_ACC_DROP:
                check["fail_reason"].append(
                    f"accuracy drop {acc_drop:.4f} > threshold {MAX_ACC_DROP}"
                )
            if auc_drop > MAX_AUC_DROP:
                check["fail_reason"].append(
                    f"AUC drop {auc_drop:.4f} > threshold {MAX_AUC_DROP}"
                )

    return report


def main():
    results = load_results()
    report  = run_gate(results)

    os.makedirs(os.path.dirname(GATE_REPORT_OUT), exist_ok=True)
    with open(GATE_REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== Quality Gate Report ===")
    print(json.dumps(report, indent=2))

    if report["passed"]:
        print("\nGATE PASSED — all compressed models within accuracy thresholds.")
        sys.exit(0)
    else:
        failed = [c["model"] for c in report["checks"] if not c["passed"]]
        print(f"\nGATE FAILED — models failing gate: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
