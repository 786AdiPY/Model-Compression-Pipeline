"""CLI helper — hits /compare endpoint and pretty-prints side-by-side diff."""
import json
import sys
import urllib.request
import urllib.error

API_BASE = "http://localhost:8000"

SAMPLE = {
    "tenure_months":      24,
    "monthly_charges":    75.5,
    "total_charges":      1812.0,
    "num_products":       2,
    "support_calls":      3,
    "payment_delay_days": 5,
    "contract_type":      1,
    "internet_service":   2,
    "online_security":    0,
    "tech_support":       1,
}


def call_compare(payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{API_BASE}/compare",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def display(resp: dict):
    print("\n=== Input ===")
    print(json.dumps(resp["input"], indent=2))
    print("\n=== Model Comparison ===")
    print(f"{'Version':<30} {'Prob':>8} {'Pred':>6} {'Latency(ms)':>12}")
    print("-" * 60)
    for r in resp["results"]:
        print(f"{r['model_version']:<30} {r['churn_probability']:>8.4f} "
              f"{r['churn_prediction']:>6}  {r['latency_ms']:>10.3f}ms")


if __name__ == "__main__":
    payload = json.loads(sys.argv[1]) if len(sys.argv) > 1 else SAMPLE
    try:
        resp = call_compare(payload)
        display(resp)
    except urllib.error.URLError as e:
        print(f"Error: {e}. Is the server running at {API_BASE}?")
        sys.exit(1)
