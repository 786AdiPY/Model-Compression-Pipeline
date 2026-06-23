"""Generate synthetic churn dataset."""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_products",
    "support_calls",
    "payment_delay_days",
    "contract_type",   # 0=month, 1=1yr, 2=2yr
    "internet_service", # 0=none, 1=DSL, 2=fiber
    "online_security",  # 0/1
    "tech_support",     # 0/1
]

def generate(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure         = rng.integers(1, 73, n)
    monthly        = rng.uniform(20, 120, n).round(2)
    total          = (tenure * monthly * rng.uniform(0.95, 1.05, n)).round(2)
    num_products   = rng.integers(1, 6, n)
    support_calls  = rng.integers(0, 10, n)
    pay_delay      = rng.integers(0, 31, n)
    contract       = rng.integers(0, 3, n)
    internet       = rng.integers(0, 3, n)
    online_sec     = rng.integers(0, 2, n)
    tech_support   = rng.integers(0, 2, n)

    # Logistic signal
    logit = (
        -3.0
        + 0.03  * (72 - tenure)
        + 0.02  * monthly
        - 0.001 * total
        + 0.1   * support_calls
        + 0.05  * pay_delay
        - 0.5   * contract
        - 0.3   * internet
        - 0.4   * online_sec
        - 0.3   * tech_support
        + rng.normal(0, 0.3, n)
    )
    prob  = 1 / (1 + np.exp(-logit))
    churn = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "tenure_months":      tenure,
        "monthly_charges":    monthly,
        "total_charges":      total,
        "num_products":       num_products,
        "support_calls":      support_calls,
        "payment_delay_days": pay_delay,
        "contract_type":      contract,
        "internet_service":   internet,
        "online_security":    online_sec,
        "tech_support":       tech_support,
        "churn":              churn,
    })
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",    type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out",  type=str, default="data")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = generate(args.n, args.seed)
    train, test = train_test_split(df, test_size=0.2, random_state=args.seed, stratify=df["churn"])

    train.to_csv(os.path.join(args.out, "train.csv"), index=False)
    test.to_csv(os.path.join(args.out, "test.csv"),  index=False)
    print(f"train={len(train)}  test={len(test)}  churn_rate={df['churn'].mean():.2%}")


if __name__ == "__main__":
    main()
