"""Generate a reproducible synthetic customer churn dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_customer_data(n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Return a realistic, fully synthetic subscription-customer dataset."""
    if n_samples < 100:
        raise ValueError("n_samples must be at least 100")

    rng = np.random.default_rng(seed)
    contract_type = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_samples,
        p=[0.56, 0.25, 0.19],
    )
    internet_service = rng.choice(
        ["fiber", "dsl", "none"], size=n_samples, p=[0.51, 0.40, 0.09]
    )
    payment_method = rng.choice(
        ["credit-card", "bank-transfer", "electronic-check", "mailed-check"],
        size=n_samples,
        p=[0.32, 0.28, 0.28, 0.12],
    )
    region = rng.choice(
        ["Ontario", "Quebec", "West", "Atlantic"],
        size=n_samples,
        p=[0.42, 0.22, 0.27, 0.09],
    )

    tenure_months = np.clip(rng.gamma(shape=2.1, scale=17.0, size=n_samples), 1, 72).astype(int)
    support_tickets = np.clip(rng.poisson(lam=1.15, size=n_samples), 0, 7)
    late_payments = np.clip(rng.poisson(lam=0.65, size=n_samples), 0, 5)
    paperless_billing = rng.choice(["yes", "no"], size=n_samples, p=[0.69, 0.31])

    service_charge = np.select(
        [internet_service == "fiber", internet_service == "dsl"],
        [42.0, 26.0],
        default=7.0,
    )
    monthly_charges = np.clip(
        31.0 + service_charge + rng.normal(0, 10.5, size=n_samples), 20, 125
    ).round(2)
    avg_monthly_usage_gb = np.clip(
        np.where(internet_service == "fiber", 440, np.where(internet_service == "dsl", 190, 12))
        + rng.normal(0, 85, size=n_samples),
        0,
        900,
    ).round(1)

    logit = (
        -1.85
        + 1.05 * (contract_type == "month-to-month")
        - 0.75 * (contract_type == "two-year")
        - 0.026 * tenure_months
        + 0.31 * support_tickets
        + 0.29 * late_payments
        + 0.35 * (payment_method == "electronic-check")
        + 0.28 * (internet_service == "fiber")
        + 0.010 * (monthly_charges - 70)
    )
    churn_probability = 1.0 / (1.0 + np.exp(-logit))
    churn = rng.binomial(1, churn_probability)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{i:05d}" for i in range(1, n_samples + 1)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "support_tickets": support_tickets,
            "contract_type": contract_type,
            "payment_method": payment_method,
            "paperless_billing": paperless_billing,
            "internet_service": internet_service,
            "avg_monthly_usage_gb": avg_monthly_usage_gb,
            "late_payments": late_payments,
            "region": region,
            "churn": churn,
        }
    )


def save_dataset(output_path: Path, n_samples: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate and save the synthetic dataset."""
    data = generate_customer_data(n_samples=n_samples, seed=seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/customer_churn.csv"))
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    data = save_dataset(args.output, n_samples=args.samples, seed=args.seed)
    print(f"Saved {len(data):,} customers to {args.output}")
    print(f"Synthetic churn rate: {data['churn'].mean():.1%}")


if __name__ == "__main__":
    main()
