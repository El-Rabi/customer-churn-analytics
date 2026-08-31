"""Run business-facing SQL summaries over the customer dataset."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


QUERIES = {
    "churn_by_contract": """
        SELECT contract_type, COUNT(*) AS customers,
               ROUND(AVG(churn), 4) AS churn_rate,
               ROUND(AVG(monthly_charges), 2) AS avg_monthly_charges
        FROM customer_churn
        GROUP BY contract_type
        ORDER BY churn_rate DESC
    """,
    "churn_by_support_demand": """
        SELECT CASE
                   WHEN support_tickets = 0 THEN '0'
                   WHEN support_tickets BETWEEN 1 AND 2 THEN '1-2'
                   ELSE '3+'
               END AS support_ticket_band,
               COUNT(*) AS customers,
               ROUND(AVG(churn), 4) AS churn_rate
        FROM customer_churn
        GROUP BY support_ticket_band
        ORDER BY churn_rate DESC
    """,
    "churn_by_tenure": """
        SELECT CASE
                   WHEN tenure_months <= 12 THEN '0-12 months'
                   WHEN tenure_months <= 36 THEN '13-36 months'
                   ELSE '37+ months'
               END AS tenure_band,
               COUNT(*) AS customers,
               ROUND(AVG(churn), 4) AS churn_rate
        FROM customer_churn
        GROUP BY tenure_band
        ORDER BY churn_rate DESC
    """,
}


def run_sql_analysis(data_path: Path, output_path: Path) -> pd.DataFrame:
    """Execute the named SQL analyses and save one tidy result file."""
    data = pd.read_csv(data_path)
    frames = []
    with sqlite3.connect(":memory:") as connection:
        data.to_sql("customer_churn", connection, index=False, if_exists="replace")
        for analysis_name, query in QUERIES.items():
            result = pd.read_sql_query(query, connection)
            result.insert(0, "analysis", analysis_name)
            frames.append(result)

    summary = pd.concat(frames, ignore_index=True, sort=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)
    return summary
