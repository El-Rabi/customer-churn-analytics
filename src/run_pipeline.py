"""Run data generation, SQL analysis, training, and evaluation."""

from pathlib import Path

from generate_data import save_dataset
from run_sql_analysis import run_sql_analysis
from train import train_and_evaluate


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "customer_churn.csv"
    results_dir = project_root / "results"

    data = save_dataset(data_path, n_samples=3000, seed=42)
    sql_summary = run_sql_analysis(data_path, results_dir / "sql_summary.csv")
    metrics = train_and_evaluate(data_path, results_dir, seed=42)

    print(f"Generated {len(data):,} synthetic customers")
    print(f"Created {len(sql_summary)} SQL summary rows")
    print(f"Test ROC-AUC: {metrics['test_roc_auc']:.3f}")
    print(f"Test F1: {metrics['test_f1']:.3f}")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
