"""Tests for the synthetic churn data and model pipeline."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_data import generate_customer_data  # noqa: E402
from train import build_pipeline  # noqa: E402


class ChurnPipelineTests(unittest.TestCase):
    def test_generated_dataset_has_expected_shape_and_target(self) -> None:
        data = generate_customer_data(n_samples=300, seed=7)
        self.assertEqual(len(data), 300)
        self.assertIn("churn", data.columns)
        self.assertSetEqual(set(data["churn"].unique()), {0, 1})
        self.assertGreater(data["churn"].mean(), 0.10)
        self.assertLess(data["churn"].mean(), 0.80)

    def test_pipeline_fits_mixed_features(self) -> None:
        data = generate_customer_data(n_samples=300, seed=8)
        X = data.drop(columns=["customer_id", "churn"])
        model = build_pipeline(X, seed=8)
        model.fit(X, data["churn"])
        prediction = model.predict(X.head(10))
        self.assertEqual(len(prediction), 10)


if __name__ == "__main__":
    unittest.main()
