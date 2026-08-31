"""Train and evaluate the customer churn classification pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "churn"
ID_COLUMN = "customer_id"


def build_pipeline(X: pd.DataFrame, seed: int = 42) -> Pipeline:
    """Build preprocessing and Random Forest steps."""
    categorical_columns = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_columns = [column for column in X.columns if column not in categorical_columns]

    preprocessing = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric_columns,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def train_and_evaluate(data_path: Path, results_dir: Path, seed: int = 42) -> dict[str, float]:
    """Train the pipeline and write metrics and recruiter-readable plots."""
    data = pd.read_csv(data_path)
    X = data.drop(columns=[TARGET, ID_COLUMN])
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    pipeline = build_pipeline(X, seed=seed)
    cross_validation = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
        n_jobs=-1,
    )
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    probability = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "test_accuracy": float(accuracy_score(y_test, prediction)),
        "test_precision": float(precision_score(y_test, prediction, zero_division=0)),
        "test_recall": float(recall_score(y_test, prediction, zero_division=0)),
        "test_f1": float(f1_score(y_test, prediction, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, probability)),
        "cv_roc_auc_mean": float(np.mean(cross_validation["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cross_validation["test_roc_auc"], ddof=1)),
        "cv_f1_mean": float(np.mean(cross_validation["test_f1"])),
        "test_customers": int(len(y_test)),
        "test_churn_rate": float(y_test.mean()),
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, prediction, cmap="Blues", ax=ax, colorbar=False)
    ax.set_title("Customer churn confusion matrix")
    fig.tight_layout()
    fig.savefig(results_dir / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probability)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC-AUC = {metrics['test_roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", title="Customer churn ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(results_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    feature_names = [
        name.replace("numeric__", "").replace("categorical__", "")
        for name in pipeline.named_steps["preprocess"].get_feature_names_out()
    ]
    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipeline.named_steps["model"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(results_dir / "feature_importance.csv", index=False)

    top_features = importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_features["feature"], top_features["importance"], color="#2E6F95")
    ax.set(xlabel="Random Forest importance", title="Top churn model features")
    fig.tight_layout()
    fig.savefig(results_dir / "feature_importance.png", dpi=160)
    plt.close(fig)
    return metrics
