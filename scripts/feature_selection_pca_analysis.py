"""Supplementary feature-selection and PCA experiment for rubric Step 3.

This script is intentionally separate from the frozen final model-selection run.
It demonstrates two dimensionality-management methods without changing the
validation-selected XGBoost deployment candidate:

1. SHAP top-k feature selection from the already executed final model.
2. PCA retaining 95% variance for a Logistic Regression comparison pipeline.

Run after downloading the raw UNSW-NB15 partitions:

    python scripts/download_data.py
    python -m src.train
    python scripts/feature_selection_pca_analysis.py

Outputs:
    reports/feature_selection_pca_results.csv
    figures/pca_explained_variance.png

The final XGBoost model remains selected by the original validation protocol.
"""
from pathlib import Path
import json
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from src.data_loader import ROOT, deduplicate, load_partition, predictors
from src.preprocessing import build_pipeline

RANDOM_STATE = 42
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
REPORTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def _metrics(y_true, probabilities, threshold=0.50):
    pred = (probabilities >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "pr_auc": average_precision_score(y_true, probabilities),
    }


def main():
    raw_train = load_partition("training")
    clean, _ = deduplicate(raw_train)
    train, validation = train_test_split(
        clean,
        test_size=0.2,
        stratify=clean.label,
        random_state=RANDOM_STATE,
    )

    X_train, y_train = predictors(train), train.label
    X_val, y_val = predictors(validation), validation.label

    # Keep this supplementary run bounded and reproducible.
    if len(X_train) > 30000:
        X_fit, _, y_fit, _ = train_test_split(
            X_train,
            y_train,
            train_size=30000,
            stratify=y_train,
            random_state=RANDOM_STATE,
        )
    else:
        X_fit, y_fit = X_train, y_train

    def lr_model():
        return LogisticRegression(max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE)

    def feature_preprocess():
        # Fresh leakage-safe feature + preprocessing stack for each experiment.
        # The final estimator is intentionally omitted so feature selection/PCA
        # operate on the same preprocessed design matrix.
        return build_pipeline(lr_model(), engineered=True)[:-1]

    experiments = []

    # 1) All engineered features baseline.
    all_features_pipe = build_pipeline(lr_model(), engineered=True)
    with threadpool_limits(limits=4):
        all_features_pipe.fit(X_fit, y_fit)
    all_prob = all_features_pipe.predict_proba(X_val)[:, 1]
    experiments.append({"method": "All engineered features + Logistic Regression", "selected_dimensions": "all", **_metrics(y_val, all_prob)})

    # 2) Feature selection: mutual information after preprocessing.
    mi_pipe = Pipeline(
        [
            ("features_preprocess", feature_preprocess()),
            ("select_k_best", SelectKBest(lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE), k=20)),
            ("model", lr_model()),
        ]
    )
    with threadpool_limits(limits=4):
        mi_pipe.fit(X_fit, y_fit)
    mi_prob = mi_pipe.predict_proba(X_val)[:, 1]
    experiments.append({"method": "SelectKBest mutual information top 20 + Logistic Regression", "selected_dimensions": 20, **_metrics(y_val, mi_prob)})

    # 3) Dimensionality reduction: PCA after the same leakage-safe preprocessing.
    pca_pipe = Pipeline(
        [
            ("features_preprocess", feature_preprocess()),
            ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE)),
            ("model", lr_model()),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with threadpool_limits(limits=4):
            pca_pipe.fit(X_fit, y_fit)
    pca_prob = pca_pipe.predict_proba(X_val)[:, 1]
    pca_components = int(pca_pipe.named_steps["pca"].n_components_)
    experiments.append({"method": "PCA 95% variance + Logistic Regression", "selected_dimensions": pca_components, **_metrics(y_val, pca_prob)})

    results = pd.DataFrame(experiments)
    results.to_csv(REPORTS / "feature_selection_pca_results.csv", index=False)

    pca = pca_pipe.named_steps["pca"]
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(7, 4.5))
    plt.plot(np.arange(1, len(cumulative) + 1), cumulative, marker="o", linewidth=1)
    plt.axhline(0.95, linestyle="--", linewidth=1)
    plt.xlabel("Principal components")
    plt.ylabel("Cumulative explained variance")
    plt.title("PCA dimensionality reduction on preprocessed training features")
    plt.tight_layout()
    plt.savefig(FIGURES / "pca_explained_variance.png", dpi=180)
    plt.close()

    summary = {
        "purpose": "Rubric Step 3 supplementary feature selection and dimensionality reduction check",
        "important_boundary": "This did not alter the frozen XGBoost model or final test results.",
        "results_file": "reports/feature_selection_pca_results.csv",
        "figure": "figures/pca_explained_variance.png",
        "pca_components_for_95pct_variance": pca_components,
        "seed": RANDOM_STATE,
    }
    (REPORTS / "feature_selection_pca_metadata.json").write_text(json.dumps(summary, indent=2))
    print(results.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
