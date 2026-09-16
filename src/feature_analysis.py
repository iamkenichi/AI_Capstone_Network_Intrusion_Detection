"""
Feature selection and dimensionality reduction, run as a supplement.

Two questions the main pipeline does not answer on its own:

**Feature selection.** SHAP tells you what the fitted XGBoost model *used*. It
does not tell you what a *different* model would find informative, and it cannot
be computed before training. A filter method can. Mutual information ranks
features by dependence on the target without reference to any particular
estimator, which makes it an independent second opinion on the feature set -
and, applied at several values of k, it answers "how few features does this
problem actually need?"

**Dimensionality reduction.** One-hot encoding the protocol, service and state
columns pushes the design matrix past 70 columns, most of them sparse. PCA asks
whether that width is real or redundant.

Both are deliberately kept **out of the frozen model-selection workflow**. The
deployed model was selected and its threshold tuned before any of this ran, so
nothing here can retro-fit the headline result. Everything is fitted on the
training split only and scored on validation: **the test split is not touched by
this module at all**, which is why its results are reported as a supplement
rather than folded into the headline table.
"""

from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from src import config, preprocessing

#: Feature counts to evaluate. 20 is the comparison most commonly reported on
#: UNSW-NB15; the rest trace out the curve either side of it.
K_VALUES: tuple[int, ...] = (10, 20, 30, 40)

#: Cumulative explained variance PCA must retain.
PCA_VARIANCE: float = 0.95


def _probe() -> LogisticRegression:
    """A deliberately simple, fast probe.

    Logistic regression rather than the deployed XGBoost because the question is
    "how much signal survives this transformation", not "how good can we make
    this". A linear probe answers that without the confound of a boosted model's
    ability to rebuild interactions that PCA has just destroyed.
    """
    return LogisticRegression(max_iter=2000, class_weight="balanced",
                              random_state=config.RANDOM_STATE)


def _fit_score(Z_train: np.ndarray, y_train, Z_val: np.ndarray, y_val) -> dict[str, float]:
    model = _probe().fit(Z_train, y_train)
    proba = model.predict_proba(Z_val)[:, 1]
    pred = (proba >= config.DEFAULT_THRESHOLD).astype(int)
    return {
        "f1": round(float(f1_score(y_val, pred, zero_division=0)), 6),
        "roc_auc": round(float(roc_auc_score(y_val, proba)), 6),
        "pr_auc": round(float(average_precision_score(y_val, proba)), 6),
    }


def run(verbose: bool = True) -> pd.DataFrame:
    """Fit every variant on train, score on validation, write the results."""
    config.ensure_dirs()
    train, val, _, manifest = preprocessing.split_published(verbose=verbose)
    X_train, y_train = preprocessing.split_xy(train)
    X_val, y_val = preprocessing.split_xy(val)

    # Fit the encoder ONCE, on training data only, then reuse the transformed
    # matrices. Refitting per variant would be both slower and pointless: the
    # encoding is not what is being compared.
    encoder = preprocessing.build_feature_pipeline()
    Z_train = np.asarray(encoder.fit_transform(X_train, y_train), dtype=float)
    Z_val = np.asarray(encoder.transform(X_val), dtype=float)
    names = np.asarray(preprocessing.transformed_feature_names(encoder))
    n_encoded = Z_train.shape[1]
    if verbose:
        print(f"[feature_analysis] encoded design matrix: "
              f"{Z_train.shape[0]:,} x {n_encoded}")

    rows: list[dict[str, object]] = [
        {"method": "All features (baseline)", "n_features": int(n_encoded),
         **_fit_score(Z_train, y_train, Z_val, y_val)}
    ]

    # ---------------- feature selection: mutual information ----------------
    # Computed once. mutual_info_classif is a kNN estimator and is the expensive
    # step here; every k below is a slice of this one ranking.
    if verbose:
        print("[feature_analysis] estimating mutual information (this is the slow part) ...")
    mi = mutual_info_classif(Z_train, y_train, random_state=config.RANDOM_STATE)
    order = np.argsort(mi)[::-1]

    for k in K_VALUES:
        if k >= n_encoded:
            continue
        keep = order[:k]
        rows.append({"method": f"Mutual information, top {k}", "n_features": int(k),
                     **_fit_score(Z_train[:, keep], y_train, Z_val[:, keep], y_val)})
        if verbose:
            print(f"[feature_analysis] top {k:>2} features: "
                  f"PR-AUC {rows[-1]['pr_auc']:.4f}")

    # ---------------- dimensionality reduction: PCA ----------------
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full",
              random_state=config.RANDOM_STATE)
    P_train = pca.fit_transform(Z_train)
    P_val = pca.transform(Z_val)
    rows.append({"method": f"PCA ({PCA_VARIANCE:.0%} variance)",
                 "n_features": int(pca.n_components_),
                 **_fit_score(P_train, y_train, P_val, y_val)})
    if verbose:
        print(f"[feature_analysis] PCA: {pca.n_components_} components retain "
              f"{PCA_VARIANCE:.0%} of variance")

    cumulative = np.cumsum(pca.explained_variance_ratio_)
    details = {
        "generated": date.today().isoformat(),
        "protocol": manifest["protocol"],
        "fitted_on": "training split only",
        "scored_on": "validation split",
        "test_split_touched": False,
        "random_state": config.RANDOM_STATE,
        "probe_estimator": "LogisticRegression(class_weight='balanced')",
        "n_features_after_encoding": int(n_encoded),
        "mutual_information_ranking": [
            {"rank": i + 1, "feature": str(names[j]),
             "mutual_information": round(float(mi[j]), 6)}
            for i, j in enumerate(order[:20])],
        "selected_top_20": sorted(str(n) for n in names[order[:20]]),
        "pca": {
            "variance_target": PCA_VARIANCE,
            "components_retained": int(pca.n_components_),
            "components_for_50pct": int(np.searchsorted(cumulative, 0.50) + 1),
            "first_component_variance": round(float(pca.explained_variance_ratio_[0]), 6),
            "cumulative_variance_first_10": [round(float(v), 6) for v in cumulative[:10]],
        },
    }

    table = pd.DataFrame(rows)
    table.to_csv(config.METRICS_DIR / "feature_selection_pca.csv", index=False)
    print("[feature_analysis] wrote reports/metrics/feature_selection_pca.csv")
    (config.METRICS_DIR / "feature_selection_pca.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8")
    print("[feature_analysis] wrote reports/metrics/feature_selection_pca.json")
    return table


if __name__ == "__main__":
    run()
