"""
Model definitions, hyper-parameter search, and persistence.

Protocol
--------
1. Split the deduplicated corpus 60/20/20, stratified on attack family.
2. For each model family, run ``RandomizedSearchCV`` with 5-fold *stratified*
   cross-validation **inside the training split only**, optimising F1 on the
   attack class.
3. Refit the winning configuration on the full training split.
4. Score every fitted model on the **validation** split. The test split is not
   touched anywhere in this module - it is opened once, in ``src.evaluate``,
   after the final model and the decision threshold are both frozen.

Every model is a complete ``Pipeline`` (feature engineering -> preprocessing ->
estimator), so the fold-wise refitting inside the search re-learns the scaler
and the one-hot vocabulary on each training fold. A scaler fitted on the whole
training split before cross-validation would leak fold-level information into
the validation folds; building the pipeline this way makes that impossible.

Usage
-----
    python -m src.train                        # tune + fit the four supervised models
    python -m src.train --quick                # smaller search, for a fast smoke test
    python -m src.train --ablation no_ttl      # re-fit with tuned params, TTL removed
    python -m src.train --list-ablations
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src import config, preprocessing

# XGBoost and LightGBM emit a deprecation notice about `use_label_encoder` on
# some builds. Everything else is left visible on purpose.
warnings.filterwarnings("ignore", message=".*use_label_encoder.*")

N_JOBS = -1


# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
@dataclass
class ModelSpec:
    """A model family plus the search space used to tune it."""

    key: str
    display_name: str
    build: Callable[[], Any]
    param_distributions: dict[str, Any]
    n_iter: int
    #: Set when the estimator parallelises internally; the search then runs
    #: sequentially to avoid oversubscribing 16 cores with nested pools.
    estimator_parallel: bool = False
    notes: str = ""
    tags: list[str] = field(default_factory=list)


def _logistic_regression() -> LogisticRegression:
    """
    L2-penalised logistic regression solved with L-BFGS.

    Solver choice is empirical, not default-by-habit. Benchmarked on the 92,210
    x 73 training design matrix (see reports/Model_Evaluation_Report.md,
    "Solver selection"):

        lbfgs     / L2 / C=1     3.6 s   validation F1 0.8566
        liblinear / L2 / C=1    10.0 s   validation F1 0.8560
        saga      / L2 / C=1   101.8 s   validation F1 0.8561
        saga      / L1 / C=1   199.4 s   validation F1 0.8562  (did not converge)

    L1 was dropped: it cost 55x the runtime of L-BFGS for a 0.0004 F1
    difference, and hit the iteration cap without converging. `liblinear` with
    an L1 penalty and a large C is pathological here because the TTL features
    make the classes very nearly linearly separable, so the unregularised
    optimum runs off to infinite coefficients.

    ``max_iter=3000`` is set above the observed worst case (689 iterations at
    C=100) so that a converged fit is the norm and a ConvergenceWarning is a
    real signal rather than background noise.
    """
    return LogisticRegression(
        solver="lbfgs",
        penalty="l2",
        max_iter=3000,
        random_state=config.RANDOM_STATE,
        # n_jobs is not set: lbfgs ignores it for binary problems, and leaving it
        # unset avoids oversubscribing the CPU against the search's own pool.
    )


def _random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=N_JOBS,
        bootstrap=True,
    )


def _xgboost():
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=N_JOBS,
    )


def _lightgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        objective="binary",
        random_state=config.RANDOM_STATE,
        n_jobs=N_JOBS,
        verbose=-1,
    )


def model_registry(quick: bool = False) -> dict[str, ModelSpec]:
    """Return the four supervised model families and their search spaces."""
    scale = 1 if not quick else 0  # quick mode shrinks every n_iter

    specs = [
        ModelSpec(
            key="logistic_regression",
            display_name="Logistic Regression",
            build=_logistic_regression,
            param_distributions={
                "model__C": loguniform(1e-3, 1e2),
                "model__class_weight": [None, "balanced"],
            },
            n_iter=16 if scale else 4,
            notes=(
                "Interpretable linear baseline. Coefficients are directly readable "
                "as log-odds contributions, which makes it the reference point for "
                "judging whether the ensembles' extra complexity is earning its keep."
            ),
            tags=["linear", "interpretable", "baseline"],
        ),
        ModelSpec(
            key="random_forest",
            display_name="Random Forest",
            build=_random_forest,
            param_distributions={
                "model__n_estimators": randint(200, 601),
                "model__max_depth": [None, 12, 18, 24, 32],
                "model__min_samples_leaf": randint(1, 12),
                "model__min_samples_split": randint(2, 20),
                # max_features is capped at 0.3 of 73 columns (~22). Benchmarking
                # showed 0.5 roughly quadruples fit time against "sqrt" for no
                # measurable F1 gain, so the extra cost buys nothing here.
                "model__max_features": ["sqrt", "log2", 0.3],
                "model__class_weight": [None, "balanced", "balanced_subsample"],
            },
            n_iter=25 if scale else 4,
            estimator_parallel=True,
            notes=(
                "Bagged axis-aligned trees. Robust to the dataset's heavy tails and "
                "mixed feature scales, and provides variance reduction without the "
                "sequential-fitting cost of boosting."
            ),
            tags=["ensemble", "bagging", "tree"],
        ),
        ModelSpec(
            key="xgboost",
            display_name="XGBoost",
            build=_xgboost,
            param_distributions={
                "model__n_estimators": randint(200, 801),
                "model__max_depth": randint(3, 13),
                "model__learning_rate": loguniform(0.01, 0.35),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 12),
                "model__reg_lambda": loguniform(1e-2, 20.0),
                "model__reg_alpha": loguniform(1e-4, 5.0),
                "model__gamma": uniform(0.0, 0.5),
            },
            n_iter=30 if scale else 4,
            estimator_parallel=True,
            notes=(
                "Gradient-boosted trees. Expected to lead on tabular flow data; the "
                "primary candidate for deployment and the model carried into the "
                "SHAP analysis."
            ),
            tags=["ensemble", "boosting", "tree"],
        ),
        ModelSpec(
            key="lightgbm",
            display_name="LightGBM",
            build=_lightgbm,
            param_distributions={
                "model__n_estimators": randint(200, 801),
                "model__num_leaves": randint(20, 160),
                "model__learning_rate": loguniform(0.01, 0.35),
                "model__min_child_samples": randint(5, 80),
                "model__subsample": uniform(0.6, 0.4),
                "model__subsample_freq": [0, 1, 3],
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__reg_lambda": loguniform(1e-2, 20.0),
            },
            n_iter=25 if scale else 4,
            estimator_parallel=True,
            notes=(
                "Optional fourth model: leaf-wise boosting. Included to check that "
                "the XGBoost result is a property of gradient boosting rather than "
                "of one library's defaults."
            ),
            tags=["ensemble", "boosting", "tree", "optional"],
        ),
    ]
    return {spec.key: spec for spec in specs}


# --------------------------------------------------------------------------- #
# Class-imbalance handling
# --------------------------------------------------------------------------- #
def scale_pos_weight(y: pd.Series) -> float:
    """``n_negative / n_positive`` - XGBoost/LightGBM's re-weighting knob."""
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    return float(negatives / max(positives, 1))


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #
def tune_model(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    include_engineered: bool = True,
    exclude: tuple[str, ...] = (),
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the randomised search for one model family and refit the winner."""
    estimator = spec.build()

    # Boosters take their imbalance correction as a scalar, so it is searched
    # over {1.0 (off), n_neg/n_pos (fully balanced)} rather than a string.
    params = dict(spec.param_distributions)
    if spec.key in ("xgboost", "lightgbm"):
        params["model__scale_pos_weight"] = [1.0, scale_pos_weight(y_train)]

    pipeline = preprocessing.build_model_pipeline(
        estimator, include_engineered=include_engineered, exclude=exclude)

    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=params,
        n_iter=spec.n_iter,
        scoring={"f1": "f1", "average_precision": "average_precision", "roc_auc": "roc_auc"},
        refit="f1",
        cv=cv,
        n_jobs=1 if spec.estimator_parallel else N_JOBS,
        random_state=config.RANDOM_STATE,
        return_train_score=True,
        error_score="raise",
        verbose=0,
    )

    if verbose:
        print(f"[train] {spec.display_name}: searching {spec.n_iter} configurations "
              f"x {config.CV_FOLDS} folds ...")
    started = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - started

    best_index = int(search.best_index_)
    cv_results = search.cv_results_
    outcome = {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "best_params": {k.removeprefix("model__"): _jsonify(v)
                        for k, v in search.best_params_.items()},
        "cv_f1_mean": float(cv_results["mean_test_f1"][best_index]),
        "cv_f1_std": float(cv_results["std_test_f1"][best_index]),
        "cv_average_precision_mean": float(cv_results["mean_test_average_precision"][best_index]),
        "cv_roc_auc_mean": float(cv_results["mean_test_roc_auc"][best_index]),
        "cv_train_f1_mean": float(cv_results["mean_train_f1"][best_index]),
        "search_seconds": round(elapsed, 2),
        "n_candidates": spec.n_iter,
        "notes": spec.notes,
        "tags": spec.tags,
    }
    # Overfitting signal: how much better is the model on data it was fitted on?
    outcome["cv_train_minus_test_f1"] = round(
        outcome["cv_train_f1_mean"] - outcome["cv_f1_mean"], 5)

    # Time a clean refit on the full training split so "training time" in the
    # comparison table is a single reproducible fit, not the whole search.
    best_pipeline = search.best_estimator_
    refit_started = time.perf_counter()
    best_pipeline.fit(X_train, y_train)
    outcome["fit_seconds"] = round(time.perf_counter() - refit_started, 3)

    if verbose:
        print(f"[train]   best CV F1 = {outcome['cv_f1_mean']:.4f} "
              f"(+/-{outcome['cv_f1_std']:.4f})  "
              f"PR-AUC = {outcome['cv_average_precision_mean']:.4f}  "
              f"search {elapsed:.0f}s, refit {outcome['fit_seconds']:.1f}s")
        print(f"[train]   params: {outcome['best_params']}")

    return {"pipeline": best_pipeline, "outcome": outcome, "search": search}


def _jsonify(value: Any) -> Any:
    """Make numpy scalars JSON-serialisable."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def measure_inference(pipeline: Pipeline, X: pd.DataFrame, repeats: int = 3) -> dict[str, float]:
    """
    Median wall-clock latency for scoring ``X``, reported per 1,000 flows.

    Uses whichever scoring method the estimator exposes: the supervised models
    provide ``predict_proba``; Isolation Forest is a novelty detector and offers
    ``score_samples`` instead. The timing therefore covers the same unit of work
    in both cases - transform the batch, produce one score per flow.
    """
    score = getattr(pipeline, "predict_proba", None) or pipeline.score_samples

    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        score(X)
        timings.append(time.perf_counter() - started)
    median = float(np.median(timings))
    return {
        "inference_seconds_total": round(median, 4),
        "inference_rows": int(len(X)),
        "inference_ms_per_1k_rows": round(median / len(X) * 1000 * 1000, 3),
        "throughput_flows_per_second": round(len(X) / median, 1),
    }


# --------------------------------------------------------------------------- #
# Isolation Forest: unsupervised comparison
# --------------------------------------------------------------------------- #
def train_isolation_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    include_engineered: bool = True,
    exclude: tuple[str, ...] = (),
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Fit an Isolation Forest on **benign training flows only**.

    This is the honest framing of anomaly detection for intrusion detection: the
    detector is shown what normal looks like and must flag everything else. It
    answers a question the supervised models cannot - "how far could we get
    without any attack labels at all?" - and sets the floor that supervised
    learning has to beat to justify its labelling cost.
    """
    benign = X_train.loc[y_train == 0]
    contamination = float(np.clip(1.0 - y_train.mean(), 0.0, 1.0))
    detector = IsolationForest(
        n_estimators=300,
        max_samples="auto",
        contamination="auto",
        random_state=config.RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    pipeline = preprocessing.build_model_pipeline(
        detector, include_engineered=include_engineered, exclude=exclude)

    started = time.perf_counter()
    pipeline.fit(benign)
    elapsed = time.perf_counter() - started

    if verbose:
        print(f"[train] Isolation Forest fitted on {len(benign):,} benign flows "
              f"in {elapsed:.1f}s (attack labels unused)")

    return {
        "pipeline": pipeline,
        "outcome": {
            "model_key": "isolation_forest",
            "display_name": "Isolation Forest (unsupervised)",
            "best_params": {"n_estimators": 300, "contamination": "auto",
                            "fitted_on": "benign training flows only"},
            "fit_seconds": round(elapsed, 3),
            "benign_training_rows": int(len(benign)),
            "benign_fraction_of_train": round(contamination, 4),
            "notes": (
                "Unsupervised baseline trained without any attack labels. Included "
                "to quantify what supervised labelling actually buys, and because a "
                "novelty detector is the natural complement to a supervised model "
                "for zero-day traffic the classifier has never seen."
            ),
            "tags": ["unsupervised", "anomaly", "optional"],
        },
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
ABLATIONS: dict[str, str] = {
    "main": "Primary protocol: deduplicated corpus, all features, engineered features on.",
    "keep_duplicates": "Duplicate flow records retained, quantifying the optimism they cause.",
    "no_ttl": "sttl, dttl and ct_state_ttl removed, quantifying reliance on the TTL artefact.",
    "no_engineered": "Engineered features disabled, isolating their contribution.",
    "smote": "SMOTE oversampling of the training split instead of class weighting.",
    "official_split": "The dataset authors' published train/test partition, duplicates intact.",
}


def load_splits(ablation: str = "main"):
    """Return ``(train, val, test)`` frames for the requested experiment."""
    if ablation == "official_split":
        return preprocessing.split_official(verbose=True)
    deduplicate = ablation != "keep_duplicates"
    corpus, _ = preprocessing.prepare_corpus(deduplicate=deduplicate, verbose=True)
    return preprocessing.split_corpus(corpus, verbose=True)


def ablation_settings(ablation: str) -> dict[str, Any]:
    """Feature-level switches implied by an ablation name."""
    return {
        "include_engineered": ablation != "no_engineered",
        "exclude": config.TTL_FEATURES if ablation == "no_ttl" else (),
        "use_smote": ablation == "smote",
    }


def run(
    ablation: str = "main",
    quick: bool = False,
    models: list[str] | None = None,
) -> dict[str, Any]:
    """Tune (or re-fit) every model for one experiment and persist the results."""
    config.ensure_dirs()
    settings = ablation_settings(ablation)
    print(f"\n{'=' * 78}\n[train] experiment: {ablation}\n"
          f"[train] {ABLATIONS[ablation]}\n{'=' * 78}")

    train_df, val_df, _ = load_splits(ablation)
    X_train, y_train = preprocessing.split_xy(train_df)
    X_val, y_val = preprocessing.split_xy(val_df)

    if settings["use_smote"]:
        X_train, y_train = _apply_smote(X_train, y_train)

    registry = model_registry(quick=quick)
    selected = models or list(registry)

    # Ablations reuse the hyper-parameters tuned in the main run rather than
    # re-searching. That keeps the comparison honest (only one thing changes)
    # and keeps the runtime tractable.
    tuned_params = _load_tuned_params() if ablation != "main" else None
    if tuned_params:
        print("[train] reusing hyper-parameters tuned in the main experiment")

    results: dict[str, Any] = {}
    fitted: dict[str, Pipeline] = {}

    for key in selected:
        spec = registry[key]
        if tuned_params and key in tuned_params:
            fitted_model, outcome = _refit_with_params(
                spec, tuned_params[key], X_train, y_train, settings)
        else:
            found = tune_model(
                spec, X_train, y_train,
                include_engineered=settings["include_engineered"],
                exclude=settings["exclude"])
            fitted_model, outcome = found["pipeline"], found["outcome"]

        outcome.update(measure_inference(fitted_model, X_val))
        outcome["n_features_after_encoding"] = int(
            len(preprocessing.transformed_feature_names(fitted_model)))
        results[key] = outcome
        fitted[key] = fitted_model

    # Unsupervised comparison (never tuned - it sees no labels).
    iso = train_isolation_forest(
        X_train, y_train,
        include_engineered=settings["include_engineered"],
        exclude=settings["exclude"])
    iso["outcome"].update(measure_inference(iso["pipeline"], X_val))
    results["isolation_forest"] = iso["outcome"]
    fitted["isolation_forest"] = iso["pipeline"]

    _persist(fitted, results, ablation, settings, X_train, y_train, X_val, y_val)
    return {"results": results, "pipelines": fitted}


def _apply_smote(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """
    Oversample the minority class with SMOTE - training split only.

    SMOTE interpolates between neighbouring minority points. On this data that
    is questionable: many features are Boolean flags or bounded counters, and a
    synthetic flow with ``tcp_handshake_complete = 0.43`` does not correspond to
    any packet sequence that could exist on a wire. It is applied here purely so
    the ``smote`` ablation can compare it against class weighting on evidence
    rather than on assertion. The categorical columns are excluded from the
    interpolation by SMOTENC's nominal handling.
    """
    from imblearn.over_sampling import SMOTENC

    categorical_idx = [X.columns.get_loc(c) for c in config.CATEGORICAL_FEATURES
                       if c in X.columns]
    sampler = SMOTENC(
        categorical_features=categorical_idx,
        random_state=config.RANDOM_STATE,
        k_neighbors=5,
    )
    before = dict(y.value_counts().sort_index())
    X_res, y_res = sampler.fit_resample(X, y)
    after = dict(pd.Series(y_res).value_counts().sort_index())
    print(f"[train] SMOTENC applied to the TRAINING SPLIT ONLY: {before} -> {after}")

    # SMOTENC returns an object-dtype frame. Restore the original dtypes, or the
    # downstream StandardScaler receives object columns and fails.
    resampled = pd.DataFrame(X_res, columns=X.columns)
    for column in X.columns:
        if column in config.CATEGORICAL_FEATURES:
            resampled[column] = resampled[column].astype(str)
        else:
            resampled[column] = pd.to_numeric(resampled[column], errors="coerce").fillna(0.0)
    return resampled, pd.Series(y_res, name=y.name).astype(int)


def _load_tuned_params() -> dict[str, dict] | None:
    path = config.MODELS_DIR / "best_params_main.json"
    if not path.exists():
        print("[train] no tuned parameters found; this ablation will tune from scratch")
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {k: v["best_params"] for k, v in payload.items() if "best_params" in v}


def _refit_with_params(spec, params, X_train, y_train, settings):
    """Re-fit one model family with previously tuned hyper-parameters."""
    estimator = spec.build()
    usable = {k: v for k, v in params.items() if k in estimator.get_params()}
    estimator.set_params(**usable)
    pipeline = preprocessing.build_model_pipeline(
        estimator,
        include_engineered=settings["include_engineered"],
        exclude=settings["exclude"])
    started = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - started
    print(f"[train] {spec.display_name}: refitted with tuned params in {elapsed:.1f}s")
    return pipeline, {
        "model_key": spec.key,
        "display_name": spec.display_name,
        "best_params": {k: _jsonify(v) for k, v in usable.items()},
        "fit_seconds": round(elapsed, 3),
        "reused_tuning_from": "main",
        "notes": spec.notes,
        "tags": spec.tags,
    }


def _persist(fitted, results, ablation, settings, X_train, y_train, X_val, y_val) -> None:
    """Write fitted pipelines and the tuning record to disk."""
    suffix = "" if ablation == "main" else f"_{ablation}"
    for key, pipeline in fitted.items():
        path = config.MODELS_DIR / f"{key}{suffix}.joblib"
        joblib.dump(pipeline, path, compress=3)
        print(f"[train] saved models/{path.name} "
              f"({path.stat().st_size / 1e6:.1f} MB)")

    params_path = config.MODELS_DIR / f"best_params{suffix or '_main'}.json"
    params_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"[train] saved models/{params_path.name}")

    manifest = {
        "experiment": ablation,
        "description": ABLATIONS[ablation],
        "settings": {k: list(v) if isinstance(v, tuple) else v
                     for k, v in settings.items()},
        "random_state": config.RANDOM_STATE,
        "cv_folds": config.CV_FOLDS,
        "tuning_scorer": config.TUNING_SCORER,
        "n_train_rows": int(len(X_train)),
        "n_val_rows": int(len(X_val)),
        "train_attack_rate": round(float(y_train.mean()), 5),
        "val_attack_rate": round(float(y_val.mean()), 5),
        "scale_pos_weight_used": round(scale_pos_weight(y_train), 4),
        "models": sorted(fitted),
    }
    manifest_path = config.MODELS_DIR / f"manifest{suffix or '_main'}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[train] saved models/{manifest_path.name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train intrusion-detection models.")
    parser.add_argument("--ablation", default="main", choices=sorted(ABLATIONS))
    parser.add_argument("--quick", action="store_true",
                        help="tiny search space; smoke test only, not for reported results")
    parser.add_argument("--models", nargs="*", default=None,
                        help="subset of model keys to run")
    parser.add_argument("--list-ablations", action="store_true")
    args = parser.parse_args(argv)

    if args.list_ablations:
        for name, description in ABLATIONS.items():
            print(f"  {name:<16} {description}")
        return 0

    run(ablation=args.ablation, quick=args.quick, models=args.models)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
