"""
End-to-end orchestration: evaluate, select, tune the threshold, explain, audit.

This is the module that enforces the experimental protocol. It runs in a fixed
order and the order is the point:

1.  Score every trained model on the **validation** split.
2.  Select the deployment model from those validation numbers.
3.  Tune the decision threshold on the **validation** split, for that model only.
4.  Freeze both into ``models/deployment.json``.
5.  *Only then* open the **test** split, score it once, and report.
6.  Explain, audit and error-analyse the frozen model on the test split.

Steps 1-4 never touch the test data; step 5 onwards never changes a parameter.
Running it in this order is what lets the reported test metrics be read as an
estimate of generalisation rather than as a number that was optimised toward.

Usage
-----
    python -m src.pipeline --evaluate          # steps 1-6 for the main experiment
    python -m src.pipeline --ablations         # run every ablation experiment
    python -m src.pipeline --all               # train + evaluate + ablations + reports
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config, evaluate, explain, plotting, preprocessing, train

#: Model families eligible for deployment. Isolation Forest is evaluated and
#: reported for comparison but is not a deployment candidate: it is an
#: unsupervised novelty detector whose score is not a probability, so it cannot
#: be operated against a calibrated risk threshold the way the others can.
DEPLOYMENT_CANDIDATES = ("logistic_regression", "random_forest", "xgboost", "lightgbm")


def load_trained(ablation: str = "main") -> dict[str, Any]:
    """Load every persisted pipeline for one experiment."""
    suffix = "" if ablation == "main" else f"_{ablation}"
    pipelines: dict[str, Any] = {}
    for key in evaluate.MODEL_ORDER:
        path = config.MODELS_DIR / f"{key}{suffix}.joblib"
        if path.exists():
            pipelines[key] = joblib.load(path)
    if not pipelines:
        raise FileNotFoundError(
            f"No trained models found for experiment '{ablation}'.\n"
            f"Run: python -m src.train"
            + (f" --ablation {ablation}" if ablation != "main" else "")
        )
    print(f"[pipeline] loaded {len(pipelines)} models for '{ablation}': "
          f"{', '.join(pipelines)}")
    return pipelines


def select_model(validation_table: pd.DataFrame) -> tuple[str, str]:
    """
    Choose the deployment model from validation results. Returns (key, reason).

    Selection is on PR-AUC, not accuracy. PR-AUC summarises how well the model
    *ranks* attacks above benign flows across all thresholds, on the positive
    class only - which is the quantity that survives when the deployment base
    rate differs from the dataset's, and the one that determines whether a
    useful operating point exists at all. Accuracy would reward a model that is
    merely good at the majority class, and F1 alone bakes in the arbitrary 0.50
    threshold that Phase 10 then goes on to replace.
    """
    candidates = validation_table.loc[
        [k for k in DEPLOYMENT_CANDIDATES if k in validation_table.index]]
    best = candidates["pr_auc"].idxmax()
    runner_up = candidates["pr_auc"].drop(index=best).idxmax()
    margin = candidates.loc[best, "pr_auc"] - candidates.loc[runner_up, "pr_auc"]
    reason = (
        f"Highest validation PR-AUC ({candidates.loc[best, 'pr_auc']:.4f}), "
        f"{margin:+.4f} ahead of {evaluate.DISPLAY_NAMES[runner_up]} "
        f"({candidates.loc[runner_up, 'pr_auc']:.4f}). "
        f"Validation recall {candidates.loc[best, 'recall']:.4f}, "
        f"FPR {candidates.loc[best, 'false_positive_rate']:.4f}, "
        f"throughput {candidates.loc[best, 'throughput_flows_per_second']:,.0f} flows/s."
    )
    return str(best), reason


def run_experiment(ablation: str = "main", make_figures: bool = True) -> dict[str, Any]:
    """Evaluate one experiment end to end and persist every artefact."""
    plotting.use_house_style()
    config.ensure_dirs()
    started = time.perf_counter()
    print(f"\n{'=' * 78}\n[pipeline] evaluating experiment: {ablation}\n{'=' * 78}")

    train_df, val_df, test_df = train.load_splits(ablation)
    X_val, y_val = preprocessing.split_xy(val_df)
    X_test, y_test = preprocessing.split_xy(test_df)
    pipelines = load_trained(ablation)

    # ---- Step 1-2: validation scoring and model selection --------------------
    validation_table, validation_scores = evaluate.evaluate_models(pipelines, X_val, y_val)
    tuning_record = _read_json(config.MODELS_DIR /
                               f"best_params{'_main' if ablation == 'main' else '_' + ablation}.json")
    for key, record in (tuning_record or {}).items():
        if key in validation_table.index:
            for field in ("fit_seconds", "search_seconds", "throughput_flows_per_second",
                          "inference_ms_per_1k_rows", "cv_f1_mean", "cv_f1_std",
                          "cv_train_minus_test_f1", "n_features_after_encoding"):
                if field in record:
                    validation_table.loc[key, field] = record[field]
    evaluate.save_table(validation_table, f"validation_metrics_{ablation}")

    best_key, reason = select_model(validation_table)
    print(f"[pipeline] selected: {evaluate.DISPLAY_NAMES[best_key]}\n           {reason}")

    # ---- Step 3: threshold tuning, on validation, for the winner only --------
    sweep, recommendation = evaluate.threshold_analysis(y_val, validation_scores[best_key])
    evaluate.save_table(sweep, f"threshold_sweep_{ablation}")

    # The F1-optimal point is adopted as the headline operating threshold; the
    # FPR-constrained point is reported as the alternative a bandwidth-limited
    # SOC would choose. Both are recorded so the trade-off stays visible.
    chosen_threshold = float(recommendation["max_f1"]["threshold"])
    recommendation["adopted"] = {
        "threshold": chosen_threshold,
        "basis": "F1-optimal on the validation split",
        "alternative_for_constrained_soc": recommendation.get("fpr_constrained"),
    }
    evaluate.save_json(recommendation, f"threshold_recommendation_{ablation}")

    # ---- Step 4: freeze -----------------------------------------------------
    if ablation == "main":
        deployment = {
            "model_key": best_key,
            "display_name": evaluate.DISPLAY_NAMES[best_key],
            "threshold": chosen_threshold,
            "selection_reason": reason,
            "selected_on": "validation split",
            "validation_pr_auc": float(validation_table.loc[best_key, "pr_auc"]),
            "random_state": config.RANDOM_STATE,
            "frozen_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        (config.MODELS_DIR / "deployment.json").write_text(
            json.dumps(deployment, indent=2), encoding="utf-8")
        print(f"[pipeline] froze deployment config: {best_key} @ threshold {chosen_threshold:.3f}")

        from src.predict import build_feature_reference
        reference = build_feature_reference(train_df)
        (config.MODELS_DIR / "feature_reference.json").write_text(
            json.dumps(reference, indent=2), encoding="utf-8")
        print("[pipeline] wrote models/feature_reference.json")

    # ---- Step 5: the test split, opened once --------------------------------
    print(f"[pipeline] scoring the held-out test split (n={len(X_test):,}) "
          "- model and threshold are now frozen")
    thresholds = {key: (chosen_threshold if key == best_key else config.DEFAULT_THRESHOLD)
                  for key in pipelines}
    test_table, test_scores = evaluate.evaluate_models(pipelines, X_test, y_test, thresholds)
    for key in test_table.index:
        if key in validation_table.index:
            for field in ("fit_seconds", "search_seconds", "throughput_flows_per_second",
                          "inference_ms_per_1k_rows", "cv_f1_mean", "cv_f1_std",
                          "cv_train_minus_test_f1", "n_features_after_encoding"):
                if field in validation_table.columns:
                    test_table.loc[key, field] = validation_table.loc[key, field]
    evaluate.save_table(test_table, f"test_metrics_{ablation}")

    # Also record the winner's test metrics at the FPR-constrained threshold, so
    # the report can quote both operating points from measured numbers.
    operating_points = {"adopted_f1_optimal": evaluate.compute_metrics(
        y_test, test_scores[best_key], chosen_threshold)}
    if "fpr_constrained" in recommendation:
        operating_points["fpr_constrained"] = evaluate.compute_metrics(
            y_test, test_scores[best_key], recommendation["fpr_constrained"]["threshold"])
    operating_points["default_0.50"] = evaluate.compute_metrics(
        y_test, test_scores[best_key], 0.50)
    evaluate.save_json(operating_points, f"operating_points_{ablation}")

    results: dict[str, Any] = {
        "ablation": ablation,
        "best_model": best_key,
        "selection_reason": reason,
        "threshold": chosen_threshold,
        "validation_table": validation_table,
        "test_table": test_table,
        "test_scores": test_scores,
        "recommendation": recommendation,
        "operating_points": operating_points,
    }

    if not make_figures:
        results["elapsed_seconds"] = round(time.perf_counter() - started, 1)
        return results

    # ---- Step 6: figures, explanation, audit --------------------------------
    suffix = "" if ablation == "main" else f"_{ablation}"
    evaluate.fig_confusion_matrices(
        y_test, test_scores, thresholds, name=f"fig14_confusion_matrices{suffix}")
    evaluate.fig_roc_pr_curves(y_test, test_scores, name=f"fig15_roc_pr_curves{suffix}")
    evaluate.fig_model_comparison(test_table, name=f"fig16_model_comparison{suffix}")
    evaluate.fig_threshold_analysis(
        sweep, recommendation, evaluate.DISPLAY_NAMES[best_key],
        name=f"fig17_threshold_analysis{suffix}")

    audit = evaluate.subgroup_audit(
        test_df, y_test, test_scores[best_key], chosen_threshold)
    evaluate.save_table(audit, f"subgroup_audit_{ablation}")
    evaluate.fig_subgroup_audit(audit, name=f"fig18_subgroup_audit{suffix}")

    errors = evaluate.error_analysis(
        test_df, y_test, test_scores[best_key], chosen_threshold)
    evaluate.save_json(errors, f"error_analysis_{ablation}")
    results["subgroup_audit"] = audit
    results["error_analysis"] = errors

    if ablation == "main":
        try:
            results["shap"] = explain.run(
                pipelines[best_key], X_test, y_test, test_scores[best_key],
                chosen_threshold, evaluate.DISPLAY_NAMES[best_key], test_df)
        except explain.ExplanationUnavailable as exc:
            print(f"[pipeline] SHAP skipped: {exc}")
            results["shap"] = {"error": str(exc)}

    results["elapsed_seconds"] = round(time.perf_counter() - started, 1)
    print(f"[pipeline] experiment '{ablation}' complete in {results['elapsed_seconds']:.0f}s")
    return results


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def run_ablations(experiments: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Train (with reused hyper-parameters) and evaluate each ablation."""
    experiments = experiments or [k for k in train.ABLATIONS if k != "main"]
    tables: dict[str, pd.DataFrame] = {}

    main_table = config.METRICS_DIR / "test_metrics_main.csv"
    if main_table.exists():
        tables["main"] = pd.read_csv(main_table, index_col=0)

    for name in experiments:
        try:
            train.run(ablation=name)
            outcome = run_experiment(name, make_figures=False)
            tables[name] = outcome["test_table"]
        except Exception as exc:  # noqa: BLE001 - one failed ablation must not
            # abort the rest; the failure is reported rather than hidden.
            print(f"[pipeline] ABLATION '{name}' FAILED: {type(exc).__name__}: {exc}")
            continue

    if len(tables) > 1:
        summary = pd.concat(
            {name: table for name, table in tables.items()}, names=["experiment", "model"])
        evaluate.save_table(summary, "ablation_summary")
        # Chart the model that was actually deployed, not a hardcoded default -
        # otherwise the figure and the reports describe different models.
        deployment = _read_json(config.MODELS_DIR / "deployment.json")
        model_key = (deployment or {}).get("model_key", "xgboost")
        evaluate.fig_ablation_comparison(tables, model_key=model_key)
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the intrusion-detection pipeline.")
    parser.add_argument("--train", action="store_true", help="tune and fit the main models")
    parser.add_argument("--evaluate", action="store_true", help="evaluate the main experiment")
    parser.add_argument("--ablations", action="store_true", help="run every ablation")
    parser.add_argument("--feature-analysis", action="store_true",
                        help="feature selection and PCA supplement")
    parser.add_argument("--reports", action="store_true", help="regenerate the written reports")
    parser.add_argument("--all", action="store_true", help="everything, in order")
    args = parser.parse_args(argv)

    stages = [args.train, args.evaluate, args.ablations, args.feature_analysis,
              args.reports, args.all]
    if not any(stages):
        parser.error("choose at least one stage, or --all")

    if args.train or args.all:
        train.run(ablation="main")
    if args.evaluate or args.all:
        run_experiment("main")
    if args.ablations or args.all:
        run_ablations()
    if args.feature_analysis or args.all:
        # After evaluation, deliberately: this is a supplement, and running it
        # later makes it structurally impossible for it to influence selection.
        from src import feature_analysis
        feature_analysis.run()
    if args.reports or args.all:
        from src import report
        report.generate_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
