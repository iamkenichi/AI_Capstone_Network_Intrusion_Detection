"""
Evaluation: metrics, curves, threshold analysis, error analysis and the
operational subgroup audit.

Separation of concerns
----------------------
Everything that *selects* something - which model wins, which decision
threshold to operate at - is computed on the **validation** split. The test
split is scored exactly once, at the end, with the model and threshold already
frozen. That ordering is enforced by the function signatures: ``select_model``
and ``threshold_analysis`` take validation data, ``final_evaluation`` takes the
test data and no longer has any tunable argument.

Why these metrics
-----------------
Accuracy is reported because the rubric asks for it, but it is close to useless
here and is never used to choose anything. The metrics that carry the decision
are recall on the attack class (what fraction of intrusions do we catch),
false-negative rate (its complement - the risk that matters), false-positive
rate (the analyst workload, and the only workload metric that transfers to a
different base rate), and PR-AUC (threshold-free ranking quality on the
positive class).
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config, plotting

#: Order in which models appear in every table and chart.
MODEL_ORDER = [
    "logistic_regression", "random_forest", "xgboost", "lightgbm", "isolation_forest",
]

DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "isolation_forest": "Isolation Forest",
}


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def attack_scores(pipeline, X: pd.DataFrame) -> np.ndarray:
    """
    Return one "how malicious is this flow" score per row, higher = more likely
    an attack.

    Supervised models return a calibrated-ish probability from ``predict_proba``.
    Isolation Forest is a novelty detector: ``score_samples`` returns higher
    values for *more normal* points, so the sign is flipped and the result is
    min-max rescaled to [0, 1]. The rescaled value is a **ranking score, not a
    probability** - ROC-AUC and PR-AUC are rank-based and therefore valid, but
    it must not be read as "68% chance this is an attack", and its threshold is
    not comparable to a supervised model's threshold.
    """
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    raw = -pipeline.score_samples(X)  # flip: higher = more anomalous
    span = raw.max() - raw.min()
    return (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_score: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
    include_probabilistic: bool = True,
) -> dict[str, float]:
    """Full metric panel at one decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics: dict[str, float] = {
        "threshold": round(float(threshold), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # FPR: share of BENIGN flows wrongly alerted -> analyst workload.
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        # FNR: share of ATTACK flows missed -> residual security risk.
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) else 0.0,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "n": int(len(y_true)),
        "positive_rate_actual": float(y_true.mean()),
        "positive_rate_predicted": float(y_pred.mean()),
    }
    if include_probabilistic:
        metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        metrics["pr_auc"] = average_precision_score(y_true, y_score)
        metrics["brier"] = brier_score_loss(y_true, np.clip(y_score, 0, 1))
    return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in metrics.items()}


def evaluate_models(
    pipelines: dict[str, Any],
    X: pd.DataFrame,
    y: pd.Series,
    thresholds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Score every model on one split. Returns ``(table, scores_by_model)``."""
    thresholds = thresholds or {}
    rows, scores = [], {}
    for key in [k for k in MODEL_ORDER if k in pipelines]:
        score = attack_scores(pipelines[key], X)
        scores[key] = score
        threshold = thresholds.get(key, config.DEFAULT_THRESHOLD)
        row = {"model": key, "display_name": DISPLAY_NAMES[key]}
        row.update(compute_metrics(y, score, threshold))
        rows.append(row)
    return pd.DataFrame(rows).set_index("model"), scores


# --------------------------------------------------------------------------- #
# Threshold analysis (validation split only)
# --------------------------------------------------------------------------- #
#: Relative cost of a missed intrusion versus a false alert.
#:
#: This is an ASSUMPTION, stated openly, not a measurement. The reasoning: a
#: false positive costs a few minutes of analyst triage; a missed intrusion can
#: cost a breach. Published incident-cost studies put that ratio in the hundreds
#: or thousands, but a detector tuned at that ratio alerts on everything, so
#: 20:1 is used as a deliberately conservative working figure and the
#: sensitivity of the recommended threshold to it is reported alongside.
FN_TO_FP_COST_RATIO: float = 20.0

#: Ceiling on the share of benign traffic that may be alerted on. A SOC seeing
#: 10 million benign flows a day cannot absorb a 1% FPR (100,000 alerts), so the
#: constrained operating point uses a far tighter budget.
MAX_ACCEPTABLE_FPR: float = 0.01


def threshold_analysis(
    y_val: pd.Series,
    scores: np.ndarray,
    cost_ratio: float = FN_TO_FP_COST_RATIO,
    max_fpr: float = MAX_ACCEPTABLE_FPR,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Sweep the decision threshold on the validation split and recommend one.

    Three candidate operating points are computed:

    ``max_f1``
        The threshold maximising F1 - the balanced default.
    ``min_cost``
        The threshold minimising ``cost_ratio * FN + FP``, i.e. the point that
        is optimal if a missed intrusion really is ``cost_ratio`` times worse
        than a false alert.
    ``fpr_constrained``
        The threshold maximising recall subject to ``FPR <= max_fpr`` - the
        point a SOC with a fixed alert budget would actually run.
    """
    y_true = np.asarray(y_val).astype(int)
    grid = np.unique(np.concatenate([
        np.linspace(0.01, 0.99, 197), [config.DEFAULT_THRESHOLD]]))

    rows = []
    for threshold in grid:
        metrics = compute_metrics(y_true, scores, threshold, include_probabilistic=False)
        metrics["cost"] = cost_ratio * metrics["fn"] + metrics["fp"]
        rows.append(metrics)
    sweep = pd.DataFrame(rows)

    best_f1 = sweep.loc[sweep["f1"].idxmax()]
    min_cost = sweep.loc[sweep["cost"].idxmin()]
    feasible = sweep[sweep["false_positive_rate"] <= max_fpr]
    constrained = (feasible.loc[feasible["recall"].idxmax()]
                   if not feasible.empty else None)

    recommendation = {
        "cost_ratio_assumed": cost_ratio,
        "max_fpr_budget": max_fpr,
        "max_f1": {"threshold": float(best_f1["threshold"]),
                   "recall": float(best_f1["recall"]),
                   "precision": float(best_f1["precision"]),
                   "f1": float(best_f1["f1"]),
                   "false_positive_rate": float(best_f1["false_positive_rate"]),
                   "false_negative_rate": float(best_f1["false_negative_rate"])},
        "min_cost": {"threshold": float(min_cost["threshold"]),
                     "recall": float(min_cost["recall"]),
                     "precision": float(min_cost["precision"]),
                     "f1": float(min_cost["f1"]),
                     "false_positive_rate": float(min_cost["false_positive_rate"]),
                     "false_negative_rate": float(min_cost["false_negative_rate"])},
    }
    if constrained is not None:
        recommendation["fpr_constrained"] = {
            "threshold": float(constrained["threshold"]),
            "recall": float(constrained["recall"]),
            "precision": float(constrained["precision"]),
            "f1": float(constrained["f1"]),
            "false_positive_rate": float(constrained["false_positive_rate"]),
            "false_negative_rate": float(constrained["false_negative_rate"])}

    # Sensitivity: how much does the cost-optimal threshold move with the ratio?
    sensitivity = {}
    for ratio in (5, 10, 20, 50, 100):
        cost = ratio * sweep["fn"] + sweep["fp"]
        row = sweep.loc[cost.idxmin()]
        sensitivity[f"ratio_{ratio}"] = {
            "threshold": float(row["threshold"]),
            "recall": float(row["recall"]),
            "false_positive_rate": float(row["false_positive_rate"]),
        }
    recommendation["cost_ratio_sensitivity"] = sensitivity
    return sweep, recommendation


# --------------------------------------------------------------------------- #
# Subgroup / operational audit
# --------------------------------------------------------------------------- #
def subgroup_audit(
    frame: pd.DataFrame,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
    group_columns: tuple[str, ...] = config.AUDIT_GROUPS,
    min_group_size: int = 30,
) -> pd.DataFrame:
    """
    Per-group precision / recall / FPR / FNR for every operational stratum.

    These groups - protocol, application service, connection state, attack
    family - are *operational* strata. They are not demographic protected
    attributes, and this dataset contains none: UNSW-NB15 is synthetic network
    telemetry with no human subjects, no demographic fields and no way to infer
    them. See reports/Bias_Fairness_Analysis.md for what that does and does not
    let this audit conclude.

    For the ``attack_cat`` grouping, benign records are excluded from the
    per-family rows because a family row answers "of the Exploits flows, how
    many did we catch?" - a recall question with no meaningful FPR. Precision
    and FPR are therefore reported as NaN for those rows.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (scores >= threshold).astype(int)
    records = []

    for column in group_columns:
        if column not in frame.columns:
            continue
        is_family = column == config.ATTACK_CAT
        for value, index in frame.groupby(column, observed=True).groups.items():
            positions = frame.index.get_indexer(index)
            if len(positions) < min_group_size:
                continue
            gt, gp = y_true[positions], y_pred[positions]
            tn, fp, fn, tp = confusion_matrix(gt, gp, labels=[0, 1]).ravel()

            record = {
                "group_type": column,
                "group": str(value),
                "n": int(len(positions)),
                "n_attack": int(gt.sum()),
                "n_benign": int((gt == 0).sum()),
                "recall": float(tp / (tp + fn)) if (tp + fn) else np.nan,
                "false_negative_rate": float(fn / (tp + fn)) if (tp + fn) else np.nan,
                "precision": np.nan if is_family else (
                    float(tp / (tp + fp)) if (tp + fp) else np.nan),
                "false_positive_rate": np.nan if is_family else (
                    float(fp / (fp + tn)) if (fp + tn) else np.nan),
                "f1": np.nan if is_family else f1_score(gt, gp, zero_division=0),
                "alert_rate": float(gp.mean()),
            }
            if not is_family and len(np.unique(gt)) > 1:
                record["roc_auc"] = float(roc_auc_score(gt, scores[positions]))
            else:
                record["roc_auc"] = np.nan
            records.append(record)

    audit = pd.DataFrame(records)
    return audit.sort_values(["group_type", "n"], ascending=[True, False]).reset_index(drop=True)


def error_analysis(
    frame: pd.DataFrame,
    y_true: pd.Series,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Characterise what the model gets wrong and where the errors concentrate."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (scores >= threshold).astype(int)
    work = frame.copy()
    work["_y"], work["_pred"], work["_score"] = y_true, y_pred, scores
    work["_outcome"] = np.select(
        [(y_true == 1) & (y_pred == 1), (y_true == 0) & (y_pred == 0),
         (y_true == 0) & (y_pred == 1), (y_true == 1) & (y_pred == 0)],
        ["true_positive", "true_negative", "false_positive", "false_negative"],
        default="unknown")

    false_negatives = work[work["_outcome"] == "false_negative"]
    false_positives = work[work["_outcome"] == "false_positive"]

    out: dict[str, Any] = {
        "threshold": float(threshold),
        "counts": work["_outcome"].value_counts().to_dict(),
        "false_negatives_by_attack_cat": (
            false_negatives[config.ATTACK_CAT].value_counts().to_dict()),
        "false_negative_rate_by_attack_cat": (
            work[work["_y"] == 1]
            .groupby(config.ATTACK_CAT, observed=True)
            .apply(lambda g: float((g["_pred"] == 0).mean()), include_groups=False)
            .round(4).to_dict()),
        "false_positives_by_service": (
            false_positives["service"].value_counts().to_dict()),
        "false_positive_rate_by_service": (
            work[work["_y"] == 0]
            .groupby("service", observed=True)
            .apply(lambda g: float((g["_pred"] == 1).mean()), include_groups=False)
            .round(4).to_dict()),
        "false_positives_by_proto": (
            false_positives["proto"].value_counts().head(10).to_dict()),
        "false_positives_by_state": (
            false_positives["state"].value_counts().to_dict()),
    }

    # How confident is the model when it is wrong? Confidently-wrong errors are
    # the dangerous kind: they will never be caught by a "review low-confidence
    # alerts" workflow.
    if len(false_negatives):
        out["false_negative_score_stats"] = {
            "mean": round(float(false_negatives["_score"].mean()), 4),
            "median": round(float(false_negatives["_score"].median()), 4),
            "share_below_0.10": round(float((false_negatives["_score"] < 0.10).mean()), 4),
        }
    if len(false_positives):
        out["false_positive_score_stats"] = {
            "mean": round(float(false_positives["_score"].mean()), 4),
            "median": round(float(false_positives["_score"].median()), 4),
            "share_above_0.90": round(float((false_positives["_score"] > 0.90).mean()), 4),
        }

    # Median flow profile of each error type - the raw material for the
    # narrative "what does a missed attack look like?"
    profile_columns = ["dur", "sbytes", "dbytes", "spkts", "dpkts",
                       "sttl", "dttl", "rate", "smean"]
    out["median_profile_by_outcome"] = (
        work.groupby("_outcome", observed=True)[profile_columns]
        .median().round(3).to_dict("index"))
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_confusion_matrices(
    y_true: pd.Series, scores: dict[str, np.ndarray],
    thresholds: dict[str, float], name: str = "fig14_confusion_matrices",
) -> None:
    """Confusion matrix per model, annotated with counts and row percentages."""
    keys = [k for k in MODEL_ORDER if k in scores]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.55 * len(keys), 4.0))
    axes = np.atleast_1d(axes)
    y = np.asarray(y_true).astype(int)

    for ax, key in zip(axes, keys):
        threshold = thresholds.get(key, config.DEFAULT_THRESHOLD)
        cm = confusion_matrix(y, (scores[key] >= threshold).astype(int), labels=[0, 1])
        shares = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n{shares[i, j]:.1%}",
                        ha="center", va="center", fontsize=10,
                        color="white" if shares[i, j] > 0.55 else "#1a1a1a",
                        fontweight="bold" if i != j else "normal")
        ax.set_xticks([0, 1], ["Pred\nbenign", "Pred\nattack"], fontsize=9)
        ax.set_yticks([0, 1], ["True\nbenign", "True\nattack"], fontsize=9)
        ax.set_title(f"{DISPLAY_NAMES[key]}\n(threshold {threshold:.2f})", fontsize=10)
        ax.grid(False)

    fig.suptitle("Confusion matrices on the held-out test split "
                 f"(n={len(y):,}; {int(y.sum()):,} attacks)")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    plotting.caption(fig, (
        "Cells are shaded by row percentage, so the bottom-right cell is recall and the "
        "bottom-left cell is the false-negative count - the intrusions that reached the network "
        "unchallenged. Bold cells are errors."), y=0.02)
    plotting.save(fig, name)


def fig_roc_pr_curves(
    y_true: pd.Series, scores: dict[str, np.ndarray],
    name: str = "fig15_roc_pr_curves",
) -> None:
    """ROC and precision-recall curves for every model on one axis pair."""
    y = np.asarray(y_true).astype(int)
    keys = [k for k in MODEL_ORDER if k in scores]
    colors = plotting.class_colors(len(keys))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    for key, color in zip(keys, colors):
        fpr, tpr, _ = roc_curve(y, scores[key])
        axes[0].plot(fpr, tpr, color=color, linewidth=1.9,
                     label=f"{DISPLAY_NAMES[key]} (AUC={auc(fpr, tpr):.4f})")
        precision, recall, _ = precision_recall_curve(y, scores[key])
        axes[1].plot(recall, precision, color=color, linewidth=1.9,
                     label=f"{DISPLAY_NAMES[key]} (AP={average_precision_score(y, scores[key]):.4f})")

    axes[0].plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1.2,
                 label="Random")
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate (attack recall)")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right", fontsize=8.5)
    axes[0].set_xlim(-0.01, 1.01)
    axes[0].set_ylim(-0.01, 1.01)

    axes[1].axhline(y.mean(), color="#999999", linestyle=":", linewidth=1.2,
                    label=f"Base rate ({y.mean():.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall")
    axes[1].legend(loc="lower left", fontsize=8.5)
    axes[1].set_xlim(-0.01, 1.01)
    axes[1].set_ylim(-0.01, 1.01)
    for ax in axes:
        plotting.despine(ax)

    fig.suptitle(f"Threshold-free ranking quality on the test split (n={len(y):,})")
    plotting.caption(fig, (
        "The PR curve is the more informative of the two here. ROC uses the false-positive RATE, "
        "which stays flattering when negatives are plentiful; precision responds directly to the "
        "alert volume an analyst must work through. Both are computed on this corpus's 44% attack "
        "base rate - precision, unlike recall and FPR, will fall sharply at a realistic "
        "production base rate."))
    plotting.save(fig, name)


def fig_model_comparison(table: pd.DataFrame, name: str = "fig16_model_comparison") -> None:
    """Grouped bar chart of the headline metrics plus error rates and latency."""
    keys = [k for k in MODEL_ORDER if k in table.index]
    labels = [DISPLAY_NAMES[k] for k in keys]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    metrics = ["recall", "precision", "f1", "roc_auc", "pr_auc"]
    x = np.arange(len(metrics))
    width = 0.8 / len(keys)
    colors = plotting.class_colors(len(keys))
    for offset, (key, color) in enumerate(zip(keys, colors)):
        values = [table.loc[key, m] for m in metrics]
        axes[0].bar(x + offset * width - 0.4 + width / 2, values,
                    width=width * 0.92, color=color, label=DISPLAY_NAMES[key])
    axes[0].set_xticks(x, ["Recall", "Precision", "F1", "ROC-AUC", "PR-AUC"], fontsize=9)
    axes[0].set_ylim(0, 1.06)
    axes[0].set_ylabel("Score")
    axes[0].set_title("Detection quality (higher is better)")
    axes[0].legend(fontsize=8, ncol=2, loc="lower left")
    plotting.despine(axes[0])

    error_metrics = ["false_negative_rate", "false_positive_rate"]
    x2 = np.arange(len(error_metrics))
    for offset, (key, color) in enumerate(zip(keys, colors)):
        values = [table.loc[key, m] * 100 for m in error_metrics]
        bars = axes[1].bar(x2 + offset * width - 0.4 + width / 2, values,
                           width=width * 0.92, color=color)
        plotting.annotate_bars(axes[1], bars, [f"{v:.1f}" for v in values], fontsize=7)
    axes[1].set_xticks(x2, ["False-negative rate\n(missed attacks)",
                            "False-positive rate\n(analyst workload)"], fontsize=9)
    axes[1].set_ylabel("Rate (%)")
    axes[1].set_title("Operational error rates (lower is better)")
    plotting.despine(axes[1])

    throughput = [table.loc[key, "throughput_flows_per_second"]
                  if "throughput_flows_per_second" in table.columns else np.nan
                  for key in keys]
    bars = axes[2].barh(labels[::-1], throughput[::-1], color=colors[::-1], height=0.62)
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Flows scored per second (log scale)")
    axes[2].set_title("Inference throughput")
    plotting.annotate_bars(axes[2], bars, [f"{v:,.0f}/s" for v in throughput[::-1]],
                           horizontal=True, pad=0.02, fontsize=8)
    plotting.despine(axes[2])

    fig.suptitle("Model comparison on the held-out test split")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    plotting.caption(fig, (
        "Accuracy is deliberately absent from the left panel: on a 44%-positive corpus it "
        "compresses every model into a narrow band and hides the recall/precision trade-off that "
        "actually decides deployment. The middle panel is the one a SOC lead reads."), y=0.02)
    plotting.save(fig, name)


def fig_threshold_analysis(
    sweep: pd.DataFrame, recommendation: dict[str, Any], model_name: str,
    name: str = "fig17_threshold_analysis",
) -> None:
    """Threshold trade-off curves with the three candidate operating points."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))

    axes[0].plot(sweep["threshold"], sweep["recall"], color=config.COLOR_ATTACK,
                 linewidth=2, label="Recall (attacks caught)")
    axes[0].plot(sweep["threshold"], sweep["precision"], color=config.COLOR_BENIGN,
                 linewidth=2, label="Precision (alert quality)")
    axes[0].plot(sweep["threshold"], sweep["f1"], color=config.COLOR_ACCENT,
                 linewidth=2, linestyle="--", label="F1")
    axes[0].set_xlabel("Decision threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / recall trade-off", pad=18)
    axes[0].legend(fontsize=8.5, loc="lower center")
    axes[0].set_ylim(0, 1.03)

    axes[1].plot(sweep["threshold"], sweep["false_negative_rate"] * 100,
                 color=config.COLOR_ATTACK, linewidth=2, label="False-negative rate")
    axes[1].plot(sweep["threshold"], sweep["false_positive_rate"] * 100,
                 color=config.COLOR_BENIGN, linewidth=2, label="False-positive rate")
    axes[1].axhline(recommendation["max_fpr_budget"] * 100, color=config.COLOR_WARN,
                    linestyle=":", linewidth=1.6,
                    label=f"FPR budget ({recommendation['max_fpr_budget']:.0%})")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Decision threshold")
    axes[1].set_ylabel("Error rate (%, log scale)")
    axes[1].set_title("Where the errors go", pad=18)
    axes[1].legend(fontsize=8.5)

    cost = sweep["cost"] / sweep["cost"].min()
    axes[2].plot(sweep["threshold"], cost, color=config.COLOR_NEUTRAL, linewidth=2)
    axes[2].set_xlabel("Decision threshold")
    axes[2].set_ylabel("Relative expected cost (min = 1.0)")
    axes[2].set_title(f"Cost at {recommendation['cost_ratio_assumed']:.0f}:1 FN:FP", pad=18)
    axes[2].set_ylim(0.9, min(float(cost.max()), 12))

    styles = {"max_f1": (config.COLOR_ACCENT, "-.", "F1-optimal"),
              "min_cost": (config.COLOR_NEUTRAL, "--", "cost-optimal"),
              "fpr_constrained": (config.COLOR_WARN, ":", "FPR budget")}
    for key, (color, dash, label) in styles.items():
        if key not in recommendation:
            continue
        threshold = recommendation[key]["threshold"]
        for ax in axes:
            ax.axvline(threshold, color=color, linestyle=dash, linewidth=1.5, alpha=0.9)
        # Labels sit in the padded strip between the axes and the panel title, so
        # they never overlap the curves, the legend, or each other.
        align = "left" if threshold < 0.15 else "right" if threshold > 0.85 else "center"
        axes[0].annotate(f"{label} {threshold:.2f}",
                         xy=(threshold, 1.01), xycoords=("data", "axes fraction"),
                         fontsize=7.5, color=color, ha=align, va="bottom")
    for ax in axes:
        plotting.despine(ax)

    fig.suptitle(f"Threshold selection for {model_name} - validation split only "
                 f"(n={int(sweep['n'].iloc[0]):,})")
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    plotting.caption(fig, (
        "This sweep is computed on the VALIDATION split. Choosing a threshold is a fitting "
        "decision; doing it on the test split would make the reported test metrics optimistic. "
        "The three vertical lines are the candidate operating points - F1-optimal, "
        "cost-optimal under the stated 20:1 assumption, and the tightest point that respects a "
        "1% false-positive budget."), y=0.02)
    plotting.save(fig, name)


def fig_subgroup_audit(
    audit: pd.DataFrame, name: str = "fig18_subgroup_audit",
) -> None:
    """Recall by attack family and false-positive rate by operational stratum."""
    families = audit[audit["group_type"] == config.ATTACK_CAT].copy()
    families = families[families["n_attack"] > 0].sort_values("recall")
    others = audit[audit["group_type"].isin(["service", "proto", "state"])].copy()
    others = others[others["n_benign"] >= 30].sort_values("false_positive_rate",
                                                          ascending=False).head(14)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))

    colors = [plotting.rate_to_color(1 - r) for r in families["recall"]]
    bars = axes[0].barh(families["group"], families["recall"] * 100,
                        color=colors, edgecolor="#666666", linewidth=0.5, height=0.7)
    axes[0].set_xlim(0, 118)
    axes[0].set_xlabel("Recall (% of that family's flows detected)")
    axes[0].set_title("Detection rate by attack family")
    plotting.annotate_bars(
        axes[0], bars,
        [f"{r:.1%} (n={int(n):,})" for r, n in zip(families["recall"], families["n_attack"])],
        horizontal=True, pad=0.012, fontsize=8)
    plotting.despine(axes[0])

    colors2 = [plotting.rate_to_color(min(r * 8, 1.0)) for r in others["false_positive_rate"]]
    labels = [f"{t}: {g}" for t, g in zip(others["group_type"], others["group"])]
    bars2 = axes[1].barh(labels[::-1], (others["false_positive_rate"] * 100)[::-1],
                         color=colors2[::-1], edgecolor="#666666", linewidth=0.5, height=0.7)
    axes[1].set_xlabel("False-positive rate (% of benign flows alerted)")
    axes[1].set_title("Where false alerts concentrate")
    plotting.annotate_bars(
        axes[1], bars2,
        [f"{r:.1%} ({int(n):,} benign)" for r, n in
         zip(others["false_positive_rate"][::-1], others["n_benign"][::-1])],
        horizontal=True, pad=0.02, fontsize=8)
    plotting.despine(axes[1])

    # Concentrate the story into measured numbers rather than adjectives.
    missed = (families["false_negative_rate"] * families["n_attack"]).round()
    total_missed = float(missed.sum())
    worst_two = families.nsmallest(2, "recall")
    worst_share = (float((worst_two["false_negative_rate"] * worst_two["n_attack"]).sum())
                   / total_missed if total_missed else float("nan"))
    spread = float(families["recall"].max() - families["recall"].min())

    top_fp = others.iloc[0] if len(others) else None
    fp_line = (f"`{top_fp['group_type']}: {top_fp['group']}` alone alerts on "
               f"{top_fp['false_positive_rate']:.1%} of its "
               f"{int(top_fp['n_benign']):,} benign flows. " if top_fp is not None else "")

    fig.suptitle("Operational subgroup audit on the test split "
                 "(operational strata, not demographic groups)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    plotting.caption(fig, (
        f"Left: recall spans {spread:.0%} across families, from "
        f"{worst_two.iloc[0]['recall']:.1%} on {worst_two.iloc[0]['group']} to 100%. The two "
        f"weakest families account for {worst_share:.0%} of every attack the model missed, so "
        "the aggregate recall figure conceals a highly concentrated failure. "
        f"Right: false alerts are not spread evenly either - {fp_line}That concentration is "
        "directly actionable through per-service thresholds or suppression rules, and it is also "
        "a warning: a stratum with an elevated false-positive rate is one where analysts learn to "
        "dismiss alerts. UNSW-NB15 contains no demographic attributes, so this is an operational "
        "performance audit and NOT a demographic fairness assessment; no conclusion about "
        "protected groups can or should be drawn from it."),
        y=0.025)
    plotting.save(fig, name)


def fig_ablation_comparison(
    ablations: dict[str, pd.DataFrame], model_key: str = "xgboost",
    name: str = "fig19_ablation_comparison",
) -> None:
    """Compare the headline model across experimental conditions."""
    labels, recalls, precisions, f1s, prs = [], [], [], [], []
    by_key: dict[str, pd.Series] = {}
    pretty = {
        "main": "Primary protocol",
        "keep_duplicates": "Duplicates retained",
        "no_ttl": "TTL features removed",
        "no_engineered": "No engineered features",
        "smote": "SMOTE instead of weights",
        "official_split": "Authors' published split",
    }
    for key, table in ablations.items():
        if model_key not in table.index:
            continue
        labels.append(pretty.get(key, key))
        by_key[key] = table.loc[model_key]
        recalls.append(table.loc[model_key, "recall"])
        precisions.append(table.loc[model_key, "precision"])
        f1s.append(table.loc[model_key, "f1"])
        prs.append(table.loc[model_key, "pr_auc"])

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    series = [("Recall", recalls, config.COLOR_ATTACK),
              ("Precision", precisions, config.COLOR_BENIGN),
              ("F1", f1s, config.COLOR_ACCENT),
              ("PR-AUC", prs, config.COLOR_NEUTRAL)]
    width = 0.2
    for i, (label, values, color) in enumerate(series):
        bars = ax.bar(x + i * width - 1.5 * width, values, width=width * 0.92,
                      color=color, label=label)
        plotting.annotate_bars(ax, bars, [f"{v:.3f}" for v in values], fontsize=7)
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("Score on that experiment's test split")
    ax.set_title(f"{DISPLAY_NAMES[model_key]} under each experimental condition")
    # Below the axes: a legend inside the plotting area would sit on top of the bars.
    ax.legend(ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09),
              frameon=False)
    plotting.despine(ax)
    plotting.caption(fig, _ablation_caption(by_key))
    plotting.save(fig, name)


def _ablation_caption(by_key: dict[str, pd.Series]) -> str:
    """Caption stating what the ablations measured, computed from the results."""
    base = by_key.get("main")
    if base is None:
        return ("Each bar group is a separate end-to-end experiment with its own split, so the "
                "bars are comparable in interpretation but not paired observation-for-observation.")
    parts = ["Each bar group is a separate end-to-end experiment with its own split."]

    dup = by_key.get("keep_duplicates")
    if dup is not None:
        dup_share = 1.0 - float(base["n"]) / float(dup["n"])
        parts.append(
            f"Leaving the {dup_share:.1%} duplicated records in raises F1 from {base['f1']:.3f} to "
            f"{dup['f1']:.3f} (+{dup['f1'] - base['f1']:.3f}) without the model having learned "
            "anything more - that gap is the size of the duplicate-leakage inflation present in "
            "benchmarks that skip deduplication.")

    ttl = by_key.get("no_ttl")
    if ttl is not None:
        delta = ttl["f1"] - base["f1"]
        direction = "unchanged" if abs(delta) < 0.005 else ("higher" if delta > 0 else "lower")
        parts.append(
            f"Dropping the TTL features leaves F1 {direction} at {ttl['f1']:.3f} "
            f"({delta:+.4f}), so although SHAP attributes most of the model's signal to `sttl`, "
            "the remaining features carry near-equivalent information and the capture artefact is "
            "not load-bearing for the boosted-tree result.")

    eng = by_key.get("no_engineered")
    if eng is not None:
        parts.append(
            f"The 15 engineered features are likewise not decisive ({eng['f1']:.3f} without them, "
            f"{eng['f1'] - base['f1']:+.4f}); they buy interpretability, not accuracy.")

    off = by_key.get("official_split")
    if off is not None:
        parts.append(
            f"On the authors' published split, precision collapses to {off['precision']:.3f} "
            f"(FPR {off['false_positive_rate']:.1%}) while recall rises to {off['recall']:.3f} - "
            "the same model, re-partitioned, is a materially different detector, which is the "
            "clearest evidence here that these numbers do not transfer unchanged to a new network.")
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def save_table(table: pd.DataFrame, stem: str) -> None:
    """Write a results table to ``reports/metrics/`` as CSV."""
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.METRICS_DIR / f"{stem}.csv"
    table.to_csv(path)
    print(f"[evaluate] wrote reports/metrics/{path.name}")


def save_json(payload: Any, stem: str) -> None:
    config.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.METRICS_DIR / f"{stem}.json"
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    print(f"[evaluate] wrote reports/metrics/{path.name}")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
