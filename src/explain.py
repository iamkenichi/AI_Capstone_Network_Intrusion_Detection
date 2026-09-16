"""
SHAP explainability for the selected tree ensemble.

What this module answers
------------------------
* Globally - which features drive the model's notion of "malicious", and by how
  much?
* Directionally - which feature *values* push a flow toward ATTACK and which
  push it toward BENIGN?
* Locally - for one specific flow, why did the model decide what it decided?
  Four cases are explained in depth: a correctly detected attack, a correctly
  cleared benign flow, a false positive and a false negative.
* Diagnostically - is any single feature dominating in a way that suggests the
  model is exploiting a capture artefact rather than learning attack behaviour?

The last question is the important one for this dataset. The EDA established
that ``sttl`` is close to a label proxy; SHAP is where that suspicion is either
confirmed or dismissed on the fitted model itself rather than on univariate
correlations.

Local explanations are computed on the **test** split, which is legitimate: by
the time this module runs the model and threshold are frozen, and nothing here
feeds back into either. Explaining a model is not fitting it.
"""

from __future__ import annotations

import json
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config, plotting, preprocessing
from src.features import ENGINEERED_FEATURE_DOCS

#: Rows sampled for the global SHAP computation. Exact TreeSHAP is
#: polynomial in tree depth, so a stratified sample keeps the runtime to
#: minutes while leaving the global ranking statistically stable.
GLOBAL_SAMPLE_SIZE = 4000


class ExplanationUnavailable(RuntimeError):
    """Raised when SHAP cannot explain the supplied estimator."""


# --------------------------------------------------------------------------- #
# Core computation
# --------------------------------------------------------------------------- #
def design_matrix(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Apply the fitted pipeline's transforms and return a *named* design matrix."""
    engineered = pipeline.named_steps["engineer"].transform(X)
    transformed = pipeline.named_steps["preprocess"].transform(engineered)
    names = preprocessing.transformed_feature_names(pipeline)
    return pd.DataFrame(np.asarray(transformed), columns=names, index=X.index)


def compute_shap(pipeline, X: pd.DataFrame, sample_size: int | None = None):
    """
    Return ``(explanation, design)`` for the positive (attack) class.

    scikit-learn and XGBoost disagree on SHAP output shape for binary
    classifiers: XGBoost emits ``(n, features)`` for the single logit, while
    RandomForest emits ``(n, features, 2)`` for the two class probabilities.
    Both are normalised here to the attack class so downstream plots and
    rankings are directly comparable across model families.
    """
    try:
        import shap
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ExplanationUnavailable(
            "shap is not installed. Run `pip install -r requirements.txt`.") from exc

    estimator = pipeline.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        raise ExplanationUnavailable(
            f"TreeExplainer needs a tree ensemble; got {type(estimator).__name__}.")

    if sample_size and len(X) > sample_size:
        X = X.sample(n=sample_size, random_state=config.RANDOM_STATE)

    design = design_matrix(pipeline, X)
    explainer = shap.TreeExplainer(estimator)
    explanation = explainer(design, check_additivity=False)

    if explanation.values.ndim == 3:
        explanation = explanation[..., -1]   # keep the attack class
    return explanation, design


def global_importance(explanation, top_n: int = 25) -> pd.DataFrame:
    """Mean |SHAP| per feature, plus the signed mean (direction of influence)."""
    values = np.asarray(explanation.values)
    frame = pd.DataFrame({
        "feature": list(explanation.feature_names),
        "mean_abs_shap": np.abs(values).mean(axis=0),
        "mean_shap": values.mean(axis=0),
        "max_abs_shap": np.abs(values).max(axis=0),
    })
    frame["share_of_total"] = frame["mean_abs_shap"] / frame["mean_abs_shap"].sum()
    return (frame.sort_values("mean_abs_shap", ascending=False)
            .head(top_n).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Global figures
# --------------------------------------------------------------------------- #
def fig_shap_bar(importance: pd.DataFrame, model_name: str,
                 name: str = "fig20_shap_global_importance") -> None:
    """Mean |SHAP| bar chart, coloured by the direction of average influence."""
    plotting.use_house_style()
    top = importance.head(20).iloc[::-1]
    colors = [config.COLOR_ATTACK if v > 0 else config.COLOR_BENIGN
              for v in top["mean_shap"]]

    fig, ax = plt.subplots(figsize=(10, 7.5))
    bars = ax.barh(top["feature"], top["mean_abs_shap"], color=colors, height=0.72)
    ax.set_xlim(0, top["mean_abs_shap"].max() * 1.26)
    ax.set_xlabel("Mean |SHAP value| (average impact on the model's attack score)")
    plotting.annotate_bars(
        ax, bars,
        [f"{v:.3f}  ({s:.1%})" for v, s in zip(top["mean_abs_shap"], top["share_of_total"])],
        horizontal=True, pad=0.012, fontsize=8)
    ax.set_title(f"Global feature importance - {model_name}")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=config.COLOR_ATTACK, label="On average pushes toward ATTACK"),
        Patch(color=config.COLOR_BENIGN, label="On average pushes toward BENIGN")],
        loc="lower right")
    plotting.despine(ax)

    top3 = importance.head(3)
    plotting.caption(fig, (
        f"The top three features ({', '.join(top3['feature'])}) account for "
        f"{top3['share_of_total'].sum():.0%} of total attributed impact. SHAP is used here in "
        "preference to impurity-based importance, which is biased toward high-cardinality "
        "features and splits credit arbitrarily between the correlated twins identified in fig08."
    ))
    plotting.save(fig, name)


def fig_shap_beeswarm(explanation, model_name: str,
                      name: str = "fig21_shap_beeswarm") -> None:
    """SHAP beeswarm: distribution and direction of every feature's effect."""
    import shap

    plt.figure(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP beeswarm - {model_name}", fontsize=12, fontweight="bold", y=1.0)
    plotting.caption(fig, (
        "Each dot is one test flow. Horizontal position is that feature's contribution to the "
        "attack score for that flow; colour is the feature's own (standardised) value. A feature "
        "whose red points sit on the right means HIGH values of it argue for 'attack'. The width "
        "of each row shows how much the feature's influence varies between flows - a narrow row "
        "is a feature the model treats almost identically everywhere."), y=0.0)
    plotting.save(fig, name)


def fig_shap_dependence(explanation, design: pd.DataFrame, importance: pd.DataFrame,
                        model_name: str, name: str = "fig22_shap_dependence") -> None:
    """Dependence plots for the strongest features."""
    plotting.use_house_style()
    features = importance.head(6)["feature"].tolist()
    values = np.asarray(explanation.values)
    names = list(explanation.feature_names)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, feature in zip(axes.ravel(), features):
        idx = names.index(feature)
        x = design[feature].to_numpy(float)
        y = values[:, idx]
        ax.scatter(x, y, s=6, alpha=0.30, color=config.COLOR_NEUTRAL, linewidths=0)
        ax.axhline(0, color="#AAAAAA", linewidth=1.0, linestyle="--")
        ax.set_xlabel(f"{feature} (standardised)", fontsize=9)
        ax.set_ylabel("SHAP value", fontsize=9)
        ax.set_title(feature, fontsize=10)
        plotting.despine(ax)

    fig.suptitle(f"How each top feature's value maps to its effect - {model_name}")
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    plotting.caption(fig, (
        "A clean step shape means the model learned a threshold rule on that feature. Vertical "
        "spread at a fixed x-value is interaction: the same feature value means different things "
        "depending on the rest of the flow. Features are shown on the standardised scale the "
        "model actually sees, so 0 is the training mean."), y=0.02)
    plotting.save(fig, name)


def fig_local_explanations(
    pipeline, X_test: pd.DataFrame, y_test: pd.Series, scores: np.ndarray,
    threshold: float, model_name: str, meta: pd.DataFrame,
    name: str = "fig23_shap_local_cases",
) -> dict[str, Any]:
    """
    Waterfall-style local explanations for one instance of each outcome type.

    Cases are chosen to be *representative rather than extreme*: the median-score
    member of each outcome class, so the explanation describes typical model
    behaviour and not a cherry-picked oddity.
    """
    import shap

    plotting.use_house_style()
    y = np.asarray(y_test).astype(int)
    predicted = (scores >= threshold).astype(int)

    masks = {
        "True positive\n(attack correctly detected)": (y == 1) & (predicted == 1),
        "True negative\n(benign correctly cleared)": (y == 0) & (predicted == 0),
        "False positive\n(benign wrongly alerted)": (y == 0) & (predicted == 1),
        "False negative\n(attack missed)": (y == 1) & (predicted == 0),
    }

    chosen: dict[str, int] = {}
    for title, mask in masks.items():
        positions = np.flatnonzero(mask)
        if positions.size == 0:
            continue
        subset_scores = scores[positions]
        chosen[title] = int(positions[np.argsort(subset_scores)[len(subset_scores) // 2]])

    if not chosen:
        raise ExplanationUnavailable("No cases available to explain.")

    rows = X_test.iloc[list(chosen.values())]
    explanation, design = compute_shap(pipeline, rows, sample_size=None)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    details: dict[str, Any] = {}

    for ax, (title, position) in zip(axes.ravel(), chosen.items()):
        order = list(chosen).index(title)
        single = explanation[order]
        contributions = pd.DataFrame({
            "feature": list(single.feature_names),
            "shap": np.asarray(single.values),
            "value": np.asarray(single.data),
        })
        contributions["abs"] = contributions["shap"].abs()
        top = contributions.sort_values("abs", ascending=False).head(12).iloc[::-1]

        colors = [config.COLOR_ATTACK if v > 0 else config.COLOR_BENIGN for v in top["shap"]]
        ax.barh(top["feature"], top["shap"], color=colors, height=0.7)
        ax.axvline(0, color="#555555", linewidth=1.0)
        ax.set_xlabel("SHAP contribution to the attack score (log-odds)", fontsize=9)
        record = meta.iloc[position]
        ax.set_title(
            f"{title}\nscore={scores[position]:.4f}  |  proto={record['proto']}  "
            f"service={record['service']}  state={record['state']}  "
            f"family={record[config.ATTACK_CAT]}",
            fontsize=9.5)
        plotting.despine(ax)

        details[title.replace("\n", " ")] = {
            "test_row_position": int(position),
            "attack_score": round(float(scores[position]), 6),
            "true_label": int(y[position]),
            "predicted": int(predicted[position]),
            "attack_cat": str(record[config.ATTACK_CAT]),
            "proto": str(record["proto"]),
            "service": str(record["service"]),
            "state": str(record["state"]),
            "base_value": round(float(np.asarray(single.base_values).ravel()[0]), 6),
            "top_contributions": (
                contributions.sort_values("abs", ascending=False).head(8)
                [["feature", "shap", "value"]].round(5).to_dict("records")),
        }

    fig.suptitle(f"Local explanations - {model_name}  "
                 f"(representative case of each outcome, threshold {threshold:.2f})")
    fig.tight_layout(rect=(0, 0.035, 1, 0.95))
    plotting.caption(fig, (
        "Red bars push the flow toward ATTACK, blue bars toward BENIGN; the final score is the "
        "model's base rate plus the sum of all contributions. Comparing the false negative "
        "(bottom right) against the true positive (top left) shows exactly which evidence was "
        "absent in the attack the model let through - which is the raw material for the "
        "detection-gap discussion in the Bias and Fairness analysis."), y=0.018)
    plotting.save(fig, name)
    return details


def fig_ttl_dependence_check(
    explanation, design: pd.DataFrame, model_name: str,
    name: str = "fig24_shap_artifact_check",
) -> dict[str, Any]:
    """
    Quantify how much of the model's attributed signal comes from TTL features.

    This is the diagnostic that decides whether the headline performance should
    be trusted as a measure of *attack detection* or read as a measure of
    *testbed recognition*.
    """
    plotting.use_house_style()
    values = np.abs(np.asarray(explanation.values))
    names = list(explanation.feature_names)
    total = values.mean(axis=0).sum()

    groups = {
        "TTL family\n(sttl, dttl, ct_state_ttl)": list(config.TTL_FEATURES),
        "Connection counters\n(ct_*, excl. ct_state_ttl)": [
            n for n in names if n.startswith("ct_") and n not in config.TTL_FEATURES],
        "Volume & rate\n(bytes, packets, load)": [
            n for n in names if n in {"sbytes", "dbytes", "spkts", "dpkts", "rate",
                                      "sload", "dload", "smean", "dmean",
                                      "flow_bytes_total", "flow_pkts_total",
                                      "bytes_per_packet"}],
        "Engineered in this project": [
            n for n in names if n in ENGINEERED_FEATURE_DOCS],
        "Categorical\n(proto, service, state)": [
            n for n in names
            if n.startswith(("proto_", "service_", "state_"))],
        "Timing & TCP state": [
            n for n in names if n in {"dur", "sinpkt", "dinpkt", "sjit", "djit",
                                      "tcprtt", "synack", "ackdat", "swin", "dwin",
                                      "stcpb", "dtcpb"}],
    }

    shares, labels = [], []
    for label, members in groups.items():
        indices = [names.index(m) for m in members if m in names]
        shares.append(values[:, indices].mean(axis=0).sum() / total if indices else 0.0)
        labels.append(label)

    order = np.argsort(shares)[::-1]
    shares = [shares[i] for i in order]
    labels = [labels[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    colors = [config.COLOR_ATTACK if "TTL family" in label else config.COLOR_NEUTRAL
              for label in labels]
    bars = axes[0].bar(range(len(labels)), [s * 100 for s in shares], color=colors, width=0.62)
    axes[0].set_xticks(range(len(labels)), labels, fontsize=8, rotation=12, ha="right")
    axes[0].set_ylabel("Share of total attributed impact (%)")
    axes[0].set_ylim(0, max(shares) * 130)
    plotting.annotate_bars(axes[0], bars, [f"{s:.1%}" for s in shares], fontsize=8.5)
    axes[0].set_title("Where the model's decisions come from")
    plotting.despine(axes[0])

    ttl_present = [f for f in config.TTL_FEATURES if f in names]
    if ttl_present:
        idx = names.index(ttl_present[0])
        axes[1].scatter(design[ttl_present[0]], np.asarray(explanation.values)[:, idx],
                        s=7, alpha=0.30, color=config.COLOR_ATTACK, linewidths=0)
        axes[1].axhline(0, color="#AAAAAA", linestyle="--", linewidth=1.0)
        axes[1].set_xlabel(f"{ttl_present[0]} (standardised)")
        axes[1].set_ylabel("SHAP contribution")
        axes[1].set_title(f"{ttl_present[0]}: a near-binary switch")
        plotting.despine(axes[1])

    ttl_share = dict(zip(labels, shares)).get("TTL family\n(sttl, dttl, ct_state_ttl)", 0.0)
    fig.suptitle(f"Artefact diagnostic - {model_name}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    plotting.caption(fig, (
        f"The TTL family carries {ttl_share:.0%} of the model's total attributed impact. The "
        "right panel shows why: the relationship is a step, not a gradient - the model has "
        "learned a near-binary switch on a field the UNSW-NB15 testbed happened to set "
        "differently for its benign and attack generators. Any real deployment must expect this "
        "component of the signal to vanish, which is precisely what the no_ttl ablation "
        "measures."), y=0.025)
    plotting.save(fig, name)

    # Cast to built-in float: json.dumps(..., default=str) would otherwise
    # silently stringify numpy scalars, and the reports format these with `:.1%`.
    return {"impact_share_by_group": {label: round(float(share), 4)
                                      for label, share in zip(labels, shares)},
            "ttl_family_share": round(float(ttl_share), 4)}


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(pipeline, X_test: pd.DataFrame, y_test: pd.Series, scores: np.ndarray,
        threshold: float, model_name: str, meta: pd.DataFrame) -> dict[str, Any]:
    """Produce every SHAP artefact and return the numbers used in the reports."""
    plotting.use_house_style()
    print(f"[explain] computing SHAP values for {model_name} "
          f"on {min(GLOBAL_SAMPLE_SIZE, len(X_test)):,} test flows ...")

    explanation, design = compute_shap(pipeline, X_test, sample_size=GLOBAL_SAMPLE_SIZE)
    importance = global_importance(explanation)

    fig_shap_bar(importance, model_name)
    fig_shap_beeswarm(explanation, model_name)
    fig_shap_dependence(explanation, design, importance, model_name)
    artifact = fig_ttl_dependence_check(explanation, design, model_name)
    local = fig_local_explanations(
        pipeline, X_test, y_test, scores, threshold, model_name, meta)

    payload = {
        "model": model_name,
        "sample_size": int(len(design)),
        "global_importance": importance.round(6).to_dict("records"),
        "artifact_diagnostic": artifact,
        "local_cases": local,
    }
    path = config.METRICS_DIR / "shap_analysis.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[explain] wrote reports/metrics/{path.name}")
    return payload
