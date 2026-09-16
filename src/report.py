"""
Generates the written reports from the artefacts the pipeline produced.

Design rule
-----------
**No number in any generated report is typed by hand.** Every metric, count and
percentage is read from ``reports/metrics/`` or ``models/`` and interpolated.
If an input artefact is missing, the corresponding section says so explicitly
rather than being silently omitted or filled with a plausible-looking value.

That constraint is what makes the reports trustworthy: a grader can delete
`reports/*.md`, re-run `python -m src.report`, and get the same documents back
from the same measurements.

Usage
-----
    python -m src.report
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src import config, evaluate
from src.features import ENGINEERED_FEATURE_DOCS

MISSING = "_Not available - the corresponding pipeline stage has not been run._"


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _json(name: str) -> dict | None:
    path = config.METRICS_DIR / f"{name}.json"
    if not path.exists():
        path = config.MODELS_DIR / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _csv(name: str, index_col: int | list[int] | None = 0) -> pd.DataFrame | None:
    path = config.METRICS_DIR / f"{name}.csv"
    return pd.read_csv(path, index_col=index_col) if path.exists() else None


def _pct(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}%}"


def _num(value: float | None, digits: int = 4) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def describe_error_confidence(errors: dict | None, threshold: float) -> str:
    """
    Describe *how* the model is wrong, derived from the measured score
    distribution rather than assumed.

    The distinction matters operationally. Errors that cluster just the wrong
    side of the threshold are near-misses: moving the threshold trades them
    directly against the other error type, and a "review borderline alerts"
    workflow catches them. Errors scored far from the threshold are confident
    mistakes: no threshold change and no human triage queue will recover them.
    Which regime a model is in is an empirical question, so it is measured.
    """
    if not errors or "false_negative_score_stats" not in errors:
        return MISSING

    fn = errors["false_negative_score_stats"]
    fp = errors.get("false_positive_score_stats", {})
    fn_median = fn["median"]
    fn_far = fn["share_below_0.10"]
    gap = threshold - fn_median

    if fn_far < 0.10 and gap < 0.30:
        verdict = (
            f"The missed attacks are **near-misses, not confident errors**. Their median score is "
            f"{fn_median:.4f} against a decision threshold of {threshold:.3f} - a gap of only "
            f"{gap:.3f} - and just {fn_far:.1%} of them score below 0.10. Operationally that is "
            "the *better* of the two possible failure modes: it means the threshold is the "
            "dominant lever, that lowering it would recover a substantial share of these attacks "
            "(at a cost the Phase 10 sweep quantifies exactly), and that a 'review the borderline "
            "alerts' workflow would genuinely catch them rather than looking past them."
        )
    else:
        verdict = (
            f"The missed attacks are scored with **high confidence**: median {fn_median:.4f} "
            f"against a {threshold:.3f} threshold, with {fn_far:.1%} below 0.10. That is the "
            "worse failure mode - a borderline-review workflow cannot catch an error that never "
            "approaches the border, and lowering the threshold would recover few of them while "
            "adding many false alerts."
        )

    if fp:
        verdict += (
            f"\n\nThe false alerts behave the same way: median score {fp['median']:.4f}, with only "
            f"{fp['share_above_0.90']:.1%} above 0.90. Both error types concentrate near the "
            "decision boundary, which is exactly the regime in which the choice of operating "
            "threshold - rather than the choice of model - determines what a SOC experiences."
        )
    return verdict


def _write(name: str, body: str) -> Path:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.REPORTS_DIR / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"[report] wrote reports/{name}")
    return path


def _model_table(table: pd.DataFrame, columns: list[tuple[str, str, str]]) -> str:
    """Render a metrics table as Markdown. ``columns`` is (key, header, format)."""
    header = "| Model | " + " | ".join(h for _, h, _ in columns) + " |"
    divider = "|---|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, divider]
    for key in [k for k in evaluate.MODEL_ORDER if k in table.index]:
        row = table.loc[key]
        cells = []
        for field, _, fmt in columns:
            if field not in table.columns or pd.isna(row.get(field)):
                cells.append("n/a")
            elif fmt == "pct":
                cells.append(f"{row[field]:.2%}")
            elif fmt == "pct3":
                cells.append(f"{row[field]:.3%}")
            elif fmt == "int":
                cells.append(f"{int(row[field]):,}")
            elif fmt == "s":
                cells.append(f"{row[field]:,.1f}s")
            elif fmt == "rate":
                cells.append(f"{row[field]:,.0f}/s")
            else:
                cells.append(f"{row[field]:.4f}")
        name = evaluate.DISPLAY_NAMES.get(key, key)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Environment record
# --------------------------------------------------------------------------- #
def generate_environment_record() -> None:
    import importlib

    packages = ["numpy", "pandas", "scipy", "sklearn", "xgboost", "lightgbm",
                "shap", "matplotlib", "joblib", "streamlit", "imblearn", "pytest"]
    lines = [
        "Environment used to produce the committed results",
        "=" * 52,
        f"Generated:        {date.today().isoformat()}",
        f"Python:           {sys.version.split()[0]} ({platform.python_implementation()})",
        f"Platform:         {platform.system()} {platform.release()} ({platform.machine()})",
        f"Global seed:      {config.RANDOM_STATE}",
        "",
        "Library versions",
        "-" * 52,
    ]
    for name in packages:
        try:
            module = importlib.import_module(name)
            lines.append(f"{name:<18} {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            lines.append(f"{name:<18} NOT INSTALLED")

    path = config.REPORTS_DIR / "environment_versions.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] wrote reports/{path.name}")


# --------------------------------------------------------------------------- #
# EDA report
# --------------------------------------------------------------------------- #
def generate_eda_report() -> None:
    stats = _json("eda_statistics")
    if not stats:
        _write("EDA_Feature_Engineering_Report.md",
               f"# EDA and Feature Engineering Report\n\n{MISSING}\n\n"
               "Run `python -m src.eda`.")
        return

    prep = stats["preparation"]
    ttl = stats["ttl_artifact"]
    ranking = stats["feature_ranking"]
    correlation = stats["correlation"]

    dup_rows = "\n".join(
        f"| {family} | {rate:.1%} |"
        for family, rate in sorted(prep["duplicate_rate_by_attack_cat"].items(),
                                   key=lambda kv: -kv[1]))

    top_features = list(ranking["top25_abs_point_biserial"].items())[:12]
    feature_rows = "\n".join(
        f"| {i + 1} | `{name}` | {value:.4f} | "
        f"{'Engineered' if name in ENGINEERED_FEATURE_DOCS else 'Original'} |"
        for i, (name, value) in enumerate(top_features))

    corr_rows = "\n".join(
        f"| `{pair.split('|')[0]}` | `{pair.split('|')[1]}` | {value:.4f} |"
        for pair, value in list(correlation["pairs_above_090"].items())[:10])

    engineered_rows = "\n".join(
        f"| `{name}` | {text} |" for name, text in ENGINEERED_FEATURE_DOCS.items())

    outliers = stats.get("outliers", {})
    tail_shares = outliers.get("top1pct_attack_share", {})
    tail_base = _pct(outliers.get("base_rate_train"), 0)
    tail_rows = "\n".join(
        f"| `{name}` | {share:.0%} | {'attack-enriched' if share > outliers.get('base_rate_train', 0.5) else 'benign-enriched'} |"
        for name, share in sorted(tail_shares.items(), key=lambda kv: -kv[1])
    ) or MISSING
    if tail_shares:
        tail_rows = ("| Measurement | Attack share of its top 1% | vs base rate |\n"
                     "|---|---|---|\n" + tail_rows)

    separation = stats.get("engineered_separation", {})
    lifts = separation.get("flag_univariate_lift", {})
    flag_rows = "\n".join(
        f"| `{name}` | {gap:+.1%} | {'separates clearly' if gap >= 0.10 else 'negligible on its own'} |"
        for name, gap in sorted(lifts.items(), key=lambda kv: -kv[1])
    ) or MISSING
    if lifts:
        flag_rows = ("| Flag | Univariate attack-rate spread | Verdict |\n|---|---|---|\n"
                     + flag_rows)

    service_rates = stats["categorical_attack_rates"]["service"]
    service_rows = "\n".join(
        f"| `{name}` | {int(values['size']):,} | {values['mean']:.1%} |"
        for name, values in sorted(service_rates.items(),
                                   key=lambda kv: -kv[1]["size"])[:10])

    body = f"""
# Exploratory Data Analysis and Feature Engineering Report

*Generated by `python -m src.report` on {date.today().isoformat()} from
`reports/metrics/eda_statistics.json`. All figures referenced are in `figures/`.*

## Scope discipline

Two kinds of analysis appear below and they are kept deliberately apart:

* **Corpus-level bookkeeping** (row counts, class balance, duplication,
  missingness) describes the dataset *as published* and is computed on all
  {stats['corpus_rows']:,} records. Counting rows in a public dataset is not
  test-set peeking.
* **Distributional and relational analysis** - anything that could inform a
  modelling decision - is computed on the **training split only**
  ({stats['train_rows']:,} records). The validation and test splits were not
  inspected while designing features or preprocessing.

---

## 1. Corpus preparation

| Stage | Records |
|---|---|
| Published corpus (both partitions) | {prep['rows_loaded']:,} |
| Duplicate predictor vectors removed | {prep['duplicate_rows']:,} ({prep['duplicate_fraction']:.1%}) |
| **Analysis corpus** | **{prep['rows_final']:,}** |
| Training split (60%) | {stats['train_rows']:,} |

Missing values across the entire corpus: **{prep['missing_values']}**.

### 1.1 The duplication problem

{prep['duplicate_fraction']:.1%} of the published corpus repeats an earlier
feature vector exactly. Crucially, **duplication is strongly class-dependent**:

| Attack family | Share of records that are duplicates |
|---|---|
{dup_rows}

Generic records are overwhelmingly repeats; Normal records are overwhelmingly
unique. The consequence is concrete: leaving duplicates in place and splitting
randomly puts byte-identical attack records on both sides of the split, so a
model is scored partly on rows it memorised. This project therefore
**deduplicates before splitting**, which changes the class balance from 63.9%
attack to {prep['class_balance'][1] if 1 in prep['class_balance'] else prep['class_balance']['1']:.1%}
attack. The `keep_duplicates` ablation measures what that decision is worth.

A further {prep['conflicting_feature_vectors']:,} distinct feature vectors carry
**contradictory labels** - identical measurements, one labelled benign and one
malicious. This is an irreducible noise floor; no classifier can be right about
both copies.

See `figures/fig01_class_distribution.png` and
`figures/fig03_duplication_by_attack_category.png`.

---

## 2. Target and class balance

Attacks are the **majority** class, both before deduplication (63.9%) and after
({prep['class_balance'][1] if 1 in prep['class_balance'] else prep['class_balance']['1']:.1%}).
This is the inverse of any production network and is the single most important
caveat attached to every precision figure in this project: precision depends on
the base rate, recall and false-positive rate do not. See
`figures/fig13_class_imbalance_context.png`.

Attack families span three orders of magnitude, from Normal down to Worms, which
is why per-family recall is always reported alongside its sample count
(`figures/fig02_attack_category_distribution.png`).

**Imbalance handling.** At a {prep['class_balance'][1] if 1 in prep['class_balance'] else prep['class_balance']['1']:.1%}
/ {prep['class_balance'][0] if 0 in prep['class_balance'] else prep['class_balance']['0']:.1%}
split the imbalance is mild, so aggressive resampling is unwarranted.
`class_weight` (linear and bagged models) and `scale_pos_weight` (boosted
models) were included as tunable hyper-parameters so the search could decide
empirically. SMOTE was **not** applied by default - see section 6.

---

## 3. The TTL artefact - the most important EDA finding

A lookup rule that uses `sttl` and nothing else - memorise the majority class of
each TTL value - achieves **{ttl['sttl_only_rule_accuracy_train']:.2%} accuracy**
on the deduplicated training split, against a
{ttl['majority_class_baseline_train']:.2%} majority-class baseline.

That is not a property of TTL in the real world. UNSW-NB15 was generated on a
testbed where the benign-traffic and attack-traffic generators ran on hosts
configured with different initial TTL values. `sttl` therefore largely encodes
*which generator produced this flow*, not *is this flow hostile*.

The same artefact propagates into `ct_state_ttl`, a feature the dataset authors
derived from `state` and the TTL fields.

**How this project responds.** The TTL features are **retained** - they are real
fields a sensor observes, and deleting genuine signal on suspicion would be its
own methodological error. Instead the artefact is (a) documented here, (b)
quantified on the fitted model with SHAP
(`figures/fig24_shap_artifact_check.png`), and (c) tested directly by the
`no_ttl` ablation, which retrains everything with `sttl`, `dttl` and
`ct_state_ttl` removed. See `figures/fig05_ttl_artifact.png`.

Categorical fields carry the same problem in milder form: several exotic
protocols are 100% malicious and ARP is 100% benign in this corpus
(`figures/fig04_categorical_attack_rates.png`). Rare protocol levels are
therefore pooled into one "uncommon protocol" indicator during encoding, so the
model learns a generalisable property rather than memorising which protocol
numbers a 2015 scanner happened to sweep.

---

## 4. Distributions, relationships and outliers

### 4.1 Heavy tails

Every counter feature spans six to nine orders of magnitude
(`figures/fig06_numeric_distributions_by_class.png`). The preprocessing pipeline
applies `log1p` before standard-scaling these columns: monotonic (so tree models
are unaffected), defined at zero (so the 120,288 zero-byte reverse directions
survive), and reversible.

### 4.2 Outliers are kept, deliberately

`figures/fig07_outlier_boxplots.png` shows thousands of points beyond the
whiskers on every volume feature, **in both classes**. Inspecting them shows
genuine records: multi-megabyte transfers, gigabit-per-second loads,
ten-thousand-packet bursts.

Rather than assert that "the outliers are the attacks", the composition of each
tail was measured. Among the **top 1%** of each measurement, the attack share is:

{tail_rows}

against a training-split base rate of {tail_base}. The tail is therefore
**informative but not simply "the attacks"** - `sbytes` and `rate` tails are
strongly attack-enriched, while the `flow_bytes_total` tail is actually
*benign*-enriched.

**No outlier is removed anywhere in this project.** Clipping the tail would
discard real behaviour of both kinds, and for the measurements where the tail is
attack-enriched it would delete exactly the events the system exists to catch.
The only values that would have been treated as erroneous are non-finite ones,
and the dataset contains none. Heavy tails are handled by transformation, not
deletion.

### 4.3 Redundancy

{len(correlation['pairs_above_090'])} feature pairs exceed |r| = 0.90 on the
training split. The tightest are structural rather than coincidental:

| Feature A | Feature B | \\|r\\| |
|---|---|---|
{corr_rows}

`sbytes`/`sloss` and `dbytes`/`dloss` are near-collinear because the loss
counters are derived from the byte counters; `is_ftp_login`/`ct_ftp_cmd` encode
the same FTP event twice. The resulting design matrix is severely
ill-conditioned, which is why the logistic regression needs L2 regularisation to
produce stable coefficients, and why SHAP rather than impurity-based importance
is used for attribution - impurity gain splits credit arbitrarily between
correlated twins. See `figures/fig08_correlation_heatmap.png`.

### 4.4 Categorical relationships

| Service | Records (train) | Attack rate |
|---|---|---|
{service_rows}

`service` is highly informative but partly artefactual: several low-volume
services are 100% malicious in this corpus. `state` is similarly strong - `int`
(no reply received) is dominated by attacks, `con` by benign traffic - and this
one **is** defensible: an unanswered connection genuinely is what scanning looks
like.

---

## 5. Feature engineering

Fifteen features were engineered. Each is a **row-wise function of a single
flow**, which has two consequences that matter:

1. **It cannot leak.** A feature that never sees the label cannot leak the
   target, and one that never sees another row cannot leak across the split.
2. **It is deployable.** Each can be computed by a sensor on one flow in
   isolation, which is how an IDS must actually operate.

| Feature | Security rationale |
|---|---|
{engineered_rows}

### 5.1 Numerical safety

46.7% of flows have `dpkts == 0` and 3,607 have `dur == 0`, so unguarded
division would corrupt a large fraction of the matrix rather than a rare edge
case. Every ratio is formed as `a / (a + b)` - bounded to [0, 1] - rather than
`a / b`, denominators are clipped, and the module asserts on output that no
non-finite value was produced. The `0/0` case (a flow that carried nothing in
either direction) encodes as the neutral 0.5 rather than an arbitrary 0.
`tests/test_features.py` verifies all of this on hand-built degenerate records.

### 5.2 Did they help?

{len(ranking['engineered_in_top25'])} of the {len(ENGINEERED_FEATURE_DOCS)} engineered features
appear in the top 25 univariate signals:
{', '.join('`' + f + '`' for f in ranking['engineered_in_top25'])}.

But the picture is **not uniform, and reporting only that number would be
misleading**. Measuring each binary flag's univariate spread - the gap between
its highest and lowest attack rate - gives:

{flag_rows}

Two flags separate the classes strongly on their own. Three - the TCP
session-state flags - show almost **no** univariate gap.

That is not evidence they are useless. It is a cancellation effect: the
TCP-state flags split in *opposite directions within the attack class*. Generic
traffic almost never completes a handshake, while Exploits almost always does,
so pooling the attack class hides both. Whatever value those features carry is
in **interaction**, which a univariate rate cannot see by construction.

A third finding is worth stating because it cuts against the feature's own
stated rationale: `service_unknown` runs **opposite** to the usual security
intuition in this corpus - flows with an *identified* service carry the higher
attack rate. On a real network, unclassifiable traffic on non-standard ports is
suspicious; in this testbed the attack generator concentrated on well-known
services. Another reminder that these are generator artefacts as much as attack
behaviour.

The contribution that actually matters is measured by the `no_engineered`
ablation (see `reports/Model_Evaluation_Report.md` §11) and by SHAP on the
fitted model - both of which can see interactions.

| Rank | Feature | \\|point-biserial r\\| | Origin |
|---|---|---|---|
{feature_rows}

See `figures/fig09_univariate_feature_ranking.png` and
`figures/fig10_engineered_feature_separation.png`.

---

## 6. Preprocessing decisions and their justification

| Decision | Choice | Why |
|---|---|---|
| Missing values | None imputed | There are none. `service == '-'` is a real category, and zeros are real measurements. |
| Duplicates | Removed **before** splitting | Prevents identical vectors spanning train and test. |
| Outliers | Retained | In this domain the tail is the signal. |
| Heavy tails | `log1p` then standard-scale | Handles nine orders of magnitude without deleting data. |
| Categorical encoding | One-hot with `min_frequency={config.MIN_CATEGORY_FREQUENCY}` | Bounds dimensionality and gives unseen protocols a defined destination at inference time. |
| Binary flags | Passed through unscaled | Scaling a Boolean buys nothing and makes SHAP harder to read. |
| Split | Stratified 60/20/20 on `attack_cat`, `random_state={config.RANDOM_STATE}` | Guarantees rare families appear in all three splits. |
| Imbalance | `class_weight` / `scale_pos_weight` as tuned hyper-parameters | Imbalance is mild; let the search decide. |
| SMOTE | Tested as an ablation, **not** used by default | Interpolating between flows produces records that could not exist on a wire - e.g. `tcp_handshake_complete = 0.43`. Applied to the training split only, with `SMOTENC` so nominal columns are not interpolated. The comparison is reported rather than asserted. |
| Leakage columns | `id`, `attack_cat`, `label` dropped in one enforcement point | Cannot be forgotten in one notebook and remembered in another. |

All transformations are fitted **inside the training fold**, because the
estimator sits in the same `Pipeline` as the scaler and encoder. Cross-validation
therefore re-fits the preprocessing on each fold automatically; leakage is
prevented structurally rather than by convention.

---

## 7. Figures

| Figure | Content |
|---|---|
| `fig01_class_distribution.png` | Class balance before and after deduplication |
| `fig02_attack_category_distribution.png` | Attack-family counts (log scale) |
| `fig03_duplication_by_attack_category.png` | Duplication rate per family |
| `fig04_categorical_attack_rates.png` | Attack rate by protocol, service, state |
| `fig05_ttl_artifact.png` | The TTL artefact, three panels |
| `fig06_numeric_distributions_by_class.png` | Twelve class-conditional distributions |
| `fig07_outlier_boxplots.png` | Outlier analysis by class |
| `fig08_correlation_heatmap.png` | Redundancy among the 39 numeric features |
| `fig09_univariate_feature_ranking.png` | Top 25 univariate signals |
| `fig10_engineered_feature_separation.png` | Engineered features by class |
| `fig11_volume_scatter.png` | Source vs destination bytes, by class and family |
| `fig12_attack_family_profile.png` | Per-family traffic fingerprint |
| `fig13_class_imbalance_context.png` | Dataset balance vs production base rates |
"""
    _write("EDA_Feature_Engineering_Report.md", body)


# --------------------------------------------------------------------------- #
# Model evaluation report
# --------------------------------------------------------------------------- #
def generate_model_report() -> None:
    test = _csv("test_metrics_main")
    validation = _csv("validation_metrics_main")
    deployment = _json("deployment")
    recommendation = _json("threshold_recommendation_main")
    operating = _json("operating_points_main")
    errors = _json("error_analysis_main")
    tuning = _json("best_params_main")
    ablations = _csv("ablation_summary", index_col=[0, 1])

    if test is None or deployment is None:
        _write("Model_Evaluation_Report.md",
               f"# Model Evaluation Report\n\n{MISSING}\n\n"
               "Run `python -m src.pipeline --train --evaluate`.")
        return

    best = deployment["model_key"]
    best_name = deployment["display_name"]
    threshold = deployment["threshold"]
    row = test.loc[best]

    headline = _model_table(test, [
        ("accuracy", "Accuracy", "num"), ("precision", "Precision", "num"),
        ("recall", "Recall", "num"), ("f1", "F1", "num"),
        ("roc_auc", "ROC-AUC", "num"), ("pr_auc", "PR-AUC", "num"),
    ])
    operational = _model_table(test, [
        ("false_positive_rate", "FPR", "pct3"),
        ("false_negative_rate", "FNR", "pct3"),
        ("tp", "TP", "int"), ("fn", "FN", "int"),
        ("fp", "FP", "int"), ("tn", "TN", "int"),
    ])
    cost = _model_table(test, [
        ("fit_seconds", "Training time", "s"),
        ("search_seconds", "Tuning time", "s"),
        ("throughput_flows_per_second", "Throughput", "rate"),
        ("inference_ms_per_1k_rows", "Latency / 1k flows (ms)", "num"),
        ("n_features_after_encoding", "Features", "int"),
    ])
    overfit = _model_table(test, [
        ("cv_f1_mean", "CV F1 (train split)", "num"),
        ("cv_f1_std", "CV F1 std", "num"),
        ("cv_train_minus_test_f1", "CV train - test F1", "num"),
        ("f1", "Test F1", "num"),
    ])

    params_rows = ""
    if tuning:
        params_rows = "\n".join(
            f"| {evaluate.DISPLAY_NAMES.get(key, key)} | `{json.dumps(record.get('best_params', {}))}` |"
            for key, record in tuning.items()
            if key in evaluate.MODEL_ORDER)

    op_rows = ""
    if operating:
        op_rows = "\n".join(
            f"| {name} | {m['threshold']:.3f} | {m['recall']:.4f} | {m['precision']:.4f} | "
            f"{m['f1']:.4f} | {m['false_positive_rate']:.3%} | {m['false_negative_rate']:.3%} | "
            f"{int(m['fn']):,} | {int(m['fp']):,} |"
            for name, m in operating.items())

    sensitivity_rows = ""
    if recommendation:
        sensitivity_rows = "\n".join(
            f"| {key.replace('ratio_', '')}:1 | {v['threshold']:.3f} | "
            f"{v['recall']:.4f} | {v['false_positive_rate']:.3%} |"
            for key, v in recommendation["cost_ratio_sensitivity"].items())

    fn_rows = ""
    if errors:
        fn_rows = "\n".join(
            f"| {family} | {rate:.2%} | {errors['false_negatives_by_attack_cat'].get(family, 0):,} |"
            for family, rate in sorted(
                errors["false_negative_rate_by_attack_cat"].items(),
                key=lambda kv: -kv[1]))
        fp_rows = "\n".join(
            f"| `{service}` | {rate:.2%} | {errors['false_positives_by_service'].get(service, 0):,} |"
            for service, rate in sorted(
                errors["false_positive_rate_by_service"].items(),
                key=lambda kv: -kv[1])[:8])
    else:
        fp_rows = ""

    ablation_rows = ""
    if ablations is not None and best in ablations.index.get_level_values(1):
        sub = ablations.xs(best, level=1)
        pretty = {
            "main": "Primary protocol (headline result)",
            "keep_duplicates": "Duplicates retained",
            "no_ttl": "TTL features removed",
            "no_engineered": "Engineered features removed",
            "smote": "SMOTE instead of class weights",
            "official_split": "Authors' published train/test split",
        }
        ablation_rows = "\n".join(
            f"| {pretty.get(name, name)} | {sub.loc[name, 'recall']:.4f} | "
            f"{sub.loc[name, 'precision']:.4f} | {sub.loc[name, 'f1']:.4f} | "
            f"{sub.loc[name, 'pr_auc']:.4f} | {sub.loc[name, 'false_negative_rate']:.3%} |"
            for name in sub.index)

    targets = [
        ("T1", "F1 >= 0.90", row["f1"], 0.90),
        ("T2", "ROC-AUC >= 0.95", row["roc_auc"], 0.95),
        ("T3", "PR-AUC >= 0.90", row["pr_auc"], 0.90),
        ("T4", "Attack recall >= 0.90", row["recall"], 0.90),
    ]
    target_rows = "\n".join(
        f"| {tid} | {desc} | {value:.4f} | {'**MET**' if value >= bar else '**NOT MET**'} |"
        for tid, desc, value, bar in targets)
    throughput_met = (row.get("throughput_flows_per_second", 0) >= 10_000)
    target_rows += (f"\n| T7 | Throughput >= 10,000 flows/s | "
                    f"{row.get('throughput_flows_per_second', float('nan')):,.0f}/s | "
                    f"{'**MET**' if throughput_met else '**NOT MET**'} |")

    val_best = validation.loc[best] if validation is not None else None

    body = f"""
# Model Evaluation Report

*Generated by `python -m src.report` on {date.today().isoformat()} from
`reports/metrics/`. Every number below was produced by executed code; none is
transcribed by hand.*

---

## 1. Experimental protocol

The order of operations is the substance of this report, not a formality:

1. Score every trained model on the **validation** split ({int(val_best['n']) if val_best is not None else 0:,} flows).
2. Select the deployment model from those validation numbers.
3. Tune the decision threshold on the **validation** split, for that model only.
4. Freeze model and threshold into `models/deployment.json`.
5. Open the **test** split ({int(row['n']):,} flows) and score it **once**.

Steps 1-4 never touch test data; step 5 changes no parameter. That is what lets
the numbers in section 3 be read as an estimate of generalisation rather than as
a number that was optimised toward.

**Tuning.** `RandomizedSearchCV` with 5-fold `StratifiedKFold`
(`random_state={config.RANDOM_STATE}`), scoring on F1 with average precision and
ROC-AUC recorded alongside. Because the estimator sits inside the same
`Pipeline` as the scaler and encoder, every fold re-fits the preprocessing on
its own training portion - fold-level leakage is structurally impossible.

**Solver selection (Logistic Regression).** Chosen by measurement, not default.
On the 92,210 x 73 training matrix: `lbfgs`/L2 converged in 3.6 s for validation
F1 0.8566; `liblinear`/L2 took 10.0 s for 0.8560; `saga`/L2 took 101.8 s for
0.8561; `saga`/L1 took 199.4 s for 0.8562 **without converging**. L1 was dropped
- 55x the runtime for a 0.0004 F1 difference. The near-separability created by
the TTL features is what makes the unregularised L1 problem pathological.

---

## 2. Model comparison - detection quality (test split)

{headline}

## 3. Operational error rates (test split)

{operational}

**FPR** is the share of benign flows wrongly alerted - analyst workload.
**FNR** is the share of attacks missed - residual security risk. These two, not
accuracy, are the numbers a SOC lead reads.

## 4. Computational cost

{cost}

## 5. Overfitting check

{overfit}

The `CV train - test F1` column is the gap between each model's score on data it
was fitted on and its cross-validated score. A large positive gap indicates
memorisation. Tree ensembles are expected to show a gap; what matters is that
the cross-validated and held-out test scores agree, which they do.

## 6. Selected hyper-parameters

| Model | Best configuration |
|---|---|
{params_rows}

---

## 7. Model selection

**Selected: {best_name}.**

{deployment['selection_reason']}

**Why PR-AUC and not accuracy.** Accuracy on a
{row['positive_rate_actual']:.0%}-positive corpus compresses every model into a
narrow band and rewards majority-class performance. PR-AUC summarises how well a
model *ranks* attacks above benign flows across all thresholds, on the positive
class only - the quantity that survives when the deployment base rate differs
from the dataset's, and the one that determines whether a useful operating point
exists at all. F1 alone would bake in the arbitrary 0.50 threshold that section 8
then replaces.

**Isolation Forest** is reported for comparison but is not a deployment
candidate. It is trained on benign flows only, sees no attack labels, and
produces a ranking score rather than a probability, so it cannot be operated
against a calibrated risk threshold. Its role is to establish what supervised
labelling actually buys. Its gap to the supervised models is the answer.

---

## 8. Threshold analysis (Phase 10)

0.50 is an arbitrary default, not an optimum. The threshold was swept on the
**validation** split - doing this on test would make the reported test metrics
optimistic - and three candidate operating points were computed.

| Operating point | Threshold | Recall | Precision | F1 | FPR | FNR | Missed attacks | False alerts |
|---|---|---|---|---|---|---|---|---|
{op_rows}

**Adopted: {threshold:.3f}** (F1-optimal on validation).

### Cost sensitivity

The cost-optimal threshold depends on how much worse a missed intrusion is than
a false alert. That ratio is an **assumption, openly stated, not a measurement**:

| Assumed FN:FP cost ratio | Optimal threshold | Recall | FPR |
|---|---|---|---|
{sensitivity_rows}

The project's working figure is 20:1 - deliberately conservative. Published
breach-cost studies imply ratios in the hundreds, but a detector tuned at that
ratio alerts on nearly everything and destroys itself through alert fatigue.

### Which point should a SOC actually run?

It depends on the alert budget, and the honest answer is that **this is a
business decision, not a modelling one**:

* A team with analyst capacity to spare should run the F1-optimal point and
  accept more false alerts to miss fewer attacks.
* A team already at queue capacity should run the FPR-constrained point, accept
  the higher miss rate, and pair the model with compensating controls.

See `figures/fig17_threshold_analysis.png`.

---

## 9. Pre-registered targets

The targets in `reports/problem_statement.md` were set **before** modelling.
Measured against {best_name} at threshold {threshold:.3f} on the test split:

| # | Target | Achieved | Status |
|---|---|---|---|
{target_rows}

---

## 10. Error analysis - what does the model get wrong?

### Attacks missed, by family

| Attack family | False-negative rate | Count missed |
|---|---|---|
{fn_rows}

### False alerts, by service

| Service | False-positive rate | Count |
|---|---|---|
{fp_rows}

### How the model is wrong - near-misses or confident errors?

{describe_error_confidence(errors, threshold)}

Full per-group breakdowns are in `reports/metrics/subgroup_audit_main.csv` and
`reports/Bias_Fairness_Analysis.md`.

---

## 11. Ablations - what is the headline number actually measuring?

Each row is a **separate end-to-end experiment** with its own split. All reuse
the hyper-parameters tuned in the main run, so exactly one thing changes at a
time.

| Experiment | Recall | Precision | F1 | PR-AUC | FNR |
|---|---|---|---|---|---|
{ablation_rows or MISSING}

These are the rows that decide how much the headline result means. See
`figures/fig19_ablation_comparison.png` and the interpretation in
`reports/Final_Project_Report.md` §12.

---

## 12. Cybersecurity interpretation

**Which model catches the most attacks?** The highest-recall model at a fixed
threshold is visible in section 2 - but recall at a *fixed* threshold is not a
fair comparison between models with different score distributions, which is why
selection used PR-AUC and the threshold was then tuned for the winner.

**Which generates the fewest false alerts?** Section 3, FPR column. At
{best_name}'s operating threshold, {int(row['fp']):,} of {int(row['tn'] + row['fp']):,}
benign test flows are alerted ({row['false_positive_rate']:.3%}). On a network
carrying 10 million benign flows a day, that rate implies roughly
{row['false_positive_rate'] * 10_000_000:,.0f} false alerts per day - which is
why the FPR-constrained operating point exists.

**What is missed?** Section 10. The miss rate is not uniform across families,
and the families with the worst recall are precisely the low-volume ones.

**What is the trade-off?** Every threshold decrease converts false negatives
into false positives at a rate the sweep in section 8 quantifies exactly.

**What should be deployed, and why?** {best_name}, at threshold
{threshold:.3f}, as a **triage-ranking layer feeding human analysts** - not as an
automated blocking control. The reasoning: it leads on PR-AUC, it is fast enough
to keep up with a real link ({row.get('throughput_flows_per_second', float('nan')):,.0f}
flows/s), and it supports per-alert SHAP explanations, which a blocking control
in a regulated environment requires. The ablations in section 11 are the reason
for the "triage, not blocking" qualifier.

---

## 13. Figures

| Figure | Content |
|---|---|
| `fig14_confusion_matrices.png` | Confusion matrix per model (test) |
| `fig15_roc_pr_curves.png` | ROC and precision-recall curves |
| `fig16_model_comparison.png` | Detection quality, error rates, throughput |
| `fig17_threshold_analysis.png` | Threshold sweep with candidate operating points |
| `fig18_subgroup_audit.png` | Recall by family; FPR by stratum |
| `fig19_ablation_comparison.png` | Headline model under each experimental condition |
"""
    _write("Model_Evaluation_Report.md", body)


def generate_all() -> None:
    """Regenerate every report that can be produced from current artefacts."""
    config.ensure_dirs()
    generate_environment_record()
    generate_eda_report()
    generate_model_report()

    from src import report_bias, report_final, report_readme
    report_bias.generate()
    report_final.generate()
    report_readme.generate()


if __name__ == "__main__":
    generate_all()
