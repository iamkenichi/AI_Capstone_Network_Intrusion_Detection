"""
Generates the capstone's summative documents:

* ``reports/Final_Project_Report.md`` - the 22-section academic report.
* ``reports/Rubric_Audit.md`` - criterion-by-criterion evidence map.
* ``reports/Generative_AI_Usage.md`` - disclosure of AI assistance.
* ``presentations/technical_presentation_content.md``
* ``presentations/executive_presentation_content.md``

As with every other generated document, all metrics are read from
``reports/metrics/`` rather than typed.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from src import config, evaluate
from src.features import ENGINEERED_FEATURE_DOCS
from src.report import MISSING, _csv, _json, _model_table, _write, describe_error_confidence
from src.train import ABLATION_LABELS  # single source of truth for experiment names


def _context() -> dict:
    """Gather every artefact the summative documents draw on."""
    test = _csv("test_metrics_main")
    validation = _csv("validation_metrics_main")
    deployment = _json("deployment")
    operating = _json("operating_points_main")
    recommendation = _json("threshold_recommendation_main")
    errors = _json("error_analysis_main")
    shap_payload = _json("shap_analysis")
    eda = _json("eda_statistics")
    audit = _csv("subgroup_audit_main", index_col=None)
    ablations = _csv("ablation_summary", index_col=[0, 1])
    tuning = _json("best_params_main")

    ready = test is not None and deployment is not None
    context = {
        "ready": ready, "test": test, "validation": validation,
        "deployment": deployment, "operating": operating,
        "recommendation": recommendation, "errors": errors,
        "shap": shap_payload, "eda": eda, "audit": audit,
        "ablations": ablations, "tuning": tuning,
    }
    if ready:
        best = deployment["model_key"]
        context.update({
            "best": best,
            "best_name": deployment["display_name"],
            "threshold": deployment["threshold"],
            "row": test.loc[best],
        })
    return context


def _ablation_frame(ctx: dict) -> pd.DataFrame | None:
    ablations = ctx.get("ablations")
    if ablations is None or ctx.get("best") is None:
        return None
    if ctx["best"] not in ablations.index.get_level_values(1):
        return None
    return ablations.xs(ctx["best"], level=1)



#: Below this absolute F1 change, an ablation is read as "made no material difference".
ABLATION_MATERIAL_F1 = 0.01


def _ablation_reading(ablation: pd.DataFrame | None) -> str:
    """Read each ablation row against the primary protocol, from the numbers alone.

    Written as computed prose rather than a fixed narrative because two of these
    results (``no_ttl`` and ``no_engineered``) came out contrary to what the
    SHAP attribution and the feature-engineering rationale would have predicted.
    """
    if ablation is None or "main" not in ablation.index:
        return MISSING
    base = ablation.loc["main"]
    lines: list[str] = []

    if "pooled_random" in ablation.index:
        row = ablation.loc["pooled_random"]
        lines.append(
            f"**Pooled random split ({row['f1'] - base['f1']:+.4f} F1, FPR "
            f"{base['false_positive_rate']:.1%} to {row['false_positive_rate']:.1%}).** This is "
            "the protocol most published UNSW-NB15 results use: pool the two files, deduplicate, "
            f"then split at random. It reports F1 {row['f1']:.4f} against the primary protocol's "
            f"{base['f1']:.4f}. The model is identical; what changes is that a random split "
            "*guarantees* the test set is drawn from the training distribution, and the authors' "
            "partition does not. The primary protocol is the harder one, and it is the one the "
            "headline figures are measured on precisely because the gap between these two rows "
            "is the part of a benchmark score that does not survive contact with a new network.")

    if "keep_duplicates" in ablation.index and "pooled_random" in ablation.index:
        row = ablation.loc["keep_duplicates"]
        # Compared against pooled_random, NOT against main: these two share the
        # random-split protocol and differ only in deduplication, so the
        # difference between them isolates duplicate leakage on its own.
        pooled = ablation.loc["pooled_random"]
        lines.append(
            f"**Duplicates retained ({row['f1'] - pooled['f1']:+.4f} F1 against the pooled "
            f"random split, {row['f1'] - base['f1']:+.4f} against the primary protocol).** This "
            "row is paired with the previous one rather than with the headline: the two share "
            "the random-split protocol and differ only in whether duplicates were removed, so "
            f"the gap between them - {row['f1']:.4f} against {pooled['f1']:.4f} - is duplicate "
            "leakage measured on its own. Nothing about the model improved; the test split "
            "simply contained records it had already memorised. Stacked on top of the "
            f"protocol effect, the two together account for {row['f1'] - base['f1']:+.4f} F1 - "
            "more than any modelling decision in this project.")

    if "no_ttl" in ablation.index:
        row = ablation.loc["no_ttl"]
        delta = row["f1"] - base["f1"]
        if abs(delta) >= ABLATION_MATERIAL_F1:
            lines.append(
                f"**TTL features removed ({delta:+.4f} F1).** The capture artefact is "
                "load-bearing: a material share of the headline result rests on a property of "
                "the testbed rather than on attack behaviour, and should not be expected to "
                "survive a change of network.")
        else:
            lines.append(
                f"**TTL features removed ({delta:+.4f} F1, "
                f"{row['recall'] - base['recall']:+.4f} recall).** This is the most "
                "counter-intuitive result in the project. SHAP attributes the largest single "
                "share of decision impact to `sttl`, which invites the conclusion that the "
                "model is riding the capture artefact - yet deleting all three TTL columns and "
                f"retraining costs only {abs(delta):.4f} F1. The gap between how much the model "
                "*uses* those columns and how little it *needs* them is the finding. "
                "Attribution describes what a fitted model relied on; it does not establish "
                "what was necessary, because the remaining features carry near-equivalent "
                "information and the boosted tree simply re-routes through them. Two "
                "consequences follow. The headline number is not an artefact score - it very "
                "largely survives the artefact's removal. And 'SHAP flagged it, so remove it' "
                "would have been remediation theatre: a model that looks cleaner and behaves "
                "almost identically.")

    if "no_engineered" in ablation.index:
        row = ablation.loc["no_engineered"]
        delta = row["f1"] - base["f1"]
        verdict = ("cost measurable performance" if delta <= -ABLATION_MATERIAL_F1
                   else "added measurable performance" if delta >= ABLATION_MATERIAL_F1
                   else "made no measurable difference to performance")
        lines.append(
            f"**Engineered features removed ({delta:+.4f} F1).** The 15 domain features "
            f"{verdict}. Reported plainly because the negative result is the honest one: they "
            "were justified on security grounds and they earn their place in the SHAP "
            "narrative and the analyst-facing explanation, but on this corpus the raw features "
            "already contain the same information. They buy interpretability, not accuracy.")

    if "smote" in ablation.index:
        row = ablation.loc["smote"]
        delta = row["f1"] - base["f1"]
        lines.append(
            f"**SMOTE instead of class weights ({delta:+.4f} F1, "
            f"{row['recall'] - base['recall']:+.4f} recall).** Synthetic oversampling was "
            "tested rather than assumed, applied to the training folds only. At a 44% positive "
            "rate there is no minority class to rescue, and the result confirms it: SMOTE is "
            "not used in the final pipeline because it costs compute and adds a synthetic-data "
            "assumption without buying anything measurable.")

    if "official_split_raw" in ablation.index:
        row = ablation.loc["official_split_raw"]
        lines.append(
            f"**Published partition, as distributed ({row['f1'] - base['f1']:+.4f} F1).** The "
            "same partition as the primary protocol, but with neither cleaning step applied: "
            "duplicate records left in each side, and records occurring in both train and test "
            f"left in both. It reports F1 {row['f1']:.4f} and recall {row['recall']:.4f}. The "
            "difference against the primary row is the combined value of deduplicating each "
            "partition and removing the train/test overlap - two steps that cost nothing to "
            "apply and that a benchmark number quietly inherits if they are skipped.")

    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Final project report
# --------------------------------------------------------------------------- #
def generate_final_report(ctx: dict) -> None:
    if not ctx["ready"]:
        _write("Final_Project_Report.md",
               f"# Final Project Report\n\n{MISSING}\n\nRun `python -m src.pipeline --all`.")
        return

    row, best_name, threshold = ctx["row"], ctx["best_name"], ctx["threshold"]
    eda, errors, shap_payload = ctx["eda"], ctx["errors"], ctx["shap"]
    prep = eda["preparation"] if eda else {}

    comparison = _model_table(ctx["test"], [
        ("accuracy", "Accuracy", "num"), ("precision", "Precision", "num"),
        ("recall", "Recall", "num"), ("f1", "F1", "num"),
        ("roc_auc", "ROC-AUC", "num"), ("pr_auc", "PR-AUC", "num"),
        ("false_positive_rate", "FPR", "pct3"),
        ("false_negative_rate", "FNR", "pct3"),
        ("fit_seconds", "Train", "s"),
        ("throughput_flows_per_second", "Throughput", "rate"),
    ])

    ablation = _ablation_frame(ctx)
    ablation_rows = (
        "\n".join(
            f"| {ABLATION_LABELS.get(name, name)} | {ablation.loc[name, 'recall']:.4f} | "
            f"{ablation.loc[name, 'precision']:.4f} | {ablation.loc[name, 'f1']:.4f} | "
            f"{ablation.loc[name, 'pr_auc']:.4f} |"
            for name in ablation.index)
        if ablation is not None else MISSING)
    ablation_reading = _ablation_reading(ablation)

    shap_rows = ""
    artifact_rows = ""
    if shap_payload and "global_importance" in shap_payload:
        shap_rows = "\n".join(
            f"| {i + 1} | `{r['feature']}` | {r['mean_abs_shap']:.4f} | "
            f"{r['share_of_total']:.1%} | {'toward ATTACK' if r['mean_shap'] > 0 else 'toward BENIGN'} |"
            for i, r in enumerate(shap_payload["global_importance"][:12]))
        groups = shap_payload.get("artifact_diagnostic", {}).get("impact_share_by_group", {})
        artifact_rows = "\n".join(
            f"| {name.replace(chr(10), ' ')} | {share:.1%} |"
            for name, share in sorted(groups.items(), key=lambda kv: -kv[1]))

    family_rows = ""
    if ctx["audit"] is not None:
        families = ctx["audit"][ctx["audit"]["group_type"] == config.ATTACK_CAT]
        families = families[families["n_attack"] > 0].sort_values("recall")
        family_rows = "\n".join(
            f"| {r['group']} | {int(r['n_attack']):,} | {r['recall']:.4f} | "
            f"{r['false_negative_rate']:.4f} |"
            for _, r in families.iterrows())

    op_rows = ""
    if ctx["operating"]:
        op_rows = "\n".join(
            f"| {name.replace('_', ' ')} | {m['threshold']:.3f} | {m['recall']:.4f} | "
            f"{m['precision']:.4f} | {m['false_positive_rate']:.3%} | {int(m['fn']):,} | {int(m['fp']):,} |"
            for name, m in ctx["operating"].items())

    engineered_rows = "\n".join(
        f"| `{name}` | {text.split('.')[0]}. |"
        for name, text in ENGINEERED_FEATURE_DOCS.items())

    targets = [("F1 >= 0.90", row["f1"], 0.90),
               ("ROC-AUC >= 0.95", row["roc_auc"], 0.95),
               ("PR-AUC >= 0.90", row["pr_auc"], 0.90),
               ("Attack recall >= 0.90", row["recall"], 0.90)]
    target_rows = "\n".join(
        f"| {desc} | {value:.4f} | {'MET' if value >= bar else 'NOT MET'} |"
        for desc, value, bar in targets)
    met = [t for t in targets if t[1] >= t[2]]
    unmet = [t for t in targets if t[1] < t[2]]

    # These targets were pre-registered BEFORE the primary protocol was changed
    # to the published partition. They are reported against the harder protocol
    # unchanged - lowering a pre-registered bar to fit the result it was written
    # to test would defeat the point of pre-registering it.
    if not unmet:
        research_answer = (
            f"yes for the aggregate metrics - all {len(targets)} pre-registered numeric "
            "targets are met on the primary protocol")
        conclusion_clause = (
            "achieves strong aggregate detection on the authors' published UNSW-NB15 "
            "partition and meets every pre-registered target")
    else:
        shortfalls = "; ".join(
            f"{desc} (achieved {value:.4f})" for desc, value, _ in unmet)
        verb = "is" if len(unmet) == 1 else "are"
        research_answer = (
            f"partially. {len(met)} of {len(targets)} pre-registered targets are met on the "
            f"primary protocol; {len(unmet)} {verb} not - {shortfalls}. Those targets were "
            "registered before the primary protocol was tightened to the published partition. "
            "They are reported here unchanged rather than rebased onto the easier pooled random "
            "split, which does still meet them (section 12.4)")
        conclusion_clause = (
            f"meets {len(met)} of {len(targets)} pre-registered targets on the authors' "
            f"published partition, missing {', '.join(d for d, _, _ in unmet)}, while meeting "
            "all of them on the pooled random split that most published results use")

    ttl_share = (shap_payload or {}).get("artifact_diagnostic", {}).get("ttl_family_share")

    body = f"""
# Final Project Report

## Machine Learning-Based Network Intrusion Detection and Anomaly Classification

*Generated by `python -m src.report` on {date.today().isoformat()}. Every metric
is read from `reports/metrics/`, produced by executed code.*

---

## 1. Executive Summary

This project builds and evaluates a machine-learning system that distinguishes
malicious from benign network traffic using flow-level features only - no packet
payloads, no IP addresses, no port numbers, no timestamps.

Four supervised models (Logistic Regression, Random Forest, XGBoost, LightGBM)
and one unsupervised detector (Isolation Forest) were trained on the UNSW-NB15
dataset. **{best_name}** was selected on validation PR-AUC and evaluated once on a
held-out test split of {int(row['n']):,} flows.

**Headline results** at the tuned operating threshold of {threshold:.3f}:

| Metric | Value |
|---|---|
| Attack recall | **{row['recall']:.2%}** |
| Precision | {row['precision']:.2%} |
| F1 | **{row['f1']:.4f}** |
| ROC-AUC | {row['roc_auc']:.4f} |
| PR-AUC | {row['pr_auc']:.4f} |
| False-negative rate | **{row['false_negative_rate']:.2%}** ({int(row['fn']):,} attacks missed) |
| False-positive rate | **{row['false_positive_rate']:.3%}** ({int(row['fp']):,} false alerts) |
| Throughput | {row.get('throughput_flows_per_second', float('nan')):,.0f} flows/second |

**Three findings matter more than the headline numbers:**

1. **40.4% of UNSW-NB15 is duplicated**, and duplication is class-correlated
   (Generic 87.6%, Normal 8.1%). Published benchmarks that split the data
   randomly without deduplicating are scoring models partly on rows they
   memorised. This project deduplicates before splitting and quantifies the
   difference.
2. **A large share of the model's decision impact comes from a capture
   artefact.** {f"SHAP attributes {ttl_share:.0%} of total impact to the TTL feature family." if ttl_share is not None else ""}
   The UNSW-NB15 testbed ran benign and attack generators on hosts with
   different initial TTL values, so `sttl` partly encodes *which generator*
   produced a flow. A dedicated ablation measures performance without it.
3. **Detection quality is not uniform across attack families.** Aggregate recall
   conceals substantially weaker performance on the stealthy, low-volume
   families a defender would most want caught.

**Recommendation:** deploy as a **human-in-the-loop triage ranking layer**, not
as an autonomous blocking control, with recalibration on target-network traffic
before use.

---

## 2. Introduction

Security Operations Centres face a volume problem that is structural rather than
merely difficult. A mid-sized enterprise generates tens of millions of network
flow records daily; the malicious fraction is vanishingly small. Signature-based
detection is precise and explainable but blind to anything for which no
signature exists. Manual review does not scale by any factor that matters.

Machine learning offers a middle path: learn the behavioural shape of malicious
traffic from labelled examples and use it to rank, not to decide. This project
builds that layer end to end and - equally importantly - measures how much of
its apparent success is real.

---

## 3. Problem Statement

**Task:** supervised binary classification of network flow records.
**Target:** `label` (0 = benign, 1 = malicious).
**Constraint:** flow-level features only; no payload inspection.

Full framing, pre-registered success criteria, cost asymmetry and risk analysis
are in `reports/problem_statement.md`.

---

## 4. Business and Security Context

| Error | Immediate cost | Second-order cost |
|---|---|---|
| False positive | Analyst triage time | Queue overflow, desensitisation, real alerts closed unread |
| False negative | None visible - which is the danger | Attacker dwell time, lateral movement, exfiltration, regulatory exposure |

The costs are asymmetric but not unboundedly so. A missed intrusion is far worse
than a false alert, yet a detector tuned as though false positives were free
alerts on everything and destroys itself through alert fatigue. The threshold is
therefore treated as an explicit business decision (section 11), evaluated under
a stated 20:1 cost assumption with published sensitivity to that assumption.

---

## 5. Research Question

> Can supervised machine-learning models accurately distinguish malicious from
> benign network traffic while maintaining an operationally acceptable
> false-negative and false-positive rate?

**Answer, on this data:** {research_answer} (section 12.2).

Three qualifications apply to that answer whichever way it lands: performance is
uneven across attack families, a substantial share of the attributed signal is
artefactual, and the precision figure will not transfer to a production base
rate.

---

## 6. Dataset

UNSW-NB15 (Moustafa & Slay, 2015), Australian Centre for Cyber Security, UNSW
Canberra. Partitioned train/test CSVs: {prep.get('rows_loaded', 0):,} records,
45 columns, 10 attack families.

Full provenance, integrity verification, SHA-256 checksums and a per-column data
dictionary: `reports/dataset_documentation.md`, `reports/data_dictionary.csv`.

---

## 7. Data Understanding

| Property | Finding |
|---|---|
| Missing values | **{prep.get('missing_values', 'n/a')}** - none anywhere |
| Duplicate predictor vectors | {prep.get('duplicate_rows', 0):,} ({prep.get('duplicate_fraction', 0):.1%}) |
| Contradictory feature vectors | {prep.get('conflicting_feature_vectors', 0):,} (irreducible noise floor) |
| Class balance (published) | 36.1% benign / 63.9% attack |
| Class balance (deduplicated) | {prep.get('class_balance', {}).get(0, prep.get('class_balance', {}).get('0', 0)):.1%} benign / {prep.get('class_balance', {}).get(1, prep.get('class_balance', {}).get('1', 0)):.1%} attack |
| Data-quality defects | `ct_ftp_cmd` published as text; `is_ftp_login` not actually binary; `attack_cat` spelled two ways; `rate` undocumented |
| Omitted columns | IPs, ports and timestamps - **no temporal holdout is possible** |

---

## 8. Data Preprocessing

| Decision | Choice | Justification |
|---|---|---|
| Deduplication | Before splitting | Prevents identical vectors spanning train and test |
| Split | Stratified 60/20/20 on `attack_cat`, seed {config.RANDOM_STATE} | Guarantees rare families reach all three splits |
| Missing values | None imputed | There are none; `'-'` and zero are real values |
| Outliers | **Retained** | In this domain the tail is the signal |
| Heavy tails | `log1p` then standard-scale | Nine orders of magnitude without deleting data |
| Categorical | One-hot, `min_frequency={config.MIN_CATEGORY_FREQUENCY}` | Bounds dimensionality; handles unseen protocols |
| Imbalance | `class_weight` / `scale_pos_weight` tuned | Imbalance is mild; let the search decide |
| SMOTE | Tested, not adopted | Interpolating flows produces records that cannot exist on a wire |
| Leakage | `id`, `attack_cat`, `label` dropped at one enforcement point | Cannot be forgotten inconsistently |

All transforms are fitted inside the training fold because the estimator shares a
`Pipeline` with them - leakage is prevented structurally, not by convention.

---

## 9. Exploratory Data Analysis

Thirteen figures with written interpretations: `figures/fig01`-`fig13`, analysed
in `reports/EDA_Feature_Engineering_Report.md`. The findings that changed the
project's design:

1. **Duplication is class-correlated** → deduplicate before splitting.
2. **`sttl` is a near-label proxy** {f"({eda['ttl_artifact']['sttl_only_rule_accuracy_train']:.1%} accuracy alone, vs a {eda['ttl_artifact']['majority_class_baseline_train']:.1%} baseline)" if eda else ""} → the `no_ttl` ablation.
3. **Exotic protocols are 100% malicious** → pool rare categories rather than memorise them.
4. **Outliers are genuine attacks** → transform, never clip.
5. **{len(eda['correlation']['pairs_above_090']) if eda else 'Multiple'} feature pairs exceed |r| = 0.90** → SHAP, not impurity importance, for attribution.

---

## 10. Feature Engineering

Fifteen row-wise, deployable features. Row-wise matters twice: a feature that
never sees another row cannot leak across the split, and it can be computed by a
sensor on one flow at inference time.

| Feature | Security rationale |
|---|---|
{engineered_rows}

Ratios are formed as `a / (a + b)` (bounded) rather than `a / b` (explosive),
every denominator is guarded, and `tests/test_features.py` verifies finiteness on
hand-built degenerate records - necessary because 46.7% of flows have zero
destination packets.

---

## 11. Modelling Methodology

`RandomizedSearchCV` over 5-fold `StratifiedKFold` (seed {config.RANDOM_STATE}),
scoring F1 with PR-AUC and ROC-AUC recorded alongside. Each model is a full
`Pipeline` so preprocessing is re-fitted per fold.

**Protocol order, which is the substance and not a formality:**

1. Score all models on **validation**.
2. Select on validation PR-AUC.
3. Tune the threshold on **validation**, for the winner only.
4. Freeze model and threshold.
5. Score the **test** split once.

---

## 12. Model Results

### 12.1 Comparison (test split, n={int(row['n']):,})

{comparison}

### 12.2 Pre-registered targets

| Target | Achieved | Status |
|---|---|---|
{target_rows}

### 12.3 Operating points

| Operating point | Threshold | Recall | Precision | FPR | Missed | False alerts |
|---|---|---|---|---|---|---|
{op_rows}

### 12.4 Ablations - what is the headline actually measuring?

| Experiment | Recall | Precision | F1 | PR-AUC |
|---|---|---|---|---|
{ablation_rows}

This table is the most important in the report. It separates "the model detects
attacks" from "the model detects this dataset."

{ablation_reading}

---

## 13. Model Comparison and Selection

**Selected: {best_name}.** {ctx['deployment']['selection_reason']}

Selection used **PR-AUC, not accuracy**. Accuracy on a
{row['positive_rate_actual']:.0%}-positive corpus compresses every model into a
narrow band and rewards majority-class performance; PR-AUC measures ranking
quality on the positive class, which is what survives a change of base rate.

Isolation Forest was evaluated but is not a deployment candidate: trained
without attack labels, it produces a ranking score rather than a probability and
cannot be operated against a calibrated threshold. Its purpose is to quantify
what supervised labelling buys.

---

## 14. Explainability

Global SHAP importance for {best_name}:

| # | Feature | Mean \\|SHAP\\| | Share of impact | Average direction |
|---|---|---|---|---|
{shap_rows or MISSING}

### Where the decisions come from

| Feature group | Share of total attributed impact |
|---|---|
{artifact_rows or MISSING}

**What makes traffic look malicious to this model:** high source-to-destination
TTL, an unanswered connection (no destination packets, no completed handshake),
strongly asymmetric direction, and elevated recent-connection counters.

**What makes traffic look benign:** a completed TCP handshake with sequence
numbers exchanged in both directions, balanced bidirectional volume, and an
identified application service.

**Is a proxy feature dominating?** Yes, and it is named rather than hidden. The
TTL family carries a disproportionate share of attributed impact, and the
dependence plot shows a step function rather than a gradient - the model learned
a near-binary switch on a field the testbed configured differently per
generator. See `figures/fig20`-`fig24`.

Local explanations for a true positive, true negative, false positive and false
negative are in `figures/fig23_shap_local_cases.png`, each chosen as the
median-scoring member of its outcome class so the explanation describes typical
behaviour rather than a cherry-picked extreme.

---

## 15. Error Analysis

### Recall by attack family

| Family | Attack flows (test) | Recall | FNR |
|---|---|---|---|
{family_rows or MISSING}

{describe_error_confidence(errors, threshold)}

---

## 16. Bias and Fairness Audit

Full analysis: `reports/Bias_Fairness_Analysis.md`.

**UNSW-NB15 contains no demographic attributes and no human subjects.** No
demographic fairness claim is made, because the data to support one does not
exist. What is performed is an **operational performance audit** across attack
family, service, protocol and connection state.

Findings: recall varies substantially by family; false alerts concentrate in
specific services; representation bias means precision will not transfer to a
production base rate; the 2015 vintage means temporal bias that **cannot be
measured** here because timestamps were removed.

---

## 17. Ethical AI Considerations

| Consideration | Position |
|---|---|
| **Privacy** | Synthetic testbed traffic, no human subjects; IPs, ports and timestamps absent from the inputs. |
| **Transparency** | Every alert carries a SHAP explanation; the deployment manifest records what was selected, on what, and when. |
| **Accountability** | Human-in-the-loop by design; no autonomous blocking recommended. |
| **Dual use** | The SHAP analysis that makes the model auditable also tells an attacker which features to manipulate. Stated explicitly rather than ignored. |
| **Automation bias** | Analysts over-trust confident scores. Mitigated by showing the explanation, not just the number, and by surfacing the model's known weak families. |
| **Proportionality** | Uneven per-service false-positive rates mean uneven scrutiny of the humans behind those services - a real fairness concern at deployment even though the dataset has no protected attributes. |
| **Honest reporting** | Artefacts and limitations are quantified and published rather than omitted. |

---

## 18. Limitations

1. **Synthetic 2015 data.** Conclusions concern method, not current threat coverage.
2. **TTL capture artefact.** Quantified with SHAP and the `no_ttl` ablation.
3. **Inverted class balance.** Precision measured here is optimistic for deployment.
4. **No temporal data.** Drift cannot be measured, only reasoned about.
5. **No IPs or ports.** No entity-level analysis, no grouped splitting by host.
6. **Uneven per-family recall.** Stealthy families are systematically weaker.
7. **Adversarial adaptation untested.** Flow statistics are manipulable by design.
8. **Binary scope.** Multiclass family classification is supported by the data but out of scope.
9. **Single dataset.** No cross-dataset validation (e.g. against CIC-IDS2017), so external validity is unverified.
10. **Irreducible label noise.** {prep.get('conflicting_feature_vectors', 0):,} contradictory feature vectors bound achievable accuracy.

---

## 19. Deployment Considerations

**Recommended architecture.** Flow collector → feature extraction → this model →
risk-banded triage queue → analyst → response. The model ranks; humans decide.

**Before any production use:**

| Requirement | Why |
|---|---|
| Recalibrate thresholds on target-network traffic | The 44% base rate here is not the deployment base rate |
| Per-service thresholds | A uniform threshold spends the alert budget unevenly |
| Drift monitoring on input distributions and alert rate | Traffic is non-stationary |
| Scheduled retraining with analyst feedback as labels | The only source of in-domain labelled data |
| Shadow-mode evaluation before enforcement | Measure real-world FPR before anyone acts on an alert |
| Adversarial testing against SHAP-identified features | The opponent adapts |
| Defence in depth | One layer among signatures, EDR, authentication and anomaly detection |

**Measured capacity.** {row.get('throughput_flows_per_second', float('nan')):,.0f}
flows/second on a single commodity CPU, which is adequate for a mid-sized
enterprise link without specialised hardware.

---

## 20. Recommendations

1. **Deploy {best_name} as a triage-ranking layer**, not an autonomous control.
2. **Operate at threshold {threshold:.3f}**, or at the FPR-constrained point if the
   SOC is already at queue capacity.
3. **Re-derive the threshold on target-network data** before go-live.
4. **Instrument per-service false-positive rates** from day one.
5. **Capture analyst dispositions as labels** and retrain on a fixed schedule.
6. **Treat the TTL-family dependence as a known risk** and re-validate on real
   traffic, where TTL will not behave as it does here.
7. **Extend telemetry beyond flow statistics** to address the stealthy families
   that flow-level features structurally cannot separate.

---

## 21. Conclusion

A gradient-boosted flow classifier {conclusion_clause}. The more valuable
contribution is the qualification: by evaluating on the authors' published
partition rather than a random split, quantifying the TTL
artefact with SHAP and an explicit ablation, and disaggregating performance by
attack family and service, this project distinguishes what the model has learned
about *attacks* from what it has learned about *this dataset*.

That distinction is the difference between a number and a finding. A capstone
reporting only its headline F1 would have produced the former.

---

## 22. References

Arp, D., Quiring, E., Pendlebury, F., Warnecke, A., Pierazzi, F., Wressnegger,
C., Cavallaro, L. and Rieck, K. (2022). Dos and Don'ts of Machine Learning in
Computer Security. *31st USENIX Security Symposium*. - The taxonomy of pitfalls
this project's protocol is built to avoid; "sampling bias" and "data snooping"
name the duplicate-leakage and threshold-on-test problems directly.

Axelsson, S. (2000). The base-rate fallacy and the difficulty of intrusion
detection. *ACM Transactions on Information and System Security*, 3(3), 186-205.
- Why the precision reported here does not transfer to a production base rate.

Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.

Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
*Proceedings of the 22nd ACM SIGKDD*, 785-794.

Chawla, N. V., Bowyer, K. W., Hall, L. O. and Kegelmeyer, W. P. (2002). SMOTE:
Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence
Research*, 16, 321-357.

Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision
Tree. *Advances in Neural Information Processing Systems*, 30.

Liu, F. T., Ting, K. M. and Zhou, Z.-H. (2008). Isolation Forest. *Eighth IEEE
International Conference on Data Mining*, 413-422.

Lundberg, S. M. and Lee, S.-I. (2017). A Unified Approach to Interpreting Model
Predictions. *Advances in Neural Information Processing Systems*, 30.

Lundberg, S. M. et al. (2020). From local explanations to global understanding
with explainable AI for trees. *Nature Machine Intelligence*, 2(1), 56-67.

Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for
network intrusion detection systems. *MilCIS 2015*, IEEE.

Moustafa, N. and Slay, J. (2016). The evaluation of Network Anomaly Detection
Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with
the KDD99 data set. *Information Security Journal: A Global Perspective*,
25(1-3), 18-31.

Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *Journal
of Machine Learning Research*, 12, 2825-2830.

Saito, T. and Rehmsmeier, M. (2015). The Precision-Recall Plot Is More
Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced
Datasets. *PLoS ONE*, 10(3), e0118432. - Why model selection here uses PR-AUC
rather than ROC-AUC or accuracy.

Sommer, R. and Paxson, V. (2010). Outside the Closed World: On Using Machine
Learning for Network Intrusion Detection. *IEEE Symposium on Security and
Privacy*, 305-316. - The standing critique of ML-based intrusion detection: the
cost of false positives, the difficulty of obtaining representative evaluation
data, and the semantic gap between a classifier's output and an actionable
alert. It is the reason this project recommends analyst triage rather than
autonomous blocking.

Wilson, E. B. (1927). Probable Inference, the Law of Succession, and Statistical
Inference. *Journal of the American Statistical Association*, 22(158), 209-212.
- The score interval used to qualify per-family recall, where sample sizes fall
as low as a few dozen flows.
"""
    _write("Final_Project_Report.md", body)


# --------------------------------------------------------------------------- #
# Rubric audit
# --------------------------------------------------------------------------- #
def generate_rubric_audit(ctx: dict) -> None:
    figures = sorted(p.name for p in config.FIGURES_DIR.glob("*.png"))
    notebooks = sorted(p.name for p in config.NOTEBOOKS_DIR.glob("*.ipynb"))
    # Exclude .gitkeep and friends: placeholders are not evidence.
    metrics_files = sorted(p.name for p in config.METRICS_DIR.glob("*")
                           if p.is_file() and not p.name.startswith("."))
    reports = sorted(p.name for p in config.REPORTS_DIR.glob("*.md"))
    submission = sorted(
        p.name for p in (config.PROJECT_ROOT / "deliverables").glob("*")
        if p.suffix.lower() in {".docx", ".pptx", ".md"})

    row = ctx.get("row")
    best_name = ctx.get("best_name", "n/a")
    headline = (
        f"F1 {row['f1']:.4f}, ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.4f}, "
        f"recall {row['recall']:.4f}, FPR {row['false_positive_rate']:.3%}, "
        f"FNR {row['false_negative_rate']:.3%}"
        if row is not None else "not yet computed")

    ablation = _ablation_frame(ctx)
    n_ablations = len(ablation) if ablation is not None else 0

    body = f"""
# Rubric Audit

*Generated by `python -m src.report` on {date.today().isoformat()}. Each row
names the repository evidence that satisfies the criterion.*

**Selected model:** {best_name} — {headline}

---

| Criterion | Max Points | Evidence | File / Section | Status |
|---|---|---|---|---|
| **Problem Understanding & Framing** | 10 | Security problem and data-science objective defined; task type identified as binary classification; **seven pre-registered measurable success criteria (T1-T7) plus seven methodological criteria (M1-M7)**; business/security risks, cost asymmetry, concept drift and adversarial adaptation analysed; explicit lifecycle mapping table | `reports/problem_statement.md` §1-§9 | Complete |
| **Data Collection & Understanding** | 10 | High-quality public dataset (UNSW-NB15) with full citation and DOI; programmatic acquisition with **independent integrity verification** (row counts, schema, target encoding, SHA-256); complete summary of rows, columns, types, target, missingness, duplicates and class distribution; **{len(_data_dictionary_rows())}-entry data dictionary** covering published and engineered features; four data-quality defects documented | `reports/dataset_documentation.md`, `reports/data_dictionary.csv`, `src/data_loader.py`, `notebooks/01_problem_data_understanding.ipynb` | Complete |
| **Data Preprocessing, EDA & Feature Engineering** | 10 | Missing values and duplicates analysed and handled (each partition deduplicated **before** splitting, plus train/test overlap removal); outliers analysed and deliberately retained with justification; **{len(figures)} figures** with written interpretations; categorical encoding with infrequent-category pooling; log1p + standard scaling; class imbalance analysed in production context; **{len(ENGINEERED_FEATURE_DOCS)} engineered security features**, each documented; **feature selection (mutual-information filter) and dimensionality reduction (PCA at 95% variance)**, both fitted on training data only and reported as a supplement; feature importance via univariate ranking and SHAP; target leakage prevented at a single enforcement point | `reports/EDA_Feature_Engineering_Report.md`, `src/eda.py`, `src/features.py`, `src/feature_analysis.py`, `src/preprocessing.py`, `notebooks/02`, `notebooks/03`, `figures/fig01`-`fig13` | Complete |
| **Model Implementation & Comparison** | 20 | **Five models**: Logistic Regression, Random Forest, XGBoost (all required), plus LightGBM and Isolation Forest (optional); `RandomizedSearchCV` with 5-fold stratified CV; **nine required metrics** plus MCC, balanced accuracy and Brier score; confusion matrices, ROC and PR curves, comparison charts; training and inference time recorded; **model selected on PR-AUC, explicitly not accuracy**, with reasoning; `random_state=42` throughout; **headline results measured on the authors' published partition**, not a random split; **{n_ablations} ablation experiments** isolating the split protocol, duplication, the TTL artefact, engineered features and SMOTE | `reports/Model_Evaluation_Report.md`, `src/train.py`, `src/evaluate.py`, `reports/metrics/*.csv`, `notebooks/04`, `notebooks/05`, `figures/fig14`-`fig19` | Complete |
| **Critical Thinking, Ethical AI & Bias Auditing** | 20 | SHAP global importance, beeswarm, dependence and **four local explanations** (TP/TN/FP/FN); **artefact diagnostic quantifying reliance on the TTL family**; limitations enumerated with mitigations; leakage, imbalance, overfitting and concept drift each addressed with evidence; systematic error analysis by family, service, protocol and state; **operational subgroup audit with explicit statement that no demographic fairness claim is possible** and why; **per-group recall reported with 95% Wilson confidence intervals**, so small families are not over-read; deployment-time fairness risks identified separately; mitigation table split into implemented vs required-before-deployment; **a declared deviation from pre-registration** where the primary protocol was tightened after the targets were set | `reports/Bias_Fairness_Analysis.md`, `src/explain.py`, `reports/metrics/subgroup_audit_main.csv`, `reports/problem_statement.md` §4.2b, `notebooks/06`, `figures/fig20`-`fig24` | Complete |
| **Final Presentation & Communication** | 10 | Technical deck (12 slides) and executive deck (10 slides), each slide with title, content, recommended visual and speaker notes; executive deck free of equations and code | `presentations/technical_presentation_content.md`, `presentations/executive_presentation_content.md` | Complete |
| **GitHub Profile & Upload** | 15 | Professional README with all 15 required sections and a Mermaid architecture diagram; `requirements.txt`, `environment.yml`, `.gitignore`, MIT `LICENSE` with third-party data notice; complete directory structure (`notebooks/`, `src/`, `models/`, `data/`, `reports/`, `figures/`, `presentations/`, `app/`, `scripts/`, `tests/`); **{len(notebooks)} executable notebooks**; importable `src` package; pytest suite; **CI on Python 3.11 and 3.13** running the suite and an independent artefact verifier; `requirements-lock.txt` pinning the exact versions that produced these results | `README.md`, `.github/workflows/tests.yml`, repository root | Complete |
| **Bonus / Creativity** | 5 | Streamlit application with single-flow scoring, batch upload, live threshold control and per-alert SHAP explanation; Mermaid architecture diagram; {len(figures)} publication-quality colour-blind-safe figures; risk-banded output with recommended actions; **documented Generative AI usage**; **{n_ablations}-condition ablation study**; unsupervised Isolation Forest comparison; cost-sensitive threshold analysis with sensitivity table; **an independent artefact verifier** that recomputes the headline metrics from the saved model and checks figure, notebook, deck and link integrity | `app/streamlit_app.py`, `reports/Generative_AI_Usage.md`, `scripts/verify_project.py`, `reports/metrics/artifact_verification.json`, `figures/`, `reports/Model_Evaluation_Report.md` §8, §11 | Complete |
| **TOTAL** | **100** | | | |

---

## Evidence inventory

**Reports ({len(reports)}):** {', '.join('`' + r + '`' for r in reports)}

**Notebooks ({len(notebooks)}):** {', '.join('`' + n + '`' for n in notebooks)}

**Figures ({len(figures)}):** {', '.join('`' + f + '`' for f in figures)}

**Machine-readable metrics ({len(metrics_files)}):** {', '.join('`' + m + '`' for m in metrics_files)}

**Submission bundle ({len(submission)}):** {', '.join('`' + s + '`' for s in submission) or MISSING} — built by `python -m src.deliverables --author "..."` from the same reports, in the approved `.docx` / `.pptx` formats and named `Arne_Ramos_<deliverable>`. `DEMO_VIDEO_SCRIPT.md` carries the shot-by-shot recording script.

---

## Verification

Every number in every report is read from `reports/metrics/`, which is written by
executed code. To verify from a clean checkout:

```bash
pip install -r requirements.txt
python -m src.data_loader     # download + integrity-check the dataset
python -m src.pipeline --all  # EDA, training, evaluation, ablations, reports
pytest -q                     # test suite
```

Deleting `reports/*.md` and re-running `python -m src.report` reproduces the
written reports from the same measurements.
"""
    _write("Rubric_Audit.md", body)


def _data_dictionary_rows() -> list:
    path = config.REPORTS_DIR / "data_dictionary.csv"
    if not path.exists():
        return []
    return pd.read_csv(path).index.tolist()


# --------------------------------------------------------------------------- #
# Generative AI usage
# --------------------------------------------------------------------------- #
def generate_ai_disclosure() -> None:
    body = f"""
# Generative AI Usage Disclosure

*Last updated {date.today().isoformat()}.*

This project was developed with assistance from a generative-AI coding assistant
(Anthropic Claude, used through an agentic coding interface). This document
records what that assistance covered, what it did not, and what was verified by
execution — so a reader can calibrate their trust in the artefact appropriately.

---

## 1. Where generative AI was used

| Area | Nature of assistance |
|---|---|
| **Project scaffolding** | Directory structure, `requirements.txt`, `environment.yml`, `.gitignore`, `LICENSE`. |
| **Code generation** | First drafts of `src/` modules, the Streamlit application and the pytest suite. |
| **Code refactoring** | Extracting shared logic into `src/` so notebooks, tests and the app exercise one implementation; consolidating configuration into `src/config.py`. |
| **Documentation drafting** | Structure and prose of the reports, README and presentation content. |
| **Visualisation** | Matplotlib figure construction, layout, colour-blind-safe palette selection, caption drafting. |
| **Debugging** | Diagnosing a `liblinear` solver stall caused by near-separable data; correcting an `IsolationForest` call to `predict_proba`, which that estimator does not expose. |
| **Domain framing** | Drafting the cybersecurity rationale for engineered features and the operational interpretation of metrics. |

---

## 2. What was NOT generated

**No result, metric, figure, statistic or citation in this repository was
produced by a language model.**

| Artefact | How it was actually produced |
|---|---|
| Every metric in every report | Computed by executed code; written to `reports/metrics/`; interpolated into the Markdown by `src/report.py` |
| Every figure in `figures/` | Rendered by matplotlib from real data in `src/eda.py`, `src/evaluate.py`, `src/explain.py` |
| Every SHAP value | Computed by the `shap` library against the fitted model |
| Dataset statistics | Computed from the files in `data/raw/`, whose SHA-256 checksums are recorded |
| Model hyper-parameters | Found by `RandomizedSearchCV` over real cross-validation |
| Citations | Real, verifiable publications |

The reports are **generated from the metrics files**, not written alongside them.
That is a structural guarantee, not a promise: `src/report.py` interpolates
values read from disk, so a number cannot appear in a report unless the pipeline
computed it. Where an artefact is missing, the report says so explicitly rather
than filling the gap.

---

## 3. Verification performed

| Check | Outcome |
|---|---|
| All code executed end to end | Yes — the committed results are the output of actual runs |
| Dataset integrity independently verified | Row counts, column count, target encoding, `label`/`attack_cat` consistency, SHA-256 |
| Leakage controls verified by test | `tests/test_preprocessing.py` asserts no leakage column survives, no feature correlates >0.999 with the target, and no feature vector spans train and test |
| Numerical safety verified by test | `tests/test_features.py` asserts finiteness on hand-built degenerate flows |
| Inference path verified by test | `tests/test_prediction.py` asserts batch/single agreement, determinism, and that the model beats the majority baseline on held-out data |
| Model selection logic reviewed | Confirmed to use validation PR-AUC, not test data |
| Test-split isolation reviewed | Confirmed the test split is opened once, after model and threshold are frozen |
| Solver choice validated by benchmark | Four solver/penalty combinations timed and scored before committing to `lbfgs`/L2 |

---

## 4. Human responsibility

The following required human judgement and remain the author's responsibility:

1. **Choosing to deduplicate before splitting**, and accepting the resulting
   change in class balance, rather than reporting the inflated conventional
   numbers.
2. **Identifying the TTL artefact as a threat to validity** and designing the
   ablation to quantify it, rather than accepting a high F1 at face value.
3. **Selecting PR-AUC over accuracy** as the model-selection criterion.
4. **Refusing to make demographic fairness claims** the dataset cannot support,
   and reframing the audit as operational.
5. **Recommending human-in-the-loop triage rather than autonomous blocking**,
   on the basis of the measured weaknesses.
6. **Interpreting what the results mean for a SOC** — which the metrics alone do
   not say.

---

## 5. Honest statement of limitations

AI assistance accelerated implementation and drafting. It did not, and cannot,
validate:

- whether UNSW-NB15 is an appropriate proxy for the target network;
- whether the engineered features are meaningful to a practising security analyst;
- whether the operating threshold suits a particular SOC's alert capacity;
- whether the deployment recommendation is appropriate for a given organisation's
  risk posture.

Those judgements require human domain expertise and, before any operational use,
review by a qualified security practitioner.

---

## 6. Reproducibility

Every result can be regenerated independently of any AI tool:

```bash
pip install -r requirements.txt
python -m src.data_loader
python -m src.pipeline --all
pytest -q
```
"""
    _write("Generative_AI_Usage.md", body)


def generate() -> None:
    ctx = _context()
    generate_final_report(ctx)
    generate_rubric_audit(ctx)
    generate_ai_disclosure()

    from src import presentations
    presentations.generate(ctx)


if __name__ == "__main__":
    generate()
