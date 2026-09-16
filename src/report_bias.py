"""
Generates ``reports/Bias_Fairness_Analysis.md`` from the measured subgroup audit.

The central honesty constraint of this report: UNSW-NB15 contains **no
demographic attributes and no human subjects**. It is synthetic machine-generated
traffic with all IP addresses, ports and timestamps removed. Any claim about
demographic fairness made from it would be fabricated.

What is therefore produced is an **operational performance audit** - does the
detector work equally well across the strata that genuinely exist in the data
(protocol, service, connection state, attack family)? - plus an explicit
statement of what that audit can and cannot support, and a discussion of where
the real fairness risk lies in deployment.
"""

from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src import config, evaluate
from src.report import MISSING, _csv, _json, _write, describe_error_confidence


def generate() -> None:
    audit = _csv("subgroup_audit_main", index_col=None)
    errors = _json("error_analysis_main")
    deployment = _json("deployment")
    shap_payload = _json("shap_analysis")
    eda = _json("eda_statistics")
    ablations = _csv("ablation_summary", index_col=[0, 1])
    test = _csv("test_metrics_main")

    if audit is None or deployment is None:
        _write("Bias_Fairness_Analysis.md",
               f"# Bias, Fairness and Error Analysis\n\n{MISSING}\n\n"
               "Run `python -m src.pipeline --evaluate`.")
        return

    best = deployment["model_key"]
    best_name = deployment["display_name"]
    threshold = deployment["threshold"]
    overall = test.loc[best] if test is not None else None

    # ---------------- subgroup tables ----------------
    families = audit[audit["group_type"] == config.ATTACK_CAT].copy()
    families = families[families["n_attack"] > 0].sort_values("recall")
    family_rows = "\n".join(
        f"| {r['group']} | {int(r['n_attack']):,} | {r['recall']:.4f} | "
        f"{r['false_negative_rate']:.4f} | {int(round(r['false_negative_rate'] * r['n_attack'])):,} |"
        for _, r in families.iterrows())

    def stratum_table(group_type: str, limit: int = 12) -> str:
        sub = audit[(audit["group_type"] == group_type) & (audit["n_benign"] >= 30)]
        sub = sub.sort_values("n", ascending=False).head(limit)
        return "\n".join(
            f"| `{r['group']}` | {int(r['n']):,} | {int(r['n_attack']):,} | "
            f"{_fmt(r['recall'])} | {_fmt(r['precision'])} | "
            f"{_fmt(r['false_positive_rate'], pct=True)} | "
            f"{_fmt(r['false_negative_rate'], pct=True)} | {_fmt(r['f1'])} |"
            for _, r in sub.iterrows())

    worst_recall = families.iloc[0] if not families.empty else None
    best_recall = families.iloc[-1] if not families.empty else None
    recall_spread = (float(families["recall"].max() - families["recall"].min())
                     if not families.empty else float("nan"))

    services = audit[(audit["group_type"] == "service") & (audit["n_benign"] >= 30)]
    worst_fpr = (services.sort_values("false_positive_rate", ascending=False).iloc[0]
                 if not services.empty else None)

    ttl_share = None
    if shap_payload and "artifact_diagnostic" in shap_payload:
        ttl_share = shap_payload["artifact_diagnostic"].get("ttl_family_share")

    ablation_note = MISSING
    if ablations is not None and best in ablations.index.get_level_values(1):
        sub = ablations.xs(best, level=1)
        if "no_ttl" in sub.index and "main" in sub.index:
            # Signed as (ablated - main), i.e. the change caused by removing the features.
            delta_f1 = sub.loc["no_ttl", "f1"] - sub.loc["main", "f1"]
            delta_recall = sub.loc["no_ttl", "recall"] - sub.loc["main", "recall"]
            material = abs(delta_f1) >= 0.01
            ablation_note = (
                f"Removing `sttl`, `dttl` and `ct_state_ttl` entirely and retraining "
                f"moves F1 from {sub.loc['main', 'f1']:.4f} to {sub.loc['no_ttl', 'f1']:.4f} "
                f"({delta_f1:+.4f}) and recall from {sub.loc['main', 'recall']:.4f} to "
                f"{sub.loc['no_ttl', 'recall']:.4f} ({delta_recall:+.4f}).")
            if material:
                ablation_note += (
                    "\n\nThe artefact is therefore **load-bearing**: a material part of the "
                    "headline result depends on a property of the capture environment rather "
                    "than on attack behaviour, and should not be expected to transfer.")
            else:
                ablation_note += (
                    "\n\nThis is the most important negative result in the project, and it cuts "
                    "against the obvious reading of the SHAP chart. SHAP attributes the largest "
                    "share of attributed impact to the TTL family, but attribution measures what "
                    "the fitted model *used*, not what was *necessary*. Deleting those three "
                    "columns costs essentially nothing, which means the remaining features carry "
                    "near-equivalent information and the boosted tree simply re-routes through "
                    "them. The artefact is a **redundant shortcut, not a crutch**. Two "
                    "consequences follow. First, the headline number is not an artefact "
                    "score - it survives the artefact's removal. Second, a SHAP-driven "
                    "'remove the suspicious feature' remediation would have produced a model "
                    "that looks cleaner and behaves identically, which is a warning about "
                    "treating attribution as evidence of causation.")
        if "keep_duplicates" in sub.index and "main" in sub.index:
            ablation_note += (
                f"\n\nRetaining the duplicate records instead of removing them moves F1 from "
                f"{sub.loc['main', 'f1']:.4f} to {sub.loc['keep_duplicates', 'f1']:.4f} and "
                f"PR-AUC from {sub.loc['main', 'pr_auc']:.4f} to "
                f"{sub.loc['keep_duplicates', 'pr_auc']:.4f}. That difference is the size of the "
                "optimism a benchmark inherits by splitting UNSW-NB15 without deduplicating.")

    fn_profile = ""
    if errors and "median_profile_by_outcome" in errors:
        profile = pd.DataFrame(errors["median_profile_by_outcome"]).T
        keep = [c for c in ["dur", "sbytes", "dbytes", "spkts", "dpkts",
                            "sttl", "dttl", "rate"] if c in profile.columns]
        order = ["true_positive", "false_negative", "true_negative", "false_positive"]
        profile = profile.reindex([o for o in order if o in profile.index])[keep]
        fn_profile = ("| Outcome | " + " | ".join(keep) + " |\n"
                      "|---|" + "|".join(["---"] * len(keep)) + "|\n"
                      + "\n".join(
                          f"| {idx.replace('_', ' ')} | "
                          + " | ".join(f"{profile.loc[idx, c]:,.3g}" for c in keep) + " |"
                          for idx in profile.index))

    dup_by_family = ""
    if eda:
        dup_by_family = "\n".join(
            f"| {family} | {rate:.1%} |"
            for family, rate in sorted(
                eda["preparation"]["duplicate_rate_by_attack_cat"].items(),
                key=lambda kv: -kv[1])[:5])

    body = f"""
# Bias, Fairness and Error Analysis

*Generated by `python -m src.report` on {date.today().isoformat()} from
`reports/metrics/subgroup_audit_main.csv` and `reports/metrics/error_analysis_main.json`.
Model: **{best_name}** at threshold **{threshold:.3f}**, evaluated on the held-out
test split.*

---

## 1. What kind of audit this is - and what it is not

**UNSW-NB15 contains no demographic attributes and no human subjects.**

It is synthetic traffic generated on a closed 2015 testbed by the IXIA
PerfectStorm appliance. There are no people in it. The partitioned CSV files
used here additionally omit every IP address, port number and timestamp, so
there is not even an indirect identifier from which a protected characteristic
could be inferred.

**Therefore this report makes no demographic fairness claims, and none can be
made from this dataset.** There is no measurement of disparate impact by race,
gender, age, disability, nationality or socioeconomic status, because the data
required to compute one does not exist. Any such claim would be fabricated.

What is performed instead is an **operational performance audit**: does the
detector work equally well across the strata that genuinely exist in this data?

| Grouping | Why it is a legitimate audit axis |
|---|---|
| **Attack family** (`attack_cat`) | Uneven recall means some threat classes are systematically under-defended. |
| **Application service** (`service`) | Uneven false-positive rates mean some services - and the teams that use them - absorb a disproportionate share of investigation burden. |
| **Transport protocol** (`proto`) | Uneven performance means blind spots that an attacker can deliberately move into. |
| **Connection state** (`state`) | Tests whether detection depends on a flow completing, which determines behaviour against aborted or evasive sessions. |

This is a **model performance audit**, not a demographic fairness assessment.
Section 7 explains where genuine fairness risk does enter the picture at
deployment time - because it does, just not through the dataset.

---

## 2. Overall reference point

{f'''| Metric | Test-split value |
|---|---|
| Records | {int(overall['n']):,} |
| Attack base rate | {overall['positive_rate_actual']:.2%} |
| Recall | {overall['recall']:.4f} |
| Precision | {overall['precision']:.4f} |
| F1 | {overall['f1']:.4f} |
| False-positive rate | {overall['false_positive_rate']:.3%} |
| False-negative rate | {overall['false_negative_rate']:.3%} |
| Attacks missed | {int(overall['fn']):,} |
| False alerts | {int(overall['fp']):,} |''' if overall is not None else MISSING}

Aggregate numbers hide exactly the failures this report exists to find. The rest
of it disaggregates them.

---

## 3. Detection by attack family

| Attack family | Attack flows (test) | Recall | FNR | Attacks missed |
|---|---|---|---|---|
{family_rows}

Precision and FPR are not defined per attack family: a family row asks "of the
Exploits flows, how many did we catch?", which is a recall question. A false
positive is a *benign* flow, so it cannot belong to an attack family.

### Interpretation

{f'''Recall spans **{recall_spread:.3f}** across families - from
{worst_recall['recall']:.1%} on **{worst_recall['group']}**
({int(worst_recall['n_attack']):,} test flows) to {best_recall['recall']:.1%} on
**{best_recall['group']}** ({int(best_recall['n_attack']):,} test flows). The
detector is emphatically **not** uniformly effective.''' if worst_recall is not None else MISSING}

Three mechanisms drive the weak families, and they are distinguishable:

1. **Scarcity.** The tail families have few training examples, so the model has
   little evidence from which to learn their shape. Worms, Analysis, Backdoor
   and Shellcode are all in the low hundreds to low thousands after
   deduplication.
2. **Behavioural overlap with benign traffic.** Analysis and Backdoor traffic is
   designed to be unremarkable - a backdoor that looked like a port scan would
   be a bad backdoor. Low-and-slow attacks produce flow statistics close to
   ordinary sessions, and no amount of extra data fixes an overlap that is
   intrinsic to the threat.
3. **Duplication-driven distortion.** Families whose training data was heavily
   duplicated contributed far fewer *distinct* examples than their raw counts
   suggested:

{f'''| Attack family | Share of published records that were duplicates |
|---|---|
{dup_by_family}''' if dup_by_family else ''}

**These are not equivalent problems.** Scarcity is fixable with more data;
behavioural overlap is not, and is the more serious finding, because it bounds
what any flow-level detector can achieve against stealthy threats regardless of
model choice.

---

## 4. False alerts by operational stratum

### By application service

| Service | Flows | Attacks | Recall | Precision | FPR | FNR | F1 |
|---|---|---|---|---|---|---|---|
{stratum_table('service')}

### By transport protocol

| Protocol | Flows | Attacks | Recall | Precision | FPR | FNR | F1 |
|---|---|---|---|---|---|---|---|
{stratum_table('proto', 10)}

### By connection state

| State | Flows | Attacks | Recall | Precision | FPR | FNR | F1 |
|---|---|---|---|---|---|---|---|
{stratum_table('state', 8)}

### Interpretation

{f'''False alerts are **not** distributed evenly. `{worst_fpr['group']}` carries the
highest false-positive rate at {worst_fpr['false_positive_rate']:.2%} of its
{int(worst_fpr['n_benign']):,} benign test flows - against an overall rate of
{overall['false_positive_rate']:.3%}.''' if worst_fpr is not None and overall is not None else MISSING}

This concentration is **actionable, which is the point of measuring it**. A
uniform global threshold spends the alert budget unevenly; per-service
thresholds or suppression rules on the worst offenders would reclaim analyst
capacity without lowering recall elsewhere. It is also a warning: a stratum with
an elevated FPR is a stratum where analysts will learn to dismiss alerts, which
converts a precision problem into a recall problem through human behaviour.

---

## 5. Error profile - what do the mistakes look like?

{fn_profile or MISSING}

{describe_error_confidence(errors, threshold)}

---

## 6. Sources of bias in the data and the model

### 6.1 Collection-environment bias - the dominant problem

UNSW-NB15 is **synthetic**. Benign traffic was generated by an appliance, not by
people using a network. Real benign traffic is messier, more varied, and
contains far more unusual-but-innocent behaviour than any generator produces.

The concrete consequence is the **TTL artefact**: the benign and attack
generators ran on hosts with different initial TTL values, so `sttl` acts as a
near-label proxy. {f"A lookup rule on `sttl` alone reaches {eda['ttl_artifact']['sttl_only_rule_accuracy_train']:.1%} accuracy on the training split against a {eda['ttl_artifact']['majority_class_baseline_train']:.1%} majority baseline." if eda else ""}

{f"SHAP attributes **{ttl_share:.0%} of the model's total decision impact** to the TTL family, confirming that the fitted model does lean on it heavily (`figures/fig24_shap_artifact_check.png`)." if ttl_share is not None else ""}

{ablation_note}

**Why the features were kept anyway.** TTL is a real field that a real sensor
observes and that carries genuine information (hop count, OS fingerprint).
Deleting it because it is *too* predictive on this dataset would be a different
methodological error - discarding real signal on suspicion. The defensible
response is to keep it, measure its influence, and report the counterfactual,
which is what the `no_ttl` ablation does.

### 6.2 Representation bias

Attacks are the **majority class** here. In production they are a small fraction
of a percent. This inverts the usual imbalance problem and has a specific,
often-overlooked consequence: **precision does not transfer across base rates,
while recall and false-positive rate do.**

At a 44% base rate, a 1% FPR yields relatively few false alerts relative to true
ones. At a 0.1% base rate the same model with the same FPR is swamped: false
positives outnumber true positives by roughly an order of magnitude. **Every
precision figure in this project is optimistic for deployment**, and the
FPR-constrained operating point in the Model Evaluation Report exists precisely
because of this. See `figures/fig13_class_imbalance_context.png`.

### 6.3 Temporal bias and concept drift

The traffic is from 2015. The attack taxonomy predates modern ransomware,
living-off-the-land techniques, cloud-native attack patterns and supply-chain
compromise. Benign traffic predates the current application mix.

**This project cannot measure drift.** The partitioned CSVs omit `Stime` and
`Ltime`, so no temporal holdout is possible - a genuine limitation, stated
rather than worked around. Every claim here is about *method*, not about current
threat coverage.

### 6.4 Class-correlated duplication

Documented in section 3 and in the EDA report. Handled by deduplicating before
splitting; the cost is quantified by ablation rather than assumed.

### 6.5 Label bias

Labels come from the testbed's ground truth - what the generator was *told* to
emit. Any mislabelled or accidentally-benign attack traffic is baked in
permanently. The {f"{eda['preparation']['conflicting_feature_vectors']:,}" if eda else "several hundred"}
feature vectors carrying contradictory labels are direct evidence that the
labelling is not perfectly consistent.

### 6.6 Adversarial adaptation

Unlike most ML applications, this one has an opponent who is actively trying to
defeat it. An attacker who learns that the model keys on duration, packet rate
and directionality can pad packets, throttle rates and inject decoy reverse
traffic to move a flow across the boundary. The SHAP analysis that makes the
model explainable to defenders would also tell an attacker exactly which
features to manipulate.

### 6.7 Alert fatigue as a failure mode

A detector that produces more alerts than a team can process does not degrade
gracefully - it fails completely, because analysts stop reading the queue. The
uneven per-service FPR in section 4 makes this worse than the global average
suggests: a team whose service is over-flagged learns to dismiss *its own*
alerts first.

---

## 7. Where genuine fairness risk does enter

The dataset has no protected attributes, but a **deployed** system operates on a
real network used by real people, and that is where fairness re-enters:

1. **Uneven burden across teams.** If the model over-flags a particular service,
   the humans who use that service are investigated more often. Research groups,
   developers and network engineers whose legitimate work looks statistically
   unusual absorb disproportionate scrutiny. Section 4 shows this is already
   true across services in this model.
2. **Assistive and accessibility technologies.** Screen readers, alternative
   input devices and assistive network tooling can produce atypical traffic
   patterns. A detector trained on "typical" traffic may systematically flag
   users who rely on them.
3. **Feedback loops.** If flagged users are monitored more closely, more of
   their activity is labelled suspicious, and retraining amplifies the original
   skew.
4. **Automation bias.** An analyst shown a 0.94 score is measurably more likely
   to confirm it than to challenge it, which converts a model error into a human
   decision about a specific person.

These are **deployment-time risks**, not properties of UNSW-NB15. They are
listed because a bias audit that stops at the dataset would miss them entirely.

---

## 8. Mitigations

### Implemented in this project

| Mitigation | Where |
|---|---|
| Deduplication before splitting | `src/preprocessing.py`; `keep_duplicates` ablation quantifies the effect |
| Leakage columns excluded at one enforcement point | `src.preprocessing.split_xy` |
| Stratification on attack family so rare families reach every split | `src/preprocessing.py` |
| Artefact quantified rather than hidden | SHAP diagnostic + `no_ttl` ablation |
| Per-family and per-service disaggregated metrics | This report |
| Threshold tuned on validation, with a low-FPR alternative | `src/evaluate.py` |
| Per-alert SHAP explanation available at inference | `src.predict.explain_one`, Streamlit app |
| Rare categories pooled rather than memorised | `min_frequency` in the one-hot encoder |

### Required before any deployment

| Mitigation | Rationale |
|---|---|
| **Recalibrate on target-network data** | The 44% base rate here is not the deployment base rate; thresholds must be re-derived on real traffic. |
| **Per-service thresholds** | Section 4 shows a uniform threshold spends the alert budget unevenly. |
| **Human review; no automated blocking** | Confidently-wrong false negatives and demonstrable artefact dependence both argue against autonomous action. |
| **Scheduled retraining** | Non-stationary traffic; a 2015-trained model is a starting point, not an endpoint. |
| **Drift detection** | Monitor input feature distributions and alert-rate stability; alarm on divergence. |
| **Richer telemetry** | Flow statistics alone miss stealthy families. Host telemetry, DNS and authentication logs address exactly the overlap problem in section 3. |
| **Ensemble with signature and anomaly detection** | The Isolation Forest comparison exists to show the complementary role a novelty detector plays for traffic the classifier has never seen. |
| **Adversarial testing** | Red-team the model by perturbing the features SHAP identifies as influential. |
| **Audit the audit** | Re-run this subgroup analysis on production data, where the strata - and the humans behind them - are real. |
| **Analyst feedback capture** | Record dismissals as labels; they are the cheapest source of in-domain training data and the only way to close the loop. |

---

## 9. Honest summary

The model performs well on this dataset. Three findings qualify that:

1. **A substantial share of its decision impact comes from a capture artefact**
   ({f"{ttl_share:.0%} attributed to the TTL family" if ttl_share is not None else "quantified in the SHAP diagnostic"}),
   and the `no_ttl` ablation shows what performance looks like without it.
2. **Detection quality is not uniform.** Recall varies substantially across
   attack families, and the weakest families are the stealthy ones a defender
   would most want caught.
3. **Precision will not transfer.** It is measured at a 44% attack base rate and
   will fall sharply on a real network.

None of these invalidates the work. All three are reasons the recommendation is
**human-in-the-loop triage ranking**, not autonomous blocking - and they are
stated here rather than buried, because a capstone that reports only its
headline F1 has not actually evaluated anything.

---

## 10. Supporting artefacts

| Artefact | Content |
|---|---|
| `reports/metrics/subgroup_audit_main.csv` | Full per-group metrics, every stratum |
| `reports/metrics/error_analysis_main.json` | Error counts, rates and median profiles |
| `reports/metrics/shap_analysis.json` | Global importance and artefact diagnostic |
| `reports/metrics/ablation_summary.csv` | Every experimental condition |
| `figures/fig18_subgroup_audit.png` | Recall by family; FPR by stratum |
| `figures/fig24_shap_artifact_check.png` | Where the model's decisions come from |
| `figures/fig19_ablation_comparison.png` | Performance under each condition |
"""
    _write("Bias_Fairness_Analysis.md", body)


def _fmt(value, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "n/a"
    return f"{value:.2%}" if pct else f"{value:.4f}"


if __name__ == "__main__":
    generate()
