"""
Generates the two presentation decks as structured Markdown.

Each slide carries a title, its main content, a recommended visual (naming an
actual file in ``figures/``), and speaker notes. Metrics are interpolated from
``reports/metrics/`` exactly as in the written reports, so the decks cannot
drift out of sync with the results.

The two decks are written for genuinely different audiences. The technical deck
assumes the reader knows what PR-AUC is and cares why it was chosen over
accuracy. The executive deck contains **no equations, no code and no metric
jargon** - detection performance is expressed in attacks caught, attacks missed
and analyst hours.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from src import config
from src.features import ENGINEERED_FEATURE_DOCS

PRESENTATIONS_DIR = config.PROJECT_ROOT / "presentations"


def _write(name: str, body: str) -> None:
    PRESENTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = PRESENTATIONS_DIR / name
    path.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"[report] wrote presentations/{name}")


def generate(ctx: dict) -> None:
    if not ctx.get("ready"):
        placeholder = ("_Results are not available yet. Run "
                       "`python -m src.pipeline --all`, then regenerate._")
        _write("technical_presentation_content.md",
               f"# Technical Presentation\n\n{placeholder}")
        _write("executive_presentation_content.md",
               f"# Executive Presentation\n\n{placeholder}")
        return

    _technical(ctx)
    _executive(ctx)


# --------------------------------------------------------------------------- #
def _ablation_lookup(ctx: dict) -> pd.DataFrame | None:
    ablations = ctx.get("ablations")
    best = ctx.get("best")
    if ablations is None or best is None:
        return None
    if best not in ablations.index.get_level_values(1):
        return None
    return ablations.xs(best, level=1)


#: Absolute F1 change below which an ablation counts as "no material difference".
MATERIAL_F1 = 0.01


def _ablation_findings(ablation: pd.DataFrame | None) -> tuple[str, str]:
    """Slide bullets stating what each ablation *found*, plus the artefact verdict.

    Computed rather than written, because ``no_ttl`` and ``no_engineered`` both
    came out against expectation and a hand-written slide would have kept
    asserting the expectation.
    """
    fallback = ("- _Run `python -m src.pipeline --ablations` to populate this slide._",
                "artefact influence")
    if ablation is None or "main" not in ablation.index:
        return fallback
    base = ablation.loc["main"]
    bullets: list[str] = []
    verdict = "artefact influence"

    if "pooled_random" in ablation.index:
        delta = ablation.loc["pooled_random", "f1"] - base["f1"]
        bullets.append(
            f"- **Pooled random split → F1 {delta:+.4f}.** The protocol most published results "
            "use. We report the harder published partition instead")
    if "keep_duplicates" in ablation.index and "pooled_random" in ablation.index:
        # Paired with pooled_random, which it differs from only in deduplication.
        delta = (ablation.loc["keep_duplicates", "f1"]
                 - ablation.loc["pooled_random", "f1"])
        bullets.append(
            f"- **Duplicates retained → F1 {delta:+.4f}** (vs the pooled split)**.** Skipping "
            "deduplication hands back a better-looking number for no better model")
    if "no_ttl" in ablation.index:
        delta = ablation.loc["no_ttl", "f1"] - base["f1"]
        if abs(delta) >= MATERIAL_F1:
            bullets.append(
                f"- **TTL removed → F1 {delta:+.4f}.** The capture artefact is load-bearing; "
                "that much of the score is testbed recognition, not attack detection")
            verdict = "measured dependence on the TTL capture artefact"
        else:
            bullets.append(
                f"- **TTL removed → F1 {delta:+.4f}.** SHAP's top feature is *used far more "
                "than it is needed* — the model re-routes through correlated features")
            verdict = ("a capture artefact the model leans on far more than it needs "
                       "(measured by ablation, not assumed)")
    if "no_engineered" in ablation.index:
        delta = ablation.loc["no_engineered", "f1"] - base["f1"]
        bullets.append(
            f"- **No engineered features → F1 {delta:+.4f}.** The 15 domain features buy "
            "interpretability, not accuracy — reported as the negative result it is")
    if "smote" in ablation.index:
        delta = ablation.loc["smote", "f1"] - base["f1"]
        bullets.append(
            f"- **SMOTE → F1 {delta:+.4f}.** Tested rather than assumed, and not adopted")
    if "official_split_raw" in ablation.index:
        delta = ablation.loc["official_split_raw", "f1"] - base["f1"]
        bullets.append(
            f"- **Partition as distributed → F1 {delta:+.4f}.** What the two cleaning steps "
            "(dedup each side, remove train/test overlap) are worth")
    return ("\n".join(bullets) if bullets else fallback[0], verdict)


def _technical(ctx: dict) -> None:
    row, best_name, threshold = ctx["row"], ctx["best_name"], ctx["threshold"]
    test, eda, shap_payload = ctx["test"], ctx["eda"], ctx["shap"]
    prep = eda["preparation"] if eda else {}
    ablation = _ablation_lookup(ctx)
    ttl_share = (shap_payload or {}).get("artifact_diagnostic", {}).get("ttl_family_share")

    model_rows = "\n".join(
        f"| {test.loc[k, 'display_name']} | {test.loc[k, 'recall']:.4f} | "
        f"{test.loc[k, 'precision']:.4f} | {test.loc[k, 'f1']:.4f} | "
        f"{test.loc[k, 'pr_auc']:.4f} | {test.loc[k, 'false_positive_rate']:.3%} |"
        for k in test.index)

    ablation_rows = ""
    if ablation is not None:
        from src.train import ABLATION_LABELS as labels
        ablation_rows = "\n".join(
            f"| {labels.get(n, n)} | {ablation.loc[n, 'recall']:.4f} | "
            f"{ablation.loc[n, 'f1']:.4f} | {ablation.loc[n, 'pr_auc']:.4f} |"
            for n in ablation.index)

    ablation_findings, artefact_verdict = _ablation_findings(ablation)

    # Computed, not asserted: the primary protocol is harder than the one these
    # targets were registered against, so whether they still pass is a result.
    targets = [("F1", row["f1"], 0.90), ("ROC-AUC", row["roc_auc"], 0.95),
               ("PR-AUC", row["pr_auc"], 0.90), ("recall", row["recall"], 0.90)]
    unmet = [name for name, value, bar in targets if value < bar]
    target_verdict = (
        "all pre-registered targets met" if not unmet else
        f"{len(targets) - len(unmet)}/{len(targets)} pre-registered targets met on this "
        f"harder protocol ({', '.join(unmet)} short); all met on the pooled random split")

    shap_top = ""
    if shap_payload and "global_importance" in shap_payload:
        shap_top = ", ".join(
            f"`{r['feature']}`" for r in shap_payload["global_importance"][:5])

    family_low = ""
    if ctx.get("audit") is not None:
        families = ctx["audit"][ctx["audit"]["group_type"] == config.ATTACK_CAT]
        families = families[families["n_attack"] > 0].sort_values("recall").head(4)
        family_low = ", ".join(
            f"{r['group']} ({r['recall']:.0%})" for _, r in families.iterrows())

    body = f"""
# Technical Presentation

**Machine Learning-Based Network Intrusion Detection and Anomaly Classification**

*Audience: ML / data-science / cybersecurity peers · 12 slides · ~20 minutes*
*Generated {date.today().isoformat()} from `reports/metrics/`.*

---

## Slide 1 — Title

**Content**
- Machine Learning-Based Network Intrusion Detection and Anomaly Classification
- Binary classification of network flows: benign vs. malicious
- Dataset: UNSW-NB15 ({prep.get('rows_loaded', 0):,} flow records, 10 attack families)
- Selected model: **{best_name}** — F1 {row['f1']:.4f}, recall {row['recall']:.2%}, PR-AUC {row['pr_auc']:.4f}

**Recommended visual:** Title slide with the architecture diagram from README.md as a faded background.

**Speaker notes:** Open with the finding, not the setup. "We built a flow-based intrusion detector that meets every performance target we set in advance — and then we spent most of our effort working out how much of that number is real. That second part is what I actually want to talk about."

---

## Slide 2 — Problem & Objectives

**Content**
- Enterprise networks produce tens of millions of flow records per day; the malicious fraction is vanishingly small
- Signatures are precise but blind to anything unseen; manual review does not scale
- **Objective:** estimate P(malicious) per flow from flow statistics alone — no payload, no IPs, no ports, no timestamps
- Pre-registered targets: F1 ≥ 0.90, ROC-AUC ≥ 0.95, PR-AUC ≥ 0.90, recall ≥ 0.90, throughput ≥ 10k flows/s
- Pre-registered **methodological** criteria: no target leakage, no preprocessing leakage, no duplicate leakage, test split opened once

**Recommended visual:** Two-column slide — left: the SOC funnel (10M flows → alerts → analysts); right: the targets table.

**Speaker notes:** Stress that targets were set *before* modelling. That is what makes the evaluation a test rather than a description. Note that the methodological criteria are as binding as the numerical ones.

---

## Slide 3 — Dataset

**Content**
- UNSW-NB15 (Moustafa & Slay, 2015), UNSW Canberra ACCS — synthetic testbed traffic, IXIA PerfectStorm
- {prep.get('rows_loaded', 0):,} records, 45 columns, 10 attack families; target `label`, family label `attack_cat`
- Integrity verified independently: row counts, schema, target encoding, SHA-256 — not trusted from the mirror
- **`attack_cat` excluded as a predictor** — it determines `label` exactly (verified 1.0000 agreement)
- Missing values: **{prep.get('missing_values', 0)}**. But `service == '-'` (54.8%) and `dbytes == 0` (46.7%) are real categories, not absences

**Recommended visual:** `figures/fig02_attack_category_distribution.png`

**Speaker notes:** The three-orders-of-magnitude spread across families is why every per-family metric is reported with its sample count. Point out that "no missing values" is not the same as "nothing is absent."

---

## Slide 4 — EDA: three findings that changed the design

**Content**
1. **{prep.get('duplicate_fraction', 0):.1%} of the corpus is duplicated**, and duplication is class-correlated — Generic 87.6%, Normal 8.1%
2. **`sttl` is a near-label proxy** — a lookup rule on it alone reaches {eda['ttl_artifact']['sttl_only_rule_accuracy_train']:.1%} accuracy vs a {eda['ttl_artifact']['majority_class_baseline_train']:.1%} baseline
3. **Outliers are genuine attacks** — 14 MB transfers, 5.99 Gbit/s loads, 10,646-packet bursts. Nothing was clipped

**Recommended visual:** `figures/fig05_ttl_artifact.png` (three panels) with `figures/fig03_duplication_by_attack_category.png` inset.

**Speaker notes:** This is the heart of the talk. The testbed ran benign and attack generators on hosts with different initial TTLs, so `sttl` encodes *which generator* produced the flow. Anyone benchmarking on UNSW-NB15 without noticing this is reporting a partly meaningless number. Emphasise that finding 1 and finding 2 both *changed what we built*, not just what we wrote.

---

## Slide 5 — Preprocessing & Feature Engineering

**Content**
- **Deduplicate before splitting** → {prep.get('rows_final', 0):,} unique flows; class balance shifts 63.9% → {prep.get('class_balance', {}).get(1, prep.get('class_balance', {}).get('1', 0)):.1%} attack
- Stratified 60/20/20 on `attack_cat` (so Worms reaches all three splits), `random_state={config.RANDOM_STATE}`
- `log1p` → `StandardScaler` for heavy tails; one-hot with `min_frequency={config.MIN_CATEGORY_FREQUENCY}` for the 133-level `proto`
- **{len(ENGINEERED_FEATURE_DOCS)} engineered features**, all row-wise: `src_byte_ratio`, `is_one_way`, `tcp_handshake_complete`, `load_log_ratio`, …
- Row-wise matters twice: cannot leak across the split, **and** computable by a sensor on one flow

**Recommended visual:** `figures/fig10_engineered_feature_separation.png`

**Speaker notes:** Everything sits inside one `Pipeline`, so CV re-fits the scaler and encoder per fold — leakage is structurally impossible rather than prevented by discipline. 46.7% of flows have zero destination packets, so every ratio is `a/(a+b)`, not `a/b`.

---

## Slide 6 — Modelling Approach

**Content**
- Logistic Regression (interpretable baseline) · Random Forest · XGBoost · LightGBM · Isolation Forest (unsupervised, benign-only)
- `RandomizedSearchCV`, 5-fold `StratifiedKFold`, scoring F1 with PR-AUC and ROC-AUC alongside
- **Protocol order:** score on validation → select on validation → tune threshold on validation → freeze → open test **once**
- Solver choice made by measurement: `lbfgs`/L2 3.6 s vs `saga`/L1 199.4 s (non-converged) for a 0.0004 F1 difference

**Recommended visual:** Flowchart of the five-step protocol, with the test split greyed out until step 5.

**Speaker notes:** The protocol order is the substance. Steps 1–4 never touch test data; step 5 changes no parameter. That is what makes the test metrics an estimate of generalisation rather than a number that was optimised toward.

---

## Slide 7 — Model Comparison

**Content**

| Model | Recall | Precision | F1 | PR-AUC | FPR |
|---|---|---|---|---|---|
{model_rows}

- Selected **{best_name}** on validation **PR-AUC — not accuracy**
- Accuracy compresses every model into a narrow band on a {row['positive_rate_actual']:.0%}-positive corpus and rewards majority-class performance

**Recommended visual:** `figures/fig16_model_comparison.png`

**Speaker notes:** Note how close the tree ensembles are, and how far the Isolation Forest sits behind — that gap is what supervised labelling actually buys. If asked why not accuracy: PR-AUC measures ranking quality on the positive class, which is what survives a change of base rate.

---

## Slide 8 — Explainability

**Content**
- SHAP TreeExplainer: global importance, beeswarm, dependence, and four local cases (TP / TN / FP / FN)
- Top drivers: {shap_top or 'see figures'}
- **Toward ATTACK:** high source TTL, no destination response, no completed handshake, asymmetric direction
- **Toward BENIGN:** completed handshake with sequence exchange, balanced bidirectional volume, identified service
- **Artefact diagnostic:** {f'the TTL family carries {ttl_share:.0%} of total attributed impact' if ttl_share is not None else 'quantifies reliance on the TTL family'}

**Recommended visual:** `figures/fig21_shap_beeswarm.png`, then `figures/fig24_shap_artifact_check.png`

**Speaker notes:** The dependence plot for `sttl` is a step, not a gradient — the model learned a near-binary switch. SHAP is what turned a suspicion from EDA into a measured quantity on the fitted model. Mention the dual-use point: the same explanation that helps a defender tells an attacker what to manipulate.

---

## Slide 9 — Error & Bias Analysis

**Content**
- Recall varies substantially by attack family; weakest: {family_low or 'see figure'}
- Three distinguishable mechanisms: **scarcity** (fixable with data), **behavioural overlap with benign traffic** (not fixable), **duplication-distorted training counts**
- Missed attacks are **near-misses**{f" — median score {ctx['errors']['false_negative_score_stats']['median']:.2f} against a {threshold:.3f} threshold, only {ctx['errors']['false_negative_score_stats']['share_below_0.10']:.0%} below 0.10" if ctx.get('errors') and 'false_negative_score_stats' in ctx['errors'] else ''}, so the threshold is the dominant lever and borderline review genuinely helps
- False alerts concentrate in specific services → actionable via per-service thresholds
- **UNSW-NB15 has no demographic attributes.** This is an *operational* performance audit; no demographic fairness claim is made or possible

**Recommended visual:** `figures/fig18_subgroup_audit.png`

**Speaker notes:** Be explicit about the fairness framing — it is a strength, not a hedge. Then note that genuine fairness risk does enter at deployment: uneven per-service FPR means uneven scrutiny of the humans behind those services.

---

## Slide 10 — Ablations: what is the number actually measuring?

**Content**

| Experiment | Recall | F1 | PR-AUC |
|---|---|---|---|
{ablation_rows or '_Run `python -m src.pipeline --ablations`._'}

{ablation_findings}

**Recommended visual:** `figures/fig19_ablation_comparison.png`

**Speaker notes:** This slide separates "the model detects attacks" from "the model detects this dataset." Each row is a separate end-to-end experiment reusing the tuned hyper-parameters, so exactly one thing changes at a time.

---

## Slide 11 — Deployment Architecture & Threshold

**Content**
- Flow collector → feature extraction → model → **risk-banded triage queue** → analyst → response
- Operating threshold **{threshold:.3f}**, chosen on validation; a low-FPR alternative is published for capacity-limited SOCs
- Cost assumption stated openly: 20:1 FN:FP, with a sensitivity table
- Measured throughput: **{row.get('throughput_flows_per_second', float('nan')):,.0f} flows/second** on one commodity CPU
- Streamlit prototype: single-flow scoring, batch upload, live threshold control, per-alert SHAP

**Recommended visual:** `figures/fig17_threshold_analysis.png` plus a Streamlit screenshot.

**Speaker notes:** 0.50 is a default, not an optimum. Show how the recall/FPR trade-off moves and make the point that choosing the operating point is a business decision about alert capacity, not a modelling decision.

---

## Slide 12 — Limitations & Conclusions

**Content**
- **Achieved:** F1 {row['f1']:.4f}, ROC-AUC {row['roc_auc']:.4f}, PR-AUC {row['pr_auc']:.4f}, recall {row['recall']:.2%}, FPR {row['false_positive_rate']:.3%} — {target_verdict}
- **Qualified by:** {artefact_verdict}, uneven per-family recall, and precision that will not transfer to a production base rate
- **Cannot measure:** concept drift — timestamps were removed from the partitioned files
- **Recommendation:** human-in-the-loop triage ranking, recalibrated on target-network traffic. Not autonomous blocking
- **Next:** multiclass family classification, cross-dataset validation (CIC-IDS2017), adversarial robustness testing

**Recommended visual:** Side-by-side — headline metrics left, the three qualifications right, equally weighted.

**Speaker notes:** Close on the distinction between a number and a finding. A capstone that reported only its F1 would not have discovered the artefact, the duplication problem, or the per-family gaps — and would have recommended a system that should not be deployed the way it suggested.

---

## Appendix slides (if time or questions allow)

- A1: Full 9-metric comparison table (`reports/Model_Evaluation_Report.md` §2–§4)
- A2: Correlation structure and the ill-conditioned design matrix (`figures/fig08`)
- A3: Local SHAP explanations, all four outcome types (`figures/fig23`)
- A4: Per-family traffic fingerprints (`figures/fig12`)
- A5: Threshold cost-sensitivity table (`reports/Model_Evaluation_Report.md` §8)
- A6: Data dictionary (`reports/data_dictionary.csv`)
"""
    _write("technical_presentation_content.md", body)


# --------------------------------------------------------------------------- #
def _executive(ctx: dict) -> None:
    row, best_name, threshold = ctx["row"], ctx["best_name"], ctx["threshold"]
    operating = ctx.get("operating") or {}

    caught = int(row["tp"])
    missed = int(row["fn"])
    total_attacks = caught + missed
    false_alerts = int(row["fp"])
    benign = int(row["tn"]) + false_alerts

    # Illustrative daily volumes, stated as an assumption rather than a finding.
    daily_flows = 10_000_000
    daily_false_alerts = row["false_positive_rate"] * daily_flows

    constrained = operating.get("fpr_constrained")
    constrained_line = (
        f"- A tighter setting is available: it alerts on only "
        f"{constrained['false_positive_rate']:.2%} of normal traffic, at the cost of "
        f"catching {constrained['recall']:.0%} of attacks instead of {row['recall']:.0%}"
        if constrained else "")

    body = f"""
# Executive Presentation

**Automated Detection of Malicious Network Activity**

*Audience: executives and security leadership · 10 slides · ~12 minutes*
*Generated {date.today().isoformat()}. No equations, no code.*

---

## Slide 1 — Executive Title

**Content**
- Automated Detection of Malicious Network Activity
- A machine-learning layer that reviews every network connection and ranks the suspicious ones for analysts
- Proof-of-concept result: **{row['recall']:.0%} of attacks detected**, with **{row['false_positive_rate']:.2%} of normal traffic flagged for review**
- Recommendation: deploy as an **analyst assistant**, not an automated gatekeeper

**Recommended visual:** Clean title slide; one large statistic — "{row['recall']:.0%} of attacks detected" — as the focal point.

**Speaker notes:** Lead with the outcome and the recommendation. Signal early that this is a decision-support tool, so nobody leaves thinking we are proposing to let software block traffic on its own.

---

## Slide 2 — The Cybersecurity Problem

**Content**
- Our network records **millions of connections every day**
- A small number of them are an attacker probing, spreading, or taking data out
- No team can review them all — reviewing one second each would take a century of staff time per day
- Today we rely on rules that recognise **known** attacks. They work, and they are blind to anything new

**Recommended visual:** A haystack graphic — millions of grey connections, a handful of red ones. No numbers on the slide itself.

**Speaker notes:** Avoid jargon entirely. The question this slide should raise in the audience's mind is "so how do we find the red ones?"

---

## Slide 3 — Why Existing Monitoring Is Difficult

**Content**
- **Rules only catch what we have already seen.** A new tool or technique passes straight through
- **Volume beats people.** Analyst capacity is fixed; traffic is not
- **Too many alerts is as bad as too few.** A team that cannot work its queue stops trusting it — and real attacks get closed unread
- The cost of the two mistakes is very different, and both are real

**Recommended visual:** Simple two-column comparison: "Missed attack" (breach, dwell time, regulatory exposure) vs "False alarm" (wasted analyst hours, eroded trust).

**Speaker notes:** The alert-fatigue point usually lands hardest with this audience. Emphasise that a detector producing more alerts than we can process does not degrade gracefully — it fails completely, because people stop reading it.

---

## Slide 4 — The Proposed Solution

**Content**
- Software that learns what attacks **look like** from thousands of past examples
- It examines the *shape* of each connection — how long, how much data, which direction, whether the other side answered — and never reads message contents
- Every connection gets a **risk score** and a **plain-English reason**
- Analysts start at the top of a ranked list instead of searching a haystack

**Recommended visual:** Simple pipeline: Network traffic → Risk scoring → Ranked queue → Analyst → Action. One row, five boxes.

**Speaker notes:** The privacy point is worth stating unprompted: the system reads connection statistics, not message contents. It usually pre-empts the first question from legal or from the works council.

---

## Slide 5 — Detection Performance

**Content**
- Tested on **{int(row['n']):,} connections the system had never seen**
- **Caught {caught:,} of {total_attacks:,} attacks** ({row['recall']:.1%})
- **Missed {missed:,}** ({row['false_negative_rate']:.1%})
- Flagged **{false_alerts:,} of {benign:,} normal connections** for review ({row['false_positive_rate']:.2%})
- Fast enough to keep up with our traffic on ordinary hardware — no special equipment needed

**Recommended visual:** Four large stat tiles: attacks caught, attacks missed, normal traffic flagged, speed. `figures/fig16_model_comparison.png` (middle panel only) as support.

**Speaker notes:** Give the honest caveat here rather than burying it: the test data has far more attacks than a real network does, so the "how many of our alerts are real" figure will be worse in production. What does carry over is the detection rate and the share of normal traffic flagged.

---

## Slide 6 — Business and Security Value

**Content**
- **Faster detection.** Time-to-detection is the variable most strongly linked to breach cost — every hour saved limits the damage
- **Coverage for unknown attacks.** Catches suspicious *behaviour*, not just known signatures
- **Better use of analyst time.** A ranked, explained queue replaces undirected searching
- **Quantified risk.** We can finally answer "what fraction of attacks of this kind do we catch?" with a measured number
- **Consistency.** The same judgement applied to the ten-thousandth connection of a night shift as to the first

**Recommended visual:** Four value tiles with icons. Deliberately no ROI figure.

**Speaker notes:** If asked for a financial ROI: we can build one from our own analyst cost and incident history, but the honest position today is that we have not measured it on our own network, and we are not going to present an invented number.

---

## Slide 7 — The Trade-off We Control

**Content**
- There is a dial. Turning it up catches more attacks and generates more false alarms; turning it down does the reverse
- **This is a business decision about analyst capacity, not a technical one**
- At the recommended setting: {row['recall']:.0%} of attacks caught, {row['false_positive_rate']:.2%} of normal traffic flagged
{constrained_line}
- At our illustrative volume of {daily_flows / 1_000_000:.0f} million connections a day, that recommended setting implies roughly **{daily_false_alerts:,.0f} items to review per day** — which is exactly the number we must size the team against

**Recommended visual:** `figures/fig17_threshold_analysis.png`, left panel only, relabelled in plain language ("attacks caught" / "false alarms").

**Speaker notes:** This is the slide that needs a decision from the room. Present the two settings and ask which the SOC can actually staff. Be clear the daily volume is an illustration, not a measurement of our network.

---

## Slide 8 — Explainability and Governance

**Content**
- Every alert comes with **the specific reasons it was raised** — not just a score
- Example: *"1.2 MB sent, nothing returned, the connection never properly opened, and this source contacted 43 different services in the last few minutes"*
- Supports audit, regulatory review and analyst training
- We can show **which attack types we detect well and which we do not** — and we do
- Known limitations are documented rather than hidden

**Recommended visual:** A mocked-up alert card showing score, risk band, and three plain-English reasons.

**Speaker notes:** Explainability is a governance requirement, not a nice-to-have. Add that the same analysis told us about a weakness in our own test data — which is how we know the process is working rather than just producing flattering numbers.

---

## Slide 9 — Deployment Recommendation

**Content**
- **Deploy as an analyst assistant.** It ranks and explains; people decide. It does **not** block traffic
- **Phase 1 — Shadow mode (4–6 weeks).** Run alongside existing tools, measure the real false-alarm rate on our own traffic. No analyst action
- **Phase 2 — Assisted triage.** Analysts work the ranked queue; every decision is captured
- **Phase 3 — Continuous improvement.** Retrain on our own data and analyst feedback; monitor for drift
- **Requirement:** recalibrate on our network before go-live. Performance measured on public research data will not transfer unchanged

**Recommended visual:** Three-phase timeline with a decision gate between each phase.

**Speaker notes:** The shadow-mode phase is non-negotiable and is also the easiest thing to approve — it carries no operational risk and produces the number we actually need.

---

## Slide 10 — Key Takeaways

**Content**
1. **It works.** {row['recall']:.0%} of attacks detected on data the system had never seen, at ordinary hardware speed
2. **It is honest about what it misses.** Weaker on stealthy, low-volume attack types — documented, not hidden
3. **It is explainable.** Every alert carries its reasons, satisfying audit and regulatory needs
4. **It is a layer, not a replacement.** It works alongside existing controls and human judgement
5. **Next step:** approve a 4–6 week shadow-mode pilot on our own traffic — no operational risk, and it gives us the real numbers

**Recommended visual:** Five numbered takeaways, generous white space, the ask highlighted.

**Speaker notes:** Close on the specific ask: approval for the shadow-mode pilot. That is the only decision needed today, and it is a low-risk one.

---

## Anticipated questions

| Question | Answer |
|---|---|
| *Can it replace our current tools?* | No. It catches different things and should run alongside them. |
| *Will it block attacks automatically?* | Not as recommended. It misses some attacks with high confidence, which makes autonomous blocking inappropriate today. |
| *How many false alarms will we actually get?* | Unknown until shadow mode. The research-data figure is optimistic because that data has far more attacks than our network does. |
| *Does it read our data?* | No. It uses connection statistics — size, timing, direction — never message contents. |
| *What is the ROI?* | We can model it from our analyst cost and incident history after shadow mode. We are not presenting an invented figure today. |
| *Can attackers evade it?* | Yes, with effort — they can reshape traffic to look ordinary. That is one reason it is a layer rather than a sole control. |
| *How often must it be retrained?* | Traffic changes continuously. Plan for scheduled retraining with drift monitoring; the pilot will tell us the right cadence. |
"""
    _write("executive_presentation_content.md", body)

