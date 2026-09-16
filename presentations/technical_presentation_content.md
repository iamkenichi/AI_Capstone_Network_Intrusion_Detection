# Technical Presentation

**Machine Learning-Based Network Intrusion Detection and Anomaly Classification**

*Audience: ML / data-science / cybersecurity peers · 12 slides · ~20 minutes*
*Generated 2026-09-16 from `reports/metrics/`.*

---

## Slide 1 — Title

**Content**
- Machine Learning-Based Network Intrusion Detection and Anomaly Classification
- Binary classification of network flows: benign vs. malicious
- Dataset: UNSW-NB15 (257,673 flow records, 10 attack families)
- Selected model: **XGBoost** — F1 0.9109, recall 91.12%, PR-AUC 0.9791

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
- 257,673 records, 45 columns, 10 attack families; target `label`, family label `attack_cat`
- Integrity verified independently: row counts, schema, target encoding, SHA-256 — not trusted from the mirror
- **`attack_cat` excluded as a predictor** — it determines `label` exactly (verified 1.0000 agreement)
- Missing values: **0**. But `service == '-'` (54.8%) and `dbytes == 0` (46.7%) are real categories, not absences

**Recommended visual:** `figures/fig02_attack_category_distribution.png`

**Speaker notes:** The three-orders-of-magnitude spread across families is why every per-family metric is reported with its sample count. Point out that "no missing values" is not the same as "nothing is absent."

---

## Slide 4 — EDA: three findings that changed the design

**Content**
1. **40.4% of the corpus is duplicated**, and duplication is class-correlated — Generic 87.6%, Normal 8.1%
2. **`sttl` is a near-label proxy** — a lookup rule on it alone reaches 81.3% accuracy vs a 55.6% baseline
3. **Outliers are genuine attacks** — 14 MB transfers, 5.99 Gbit/s loads, 10,646-packet bursts. Nothing was clipped

**Recommended visual:** `figures/fig05_ttl_artifact.png` (three panels) with `figures/fig03_duplication_by_attack_category.png` inset.

**Speaker notes:** This is the heart of the talk. The testbed ran benign and attack generators on hosts with different initial TTLs, so `sttl` encodes *which generator* produced the flow. Anyone benchmarking on UNSW-NB15 without noticing this is reporting a partly meaningless number. Emphasise that finding 1 and finding 2 both *changed what we built*, not just what we wrote.

---

## Slide 5 — Preprocessing & Feature Engineering

**Content**
- **Deduplicate before splitting** → 153,684 unique flows; class balance shifts 63.9% → 44.4% attack
- Stratified 60/20/20 on `attack_cat` (so Worms reaches all three splits), `random_state=42`
- `log1p` → `StandardScaler` for heavy tails; one-hot with `min_frequency=200` for the 133-level `proto`
- **15 engineered features**, all row-wise: `src_byte_ratio`, `is_one_way`, `tcp_handshake_complete`, `load_log_ratio`, …
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
| Logistic Regression | 0.9134 | 0.8106 | 0.8590 | 0.9415 | 17.028% |
| Random Forest | 0.9137 | 0.9010 | 0.9073 | 0.9773 | 8.008% |
| XGBoost | 0.9112 | 0.9106 | 0.9109 | 0.9791 | 7.142% |
| LightGBM | 0.9134 | 0.9089 | 0.9112 | 0.9791 | 7.306% |
| Isolation Forest | 0.2199 | 0.6719 | 0.3314 | 0.5912 | 8.570% |

- Selected **XGBoost** on validation **PR-AUC — not accuracy**
- Accuracy compresses every model into a narrow band on a 44%-positive corpus and rewards majority-class performance

**Recommended visual:** `figures/fig16_model_comparison.png`

**Speaker notes:** Note how close the tree ensembles are, and how far the Isolation Forest sits behind — that gap is what supervised labelling actually buys. If asked why not accuracy: PR-AUC measures ranking quality on the positive class, which is what survives a change of base rate.

---

## Slide 8 — Explainability

**Content**
- SHAP TreeExplainer: global importance, beeswarm, dependence, and four local cases (TP / TN / FP / FN)
- Top drivers: `sttl`, `ct_dst_src_ltm`, `ct_state_ttl`, `dbytes`, `service_dns`
- **Toward ATTACK:** high source TTL, no destination response, no completed handshake, asymmetric direction
- **Toward BENIGN:** completed handshake with sequence exchange, balanced bidirectional volume, identified service
- **Artefact diagnostic:** the TTL family carries 54% of total attributed impact

**Recommended visual:** `figures/fig21_shap_beeswarm.png`, then `figures/fig24_shap_artifact_check.png`

**Speaker notes:** The dependence plot for `sttl` is a step, not a gradient — the model learned a near-binary switch. SHAP is what turned a suspicion from EDA into a measured quantity on the fitted model. Mention the dual-use point: the same explanation that helps a defender tells an attacker what to manipulate.

---

## Slide 9 — Error & Bias Analysis

**Content**
- Recall varies substantially by attack family; weakest: Analysis (59%), Fuzzers (74%), Shellcode (96%), Exploits (98%)
- Three distinguishable mechanisms: **scarcity** (fixable with data), **behavioural overlap with benign traffic** (not fixable), **duplication-distorted training counts**
- Missed attacks are **near-misses** — median score 0.35 against a 0.51 threshold, only 2% below 0.10, so the threshold is the dominant lever and borderline review genuinely helps
- False alerts concentrate in specific services → actionable via per-service thresholds
- **UNSW-NB15 has no demographic attributes.** This is an *operational* performance audit; no demographic fairness claim is made or possible

**Recommended visual:** `figures/fig18_subgroup_audit.png`

**Speaker notes:** Be explicit about the fairness framing — it is a strength, not a hedge. Then note that genuine fairness risk does enter at deployment: uneven per-service FPR means uneven scrutiny of the humans behind those services.

---

## Slide 10 — Ablations: what is the number actually measuring?

**Content**

| Experiment | Recall | F1 | PR-AUC |
|---|---|---|---|
| Primary protocol | 0.9112 | 0.9109 | 0.9791 |
| Duplicates kept | 0.9630 | 0.9607 | 0.9957 |
| TTL removed | 0.9075 | 0.9111 | 0.9789 |
| No engineered features | 0.9158 | 0.9118 | 0.9794 |
| SMOTE | 0.9124 | 0.9103 | 0.9789 |
| Published split | 0.9868 | 0.8881 | 0.9885 |

- **Duplicates retained → F1 +0.0498.** Skipping deduplication would have handed back a better-looking number for no better model
- **TTL removed → F1 +0.0003.** SHAP's top feature turns out to be *redundant, not necessary* — the model re-routes through correlated features
- **No engineered features → F1 +0.0010.** The 15 domain features buy interpretability, not accuracy — reported as the negative result it is
- **SMOTE → F1 -0.0005.** Tested rather than assumed, and not adopted
- **Published split → precision 0.911 → 0.807.** Same model, different partition, different detector

**Recommended visual:** `figures/fig19_ablation_comparison.png`

**Speaker notes:** This slide separates "the model detects attacks" from "the model detects this dataset." Each row is a separate end-to-end experiment reusing the tuned hyper-parameters, so exactly one thing changes at a time.

---

## Slide 11 — Deployment Architecture & Threshold

**Content**
- Flow collector → feature extraction → model → **risk-banded triage queue** → analyst → response
- Operating threshold **0.51**, chosen on validation; a low-FPR alternative is published for capacity-limited SOCs
- Cost assumption stated openly: 20:1 FN:FP, with a sensitivity table
- Measured throughput: **195,353 flows/second** on one commodity CPU
- Streamlit prototype: single-flow scoring, batch upload, live threshold control, per-alert SHAP

**Recommended visual:** `figures/fig17_threshold_analysis.png` plus a Streamlit screenshot.

**Speaker notes:** 0.50 is a default, not an optimum. Show how the recall/FPR trade-off moves and make the point that choosing the operating point is a business decision about alert capacity, not a modelling decision.

---

## Slide 12 — Limitations & Conclusions

**Content**
- **Achieved:** F1 0.9109, ROC-AUC 0.9824, PR-AUC 0.9791, recall 91.12%, FPR 7.142% — all pre-registered targets met
- **Qualified by:** a capture artefact the model uses but does not need (shown by ablation, not assumed), uneven per-family recall, and precision that will not transfer to a production base rate
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
