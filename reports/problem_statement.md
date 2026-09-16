# Problem Statement

**Project:** Machine Learning-Based Network Intrusion Detection and Anomaly Classification
**Domain:** Cybersecurity — Security Operations Centre (SOC) detection engineering
**Task type:** Supervised **binary classification** (benign vs. malicious network flow)
**Dataset:** UNSW-NB15 (Moustafa & Slay, 2015), partitioned train/test CSVs

---

## 1. The security problem

A mid-sized enterprise network produces on the order of tens of millions of
network flow records per day. Each record is a summary of one conversation
between two hosts: how long it lasted, how many bytes and packets moved in each
direction, which protocol and service were involved, and how the connection
terminated. Buried in that volume is a very small number of flows that represent
reconnaissance, exploitation, malware command-and-control, or data theft.

Three properties of this problem make manual analysis structurally impossible,
not merely expensive:

1. **Volume.** No analyst team can read tens of millions of records a day. Even
   at one second per record, a single day's telemetry would take a century of
   analyst-hours.
2. **Base rate.** Malicious flows are a vanishingly small fraction of the total.
   Any detection method with a non-trivial false-positive rate will bury the
   true positives — the classic base-rate problem that Axelsson (2000) showed is
   the binding constraint on intrusion detection, not detection accuracy.
3. **Adaptation.** Signature-based detection matches known-bad patterns.
   It is precise and explainable, and it fails completely against any technique
   for which a signature has not yet been written. Attackers change tooling
   faster than signature libraries are updated.

The operational consequence is a pair of failure modes that a SOC lives between:

- **Missed detections (false negatives).** An intrusion proceeds unchallenged.
  Dwell time — the interval between compromise and detection — is the single
  variable most strongly associated with breach cost, because it determines how
  far an attacker gets before anyone intervenes.
- **Alert fatigue (false positives).** A detector that alerts on 1% of benign
  traffic in a network carrying 10 million benign flows a day generates 100,000
  alerts a day. A team that cannot work its queue stops trusting it, and
  genuine detections are closed unread. False positives do not merely waste
  time; past a threshold they destroy the detection capability outright.

### The gap this project addresses

Between "write a signature for every known attack" and "read every flow by
hand" there is room for a statistical layer: a model that learns the
*behavioural shape* of malicious traffic from labelled examples, scores every
flow automatically, and hands analysts a ranked, explained shortlist. That is
what this project builds and — more importantly — what it measures honestly.

---

## 2. Data-science objective

> Given the flow-level features of a single network connection, and **without
> access to payload contents, IP addresses, port numbers, or timestamps**,
> estimate the probability that the connection is malicious, and produce a
> calibrated, explainable, thresholded decision that a SOC can operate.

**Task type.** Supervised binary classification. The target `label` takes value
0 for benign and 1 for malicious. The dataset's ten-level `attack_cat` column is
used for stratification, subgroup auditing and error analysis, and is
**explicitly excluded from the feature matrix**: because `attack_cat == "Normal"`
if and only if `label == 0` (verified at 1.0000 agreement across all 257,673
records), including it as a predictor would be textbook target leakage.

**Unit of analysis.** One bidirectional network flow record.

**Learning paradigm.** Primarily supervised, because labels exist. An
unsupervised Isolation Forest is trained on benign traffic only and evaluated
alongside, to quantify what the labelling effort actually buys and to establish
the floor a supervised system must clear.

---

## 3. Primary research question

> **Can supervised machine-learning models accurately distinguish malicious from
> benign network traffic while maintaining an operationally acceptable
> false-negative and false-positive rate?**

Three subsidiary questions follow from it, and this project answers each with
measured evidence rather than assertion:

- **RQ1.** Which model family — interpretable linear, bagged trees, or gradient
  boosting — offers the best detection-quality/operational-cost trade-off?
- **RQ2.** *Which* attacks are missed, and is the miss rate concentrated in
  particular attack families, protocols or services?
- **RQ3.** How much of the measured performance reflects genuine attack
  behaviour, and how much reflects artefacts of how UNSW-NB15 was generated?
  (This question is unusual in student work and is the one that most determines
  whether the headline numbers mean anything outside the dataset.)

---

## 4. Success criteria

### 4.1 Technical targets

These are **targets set in advance, not results**. They are stated here so that
the evaluation can be judged against a pre-registered bar rather than against
whatever the models happened to achieve. Where a target is not met, the report
says so.

| # | Criterion | Target | Rationale |
|---|---|---|---|
| T1 | F1 (attack class) | ≥ 0.90 | Balanced evidence that neither precision nor recall was sacrificed. |
| T2 | ROC-AUC | ≥ 0.95 | Threshold-free separability. |
| T3 | PR-AUC (average precision) | ≥ 0.90 | The more honest threshold-free measure on the positive class. |
| T4 | Attack recall at the deployed threshold | ≥ 0.90 | Direct statement of how many intrusions are caught. |
| T5 | False-negative rate | Reported explicitly, never hidden inside accuracy | The metric that maps to residual security risk. |
| T6 | False-positive rate | Reported explicitly, with an alternative low-FPR operating point | The metric that maps to analyst workload. |
| T7 | Inference throughput | ≥ 10,000 flows/second on commodity CPU | A detector slower than the network is not deployable. |

### 4.2 Methodological criteria

Equally binding, and in a capstone arguably more so:

| # | Criterion |
|---|---|
| M1 | No target leakage: `attack_cat`, `label` and `id` never enter the feature matrix. |
| M2 | No preprocessing leakage: every scaler and encoder is fitted inside the training fold only. |
| M3 | No duplicate leakage: identical feature vectors never appear in both train and test. |
| M4 | The test split is scored **once**, after the model and threshold are frozen. |
| M5 | The decision threshold is tuned on a **validation** split, never on test. |
| M6 | Every reported number is produced by executed code and written to `reports/metrics/`. |
| M7 | Dataset artefacts that inflate performance are identified, quantified and reported. |

### 4.2b Declared deviation from pre-registration

Pre-registration is only worth anything if departures from it are declared, so
this one is declared here rather than left for a reader to notice.

T1–T7 above were registered against an earlier evaluation protocol: pool the
published training and test files, deduplicate the pooled corpus, and draw a
stratified 60/20/20 random split. **The primary protocol was subsequently
changed** to the authors' published partition — each side deduplicated
independently, a validation set carved from the published training file, and any
test record whose feature vector also occurs in development removed.

The change was made because the two protocols are not equally informative. A
random split *guarantees* that the test set is drawn from the training
distribution. The published partition does not, and the gap between them is the
part of a benchmark score that would not survive contact with a different
network — which is exactly what an intrusion detector needs to be judged on.

Two consequences, both handled explicitly rather than quietly:

1. **The targets were not rebased.** T1–T7 are reported against the harder
   protocol at the values originally registered. Lowering a pre-registered bar
   to fit the result it was written to test would defeat its purpose. Where a
   target is not met, the report says so.
2. **The old protocol is still run**, as the `pooled_random` ablation, so the
   cost of the change is visible as a number rather than asserted as a
   principle.

### 4.3 What would count as failure

Stated explicitly, because a success criterion with no failure condition is not
a criterion:

- Meeting T1–T4 while being unable to explain *why* the model works — a black
  box that cannot be interrogated cannot be deployed in a regulated environment.
- Achieving high scores that collapse under the `no_ttl` or `keep_duplicates`
  ablations, without saying so.
- Reporting precision from this dataset as though it would transfer to a
  production base rate.

---

## 5. Business and security value

**Triage compression.** The realistic value of an ML detection layer is not
replacing analysts but shrinking what they look at. A model that ranks flows by
attack probability turns "review 10 million flows" into "review the top few
thousand, in probability order, each annotated with why it was flagged."

**Coverage beyond signatures.** A behavioural model can flag traffic whose
*shape* matches attack activity even when no signature exists for the specific
tool. It complements signature detection; it does not replace it.

**Consistency.** A model applies the same criteria to the 10,000th flow of a
night shift as to the first.

**Quantified residual risk.** Perhaps the most underrated benefit: a measured
false-negative rate turns "are we secure?" into "we detect X% of attacks of this
shape, and here is which families we miss." That is a statement a CISO can put
in front of a board and act on.

### Cost of each error type

| Error | Immediate cost | Second-order cost |
|---|---|---|
| **False positive** | Analyst triage time (minutes each) | At volume: queue overflow, desensitisation, real alerts closed unread |
| **False negative** | None visible — which is the danger | Attacker dwell time, lateral movement, exfiltration, regulatory exposure, incident-response and breach cost |

These costs are **asymmetric but not unboundedly so**. A missed intrusion is far
worse than a false alert, but a detector tuned as though false positives were
free alerts on everything and becomes useless. This project therefore treats the
threshold as an explicit, tunable business decision (Phase 10) rather than
leaving it at the default 0.50, evaluates it under a stated 20:1
false-negative-to-false-positive cost assumption, and reports the sensitivity of
that choice to the assumption.

---

## 6. Explainability requirement

A score with no reason attached is operationally useless and, in regulated
sectors, inadmissible:

- **Analyst workflow.** An analyst who receives "flow 4,812,993: malicious,
  0.94" cannot act. One who receives "malicious, 0.94 — because the source sent
  1.2 MB while the destination returned nothing, no TCP handshake completed, and
  the source contacted 43 distinct services in the preceding window" can
  immediately confirm or dismiss it.
- **Model governance.** EU AI Act transparency obligations, financial-sector
  model risk management (e.g. SR 11-7), and internal audit all require that an
  automated decision be explicable.
- **Debugging.** Explanations are how this project discovered that its strongest
  feature is partly a capture artefact. Without SHAP, that finding would have
  been invisible behind a good F1 score.

SHAP is therefore treated as a core deliverable (Phase 8), not an optional extra.

---

## 7. Constraints, risks and known limitations

### 7.1 Dataset limitations

| Limitation | Consequence |
|---|---|
| **Synthetic traffic.** Generated on a 2015 IXIA PerfectStorm testbed, not captured from a real network. | Benign traffic lacks the messiness of real user behaviour; the model may be separating "generator A from generator B" rather than "benign from malicious". |
| **Inverted class balance.** Attacks are 63.9% of the published corpus (44.4% after deduplication). | Precision measured here will **not** transfer to a production base rate. Recall and FPR will. |
| **TTL artefact.** The benign and attack generators used different initial TTL values, making `sttl` a near-label proxy. | Quantified with SHAP and an explicit `no_ttl` ablation rather than hand-waved. |
| **40.4% duplicate records.** Duplication is class-correlated (Generic 87.6%, Normal 8.1%). | Handled by deduplicating each published partition before splitting, and by removing test records whose feature vector also occurs in development; the cost of skipping either is quantified by ablation. |
| **No timestamps, IPs or ports** in the partitioned files. | **No temporal holdout is possible**, so concept drift cannot be measured — only reasoned about. No host-level or entity-grouped analysis is possible. |
| **2015 vintage.** The attack taxonomy predates modern ransomware, living-off-the-land, cloud-native and supply-chain techniques. | Conclusions are about method, not about current threat coverage. |

### 7.2 Concept drift

Networks are non-stationary in a way that most ML application domains are not.
Benign traffic changes as the organisation adopts new applications; malicious
traffic changes because adversaries deliberately change it. A model trained on
2015 flow statistics would degrade against 2026 traffic for both reasons.

This project cannot measure drift — the timestamps needed to do so were removed
from the partitioned files. It can and does: state the limitation plainly,
design a monitoring and retraining regime for deployment, and avoid claiming
temporal generalisation it has not tested.

### 7.3 Adversarial adaptation

Unlike almost every other ML application, an intrusion detector faces an
opponent who is *trying to defeat it*. An attacker who learns that the model
keys on flow duration, packet rate and directionality can pad packets, throttle
rates, and inject decoy reverse traffic to move a flow across the decision
boundary. This is not hypothetical — mimicry attacks against flow-based
detectors are a documented research area.

Consequences carried through the project: the model is positioned as **one layer
of a defence-in-depth stack, never as a sole control**; features that are cheap
for an attacker to manipulate are identified in the SHAP analysis; and the
deployment recommendation includes human review rather than automated blocking.

### 7.4 Ethical and fairness constraints

UNSW-NB15 contains **no demographic attributes and no human subjects** — it is
synthetic machine-generated traffic with all IP addresses and ports removed. It
is therefore not possible, and would be dishonest, to make demographic fairness
claims from it.

What *is* possible and is performed: an **operational subgroup performance
audit** across protocol, service, connection state and attack family, measuring
whether detection quality is uniform across the strata that actually exist.
Section 5 of `reports/Bias_Fairness_Analysis.md` states this distinction
explicitly and identifies the real fairness risk in deployment — that a detector
with uneven per-service error rates systematically over-flags the teams whose
normal work looks unusual.

---

## 8. Linkage to the machine-learning lifecycle

| Lifecycle stage | This project | Artefact |
|---|---|---|
| **Problem framing** | Security problem → binary classification; pre-registered success criteria; cost asymmetry made explicit | This document |
| **Data acquisition** | Programmatic download with independent integrity verification (row counts, schema, target encoding, SHA-256) | `src/data_loader.py`, `reports/dataset_documentation.md` |
| **Data understanding** | Full profiling; duplicate, defect and artefact discovery; per-column dictionary | `reports/data_dictionary.csv`, `notebooks/01` |
| **EDA** | 13 figures with written interpretations; outliers analysed, not deleted | `figures/fig01`–`fig13`, `notebooks/02` |
| **Preprocessing** | Leak-free `Pipeline`; published partition, each side deduplicated, train/test overlap removed; 80/20 development split stratified on `attack_cat` at `random_state=42` | `src/preprocessing.py`, `notebooks/03` |
| **Feature engineering** | 15 row-wise, deployable, domain-motivated features | `src/features.py` |
| **Feature selection & reduction** | Mutual-information filter and PCA, fitted on training data only, run as a supplement after the model was frozen | `src/feature_analysis.py` |
| **Modelling** | 4 supervised families + 1 unsupervised; randomised search under 5-fold stratified CV | `src/train.py`, `notebooks/04` |
| **Evaluation** | 9-metric comparison; ROC/PR curves; confusion matrices; threshold analysis on validation | `src/evaluate.py`, `notebooks/05` |
| **Explainability** | Global, directional and local SHAP; artefact diagnostic | `src/explain.py`, `notebooks/06` |
| **Error & bias audit** | Per-family recall, per-service FPR, error-profile analysis, mitigations | `reports/Bias_Fairness_Analysis.md` |
| **Deployment** | Frozen model + threshold, inference API, Streamlit prototype, operating model | `src/predict.py`, `app/streamlit_app.py` |
| **Monitoring** | Drift-detection and retraining regime specified (not implemented — no temporal data) | `reports/Final_Project_Report.md` §19 |

The loop closes where it should: the explainability stage fed a finding (the TTL
artefact) back into the experimental design (the `no_ttl` ablation), which in
turn changed the deployment recommendation.

---

## 9. Scope

**In scope:** binary benign/attack classification on flow records; model
comparison and selection; threshold optimisation; explainability; operational
subgroup auditing; a deployment prototype; full reproducibility.

**Out of scope:** multiclass attack-family classification (the data supports it
and it is noted as an extension, but the rubric's primary target is binary);
real-time packet capture and feature extraction; automated blocking or response;
deployment to production infrastructure; deep-learning architectures.

---

## References

Axelsson, S. (2000). The base-rate fallacy and the difficulty of intrusion
detection. *ACM Transactions on Information and System Security*, 3(3), 186–205.

Moustafa, N. and Slay, J. (2015). UNSW-NB15: a comprehensive data set for
network intrusion detection systems. *MilCIS 2015*, IEEE.

Moustafa, N. and Slay, J. (2016). The evaluation of Network Anomaly Detection
Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with
the KDD99 data set. *Information Security Journal: A Global Perspective*,
25(1–3), 18–31.
