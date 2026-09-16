# Generative AI Usage Disclosure

*Last updated 2026-09-16.*

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
