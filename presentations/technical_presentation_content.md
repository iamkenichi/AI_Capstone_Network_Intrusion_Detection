# Technical presentation content

12 slides. Metrics come from the executed run; source files are named below.

## Slide 1 — Network intrusion detection

### Main content

- Machine Learning-Based Network Intrusion Detection and Anomaly Classification
- UNSW-NB15 · Binary supervised classification

### Recommended visual

Project title and concise workflow

### Speaker notes

Introduce the research question and distinguish binary detection from attack-family classification.

## Slide 2 — Problem and objectives

### Main content

- Prioritize suspicious flows for SOC analysts.
- Research targets: F1 >= 0.90; ROC-AUC >= 0.95.
- Report missed attacks and false-alert burden.

### Recommended visual

False-positive versus false-negative decision table

### Speaker notes

Targets are provisional and are assessed against executed evidence. A flow alert is not an incident.

## Slide 3 — Dataset and evaluation population

### Main content

- UNSW-NB15: 175,341 training and 82,332 published test rows.
- Documented mirror; source filenames were reversed.
- Development and primary test deduplicated; overlaps excluded.

### Recommended visual

reports/dataset_summary.csv

### Speaker notes

Explain the provenance limitation and show split_manifest.json. Do not equate cleaned test results with an untouched published split. Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Slide 4 — Training-data EDA

### Main content

- Compare class balance and attack categories.
- Examine traffic volume, timing and protocol associations.
- Retain plausible extremes as potential signals.

### Recommended visual

figures/feature_distributions.png

### Speaker notes

All exploratory plots use development training data. Log transforms in plots aid reading; numeric model inputs are standardized without a log transform.

## Slide 5 — Preprocessing and feature engineering

### Main content

- Exclude id, label and attack_cat.
- Train-fitted imputation, scaling and rare-category one-hot encoding.
- Add volume, packet-size, directionality and rate features.

### Recommended visual

Six formulas from reports/data_dictionary.csv

### Speaker notes

Undefined ratios become missing and receive a training-fitted median. Explain the original-feature validation ablation and its limited scope.

## Slide 6 — Modeling approach

### Main content

- Logistic Regression · Random Forest · XGBoost
- Three-fold stratified CV; four sampled configurations per family.
- Tune on 30,000 training rows; refit on full development training.

### Recommended visual

reports/validation_comparison.csv

### Speaker notes

Average precision is the selection score. Validation selects the family and operating threshold; test predictions occur after those decisions are frozen.

## Slide 7 — Model comparison

### Main content

- Selected: XGBoost, threshold 0.49.
- Test recall 97.20%; precision 68.07%.
- F1 0.8007; ROC-AUC 0.9698; AP 0.9550.

### Recommended visual

figures/model_comparison.png

### Speaker notes

Show all model rows from model_comparison.csv. Training time includes tuning and refit. PR-AUC is implemented as average precision. Source: executed project outputs.

## Slide 8 — Explainability

### Main content

- Tree SHAP explains global and local predictions.
- Leading absolute attributions: numeric__sttl, numeric__ct_state_ttl, numeric__ct_srv_dst, numeric__dbytes, numeric__ct_dst_src_ltm.
- Inspect collection-environment proxies.

### Recommended visual

figures/shap_beeswarm.png

### Speaker notes

Positive SHAP pushes toward attack; negative pushes toward benign. Inspect TP, TN, FP and FN waterfalls. Attributions are not causal explanations.

## Slide 9 — Error and operational bias audit

### Main content

- 532 missed attack flows; 8,655 false alerts.
- Lowest attack-category recall: Fuzzers.
- Audit protocol, service, state and attack category.

### Recommended visual

reports/subgroup_audit.csv

### Speaker notes

Show support and Wilson recall intervals. FPR is undefined for attack-only groups. No demographic attributes are available and no demographic fairness claim is made.

## Slide 10 — Prototype deployment

### Main content

- Feature CSV → saved preprocessing/model → frozen threshold → analyst review
- Streamlit accepts examples and uploaded flow features.
- No live packet extraction or automatic blocking.

### Recommended visual

README architecture diagram

### Speaker notes

Model scores are uncalibrated. Match flow definitions and aggregation windows before any local shadow-mode trial.

## Slide 11 — Limitations

### Main content

- Historical controlled collection and possible topology shortcuts.
- No temporal or cross-network holdout; limited rare-attack support.
- Bounded search and near-duplicate/session dependence.

### Recommended visual

Limitations with corresponding validation actions

### Speaker notes

Changing the model after test inspection requires an independent new holdout. Source identity is verified internally, not against an official checksum.

## Slide 12 — Conclusions and next evaluation

### Main content

- Validation selected XGBoost.
- Use the prototype for education and analyst-reviewed experiments.
- Validate on recent local telemetry before any operational recommendation.

### Recommended visual

figures/threshold_tradeoff.png

### Speaker notes

Summarize achieved and unmet criteria from Model_Evaluation_Report.md. Generative AI assisted implementation and drafting; executed code produced metrics. Human research review remains required.
