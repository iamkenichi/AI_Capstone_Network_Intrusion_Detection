# Project completion report

## Local deliverables

Created the requested repository structure, pinned environment and dependency lock, dataset downloader and provenance manifest, reusable ML modules, six executed notebooks, three fitted model pipelines plus the final model, 24 figures, data dictionary, EDA/evaluation/explainability/operational-bias reports, final report, rubric audit, technical and executive presentation content, Streamlit demo, tests and a GitHub Actions test workflow.

Presentation deliverables contain 12 technical slides and 10 executive slides in Markdown, each with content, recommended visual and speaker notes. No PPTX was requested or generated.

## Data and modeling

- Dataset: UNSW-NB15, downloaded from a public mirror with recorded SHA-256 hashes. The mirror's training/testing filenames were reversed and normalized to the published 175,341/82,332-row convention.
- Modeling population: 80,648 training, 20,163 validation and 52,738 primary-test flows after documented duplicate/conflict/overlap handling.
- Trained and tuned: Logistic Regression, Random Forest and XGBoost. Each used four sampled hyperparameter configurations and three stratified CV folds on 30,000 training rows, followed by refitting on the entire training population.
- Selected model: **XGBoost**, chosen by validation average precision; validation-selected threshold **0.49**. Selection was fixed before final-test predictions.

## Actual primary-test results

| Metric | XGBoost |
| --- | ---: |
| Accuracy | 82.58% |
| Precision | 68.07% |
| Attack recall | 97.20% |
| F1 | 0.8007 |
| ROC-AUC | 0.9698 |
| Average precision (PR-AUC column) | 0.9550 |
| False-positive rate | 25.64% |
| False-negative rate | 2.80% |
| False-positive flows | 8,655 |
| False-negative flows | 532 |

Full precision metrics for all three models, including timing, are in `model_comparison.csv`. The original published-test secondary analysis is in `published_test_secondary.json` and is not conflated with the cleaned primary population.

## Important findings

The F1 >= 0.90 objective was not achieved. The ROC-AUC >= 0.95 objective was achieved. No tested validation threshold for the selected model met both recall >= 95% and FPR <= 5%, so the documented F1 fallback was used. False-alert burden remains too high for the proposed operating target.

Fuzzers account for 452 of the 532 missed attacks; their primary-test recall is 89.57%. SHAP shows strong dependence on source TTL, including in a false-positive case, motivating investigation of collection-environment shortcuts. The feature-engineering ablation produced only a small validation change and is not evidence of a major improvement.

## Verification

- Complete `python -m src.train` run finished with exit code 0.
- Eight pytest tests passed, including actual-model Streamlit example prediction. A later targeted demo test also passed after removing the third-party chart deprecation warning.
- Six notebooks and all 28 code cells executed successfully in the final refresh.
- Independent verification passed for all three input checksums, 42-feature predictor schema, 51-row dictionary, split isolation, saved metric recalculation, 24 PNGs, presentation content coverage and README local links.
- See `artifact_verification.json`, `notebook_execution.json` and `Verification.md` for evidence and environment recovery details.

## Limitations and incomplete work

This is a binary classification benchmark on historical controlled traffic, not a demonstration of unknown-attack detection or production readiness. Probabilities are uncalibrated. There is no temporal or cross-network evaluation, and exact duplicate handling does not eliminate near-duplicate/session dependence. Official source checksum equivalence could not be verified. No demographic fairness claim is made. Independent human review is still required.

GitHub repository: [iamkenichi/AI_Capstone_Network_Intrusion_Detection](https://github.com/iamkenichi/AI_Capstone_Network_Intrusion_Detection). The project extends the original main commit `40b4c0f3cdf39944369ff09f1359222a9909faac`. The earlier approval-service usage block was resolved, and authenticated repository access was verified. Raw CSVs, processed splits, model binaries and the virtual environment remain local and can be regenerated with the documented commands. The repository commit history records publication revisions.

## Reproduction commands

From the repository root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe scripts/execute_notebooks.py --in-process
.\.venv\Scripts\python.exe -m pytest --junitxml=reports/pytest_results.xml
.\.venv\Scripts\python.exe scripts/verify_project.py
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

Use a normal local shell with network access for installation/download and suitable process permissions for training/tests. On Linux/macOS substitute `.venv/bin/python`. Omitting `--in-process` uses a temporary Jupyter kernel instead.
