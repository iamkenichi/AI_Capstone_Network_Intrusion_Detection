# Final rubric audit

This table maps evidence to the rubric; it does not award a grade. Maximum points total **100**, including the five-point creativity bonus. Results are measured, including unmet performance targets.

| Criterion | Max Points | Evidence | File/Section | Status |
| --- | ---: | --- | --- | --- |
| Problem Understanding | 10 | Binary research question, measurable targets, SOC costs, lifecycle, risks and constraints | `problem_statement.md`; final report sections 3-5; notebook 01 | Addressed |
| Data Collection | 10 | Public UNSW-NB15, exact URLs and SHA-256 hashes, naming correction, observed row/column/type/missing/duplicate/class tables, 51-row original-plus-engineered dictionary | `dataset_documentation.md`; `data_provenance.json`; `dataset_summary.csv`; `data_dictionary.csv`; notebook 01 | Addressed; official checksum equivalence unavailable |
| EDA/Feature Engineering | 10 | Training-only class/protocol/service/state/volume/timing/TTL/TCP plots; correlations; outlier analysis; safe ratios; fold-fitted preprocessing; validation ablation; SHAP/mutual-information feature selection supplement; PCA 95% variance dimensionality-reduction supplement | `EDA_Feature_Engineering_Report.md`; `Feature_Selection_PCA_Analysis.md`; `../scripts/feature_selection_pca_analysis.py`; `../figures/`; `../src/features.py`; `../src/preprocessing.py`; notebooks 02-03 | Addressed; PCA/feature-selection supplement is separated from frozen final-test selection to avoid leakage |
| Model Implementation | 20 | Logistic Regression, Random Forest and XGBoost; three-fold randomized search; training-only refit; validation selection; frozen thresholds; complete test metrics and timings; serialized pipelines | `model_comparison.csv`; `validation_comparison.csv`; `tuning_*.csv`; `../models/`; notebooks 04-05 | Addressed; F1 and provisional alert-rate goals not met |
| Critical Thinking/Ethical AI | 20 | Real SHAP global/beeswarm/importance and TP/TN/FP/FN waterfalls; protocol/service/state/category auditing with uncertainty and undefined denominators; leakage/drift/representation/overfitting limitations; human review | `Explainability_Report.md`; `Bias_Fairness_Analysis.md`; `subgroup_audit.csv`; notebook 06 | Addressed; operational audit only, no demographic fairness claim |
| Presentation | 10 | Technical 12-slide and executive 10-slide presentation content with recommended visuals and speaker notes; exported PPTX files included in the LMS submission package | `../presentations/technical_presentation_content.md`; `../presentations/executive_presentation_content.md`; README submission-deliverables section | Addressed; exported decks delivered with assignment submission |
| GitHub | 15 | Professional README, structure, pinned environment, code-scoped MIT license, ignored data/models, CI test workflow, AI disclosure with code examples, demo transcript, and PCA/feature-selection supplement | `../README.md`; `../requirements-lock.txt`; `../.gitignore`; `../.github/workflows/tests.yml`; `Generative_AI_Usage.md`; `Feature_Selection_PCA_Analysis.md`; `../scripts/feature_selection_pca_analysis.py`; `../demo/` | Addressed in live repository |
| Creativity | 5 | Streamlit example/upload prediction demo, narrated capstone demo delivered with LMS submission, transcript in GitHub, risk interpretation, architecture diagram, polished matplotlib plots, and transparent AI disclosure with verified code examples | `../app/streamlit_app.py`; `../demo/Demo_Video_Transcript.txt`; `../demo/README.md`; README workflow; `Generative_AI_Usage.md` | Addressed; research/demo use only |
| **Total possible** | **100** | Evidence supports assessment; no guaranteed score | Entire repository and LMS artifact package | Instructor evaluation required |

## Scientific limitations that remain

- Historical controlled data and possible environment shortcuts; no cross-network or temporal validation.
- Exact decontamination changes the test population; the untouched published-test evaluation is separately disclosed.
- Four sampled configurations per model family provide bounded tuning, not an exhaustive search.
- Validation uses the same holdout for family selection and threshold tuning, so validation estimates may be optimistic.
- Scores are uncalibrated; high false-alert rates preclude recommending standalone production use.
- The PCA/feature-selection supplement is methodological evidence for Step 3 and does not replace the validation-selected XGBoost detector.
- Repository: [iamkenichi/AI_Capstone_Network_Intrusion_Detection](https://github.com/iamkenichi/AI_Capstone_Network_Intrusion_Detection). Raw data and binary models are regenerated locally rather than tracked in Git.

See `Verification.md` and `artifact_verification.json` for executed checks and environmental issues resolved during construction.
