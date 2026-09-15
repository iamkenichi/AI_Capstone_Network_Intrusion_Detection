# Machine Learning-Based Network Intrusion Detection and Anomaly Classification

## 1. Executive summary
The validation-selected model is **XGBoost**, with operating threshold **0.49**.
On the primary test set of **52,738 flows**, attack recall is **97.20%**, precision **68.07%**, F1 **0.8007**, ROC-AUC **0.9698**, and average precision **0.9550**.
The model misses **532 attacks** (FNR **2.80%**) and flags **8,655 benign flows** (FPR **25.64%**).
These are historical benchmark results, not measured production effectiveness.

## 2. Introduction
This study tests supervised binary classification of recorded network flows. It delivers a reproducible comparison and an analyst-facing research demo.

## 3. Problem statement

Security Operations Centers (SOCs) face more network telemetry than analysts can inspect manually. A supervised model can prioritize flows for investigation, but missed attacks expose systems to harm and false alerts consume analyst time. The proposed system provides decision support to analysts who can combine model output with endpoint, identity and asset context.



## 4. Business and security context
Prioritizing flow review can help a SOC manage telemetry volume. False negatives create detection gaps; false positives create workload. Neither financial benefits nor production effectiveness are measured here.

## 5. Research question
Can supervised models distinguish attacks from benign flows while meeting transparent detection and alert-rate objectives? F1 >= 0.90 and ROC-AUC >= 0.95 are targets, not promised outcomes.

## 6. Dataset
UNSW-NB15 public partitions; downloaded from a documented mirror with filename normalization. Source authentication is limited by inaccessible official files. See `dataset_documentation.md` and `data_provenance.json`.

## 7. Data understanding
| partition          |   rows |   columns |   missing_cells |   exact_duplicates |   predictor_duplicates |   benign |   attack |
|:-------------------|-------:|----------:|----------------:|-------------------:|-----------------------:|---------:|---------:|
| Published training | 175341 |        45 |               0 |                  0 |                  74301 |    56000 |   119341 |
| Published test     |  82332 |        45 |               0 |                  0 |                  28386 |    37000 |    45332 |
| Development train  |  80648 |        45 |               0 |                  0 |                      0 |    41328 |    39320 |

## 8. Data preprocessing
Deduplicate development predictor signatures, remove conflicting binary labels, stratify development into training and validation, isolate a cleaned test population, and fit transformations only inside training. Exclude label, attack category and ID. Median/mode imputation, rare-category one-hot encoding and numeric scaling live in persisted pipelines.

## 9. Exploratory data analysis
Training figures examine balance, categories, byte/packet volume, timing, TTL, TCP properties and correlations. IQR fences flag extremes but do not delete them. See `EDA_Feature_Engineering_Report.md` for interpretation and tables.

## 10. Feature engineering
Add six transparent volume, directionality and rate features with safe division. A conditional validation ablation is recorded in `feature_ablation.csv`.

## 11. Modeling methodology

Three model families are required: Logistic Regression as an interpretable linear baseline; Random Forest for nonlinear interactions; and XGBoost for boosted decision trees. Each is wrapped in the same sklearn-compatible feature and preprocessing pipeline. Standard scaling is necessary for the linear model and harmless for tree splits. No target/attack category or row ID is supplied to any estimator.

RandomizedSearchCV uses 4 candidate configurations per family, three stratified folds, seed 42 and average precision scoring on a fixed stratified subset of 30,000 training rows. This bounded search is computationally practical but not exhaustive. Winning hyperparameters are refit on all 80,648 development training rows; validation remains untouched by fitting. Weighting options are compared without SMOTE. Search results include train/CV scores and durations in `tuning_*.csv`. Training time includes tuning and full refit; inference time is a single batch measurement including preprocessing, hardware-dependent and not a production latency benchmark.

Model selection ranks validation average precision (area under the precision-recall step function). In this repository the column `pr_auc` means average precision, not trapezoidal integration. Thresholds from 0.01 through 0.99 are evaluated on validation only. Among thresholds meeting recall >= 0.95 and FPR <= 0.05, choose the highest F1, then lowest FPR, then highest threshold. If infeasible, fall back to highest validation F1 and disclose the failure. Constraint met for the selected model: **False**. The frozen threshold is **0.49**. The model is not refit on validation, so its threshold applies to the exact saved estimator. The shared validation set is used for two selections, which can make validation performance optimistic; the final test provides a separate check.


## 12. Model results
| model               |   accuracy |   precision |   recall |     f1 |   roc_auc |   pr_auc |    fpr |    fnr |
|:--------------------|-----------:|------------:|---------:|-------:|----------:|---------:|-------:|-------:|
| Logistic Regression |     0.7358 |      0.5769 |   0.9975 | 0.7311 |    0.9089 |   0.8630 | 0.4113 | 0.0025 |
| Random Forest       |     0.8354 |      0.6938 |   0.9713 | 0.8094 |    0.9678 |   0.9506 | 0.2411 | 0.0287 |
| XGBoost             |     0.8258 |      0.6807 |   0.9720 | 0.8007 |    0.9698 |   0.9550 | 0.2564 | 0.0280 |

## 13. Model comparison
Validation selects XGBoost; final-test results characterize all three families without reopening model selection. Common 0.50 threshold results and timings are provided separately. See `Model_Evaluation_Report.md`.

## 14. Explainability
SHAP explains XGBoost on 160 sampled flows and four available outcome cases. Leading absolute attributions are numeric__sttl, numeric__ct_state_ttl, numeric__ct_srv_dst, numeric__dbytes, numeric__ct_dst_src_ltm. Signed explanations are in `Explainability_Report.md`; these are model associations, not causal conclusions.

## 15. Error analysis
The selected model misses 532 attacks and flags 8,655 benign flows. Lowest attack-category recall is in Fuzzers; inspect counts and uncertainty before interpreting rare groups.

## 16. Bias and fairness audit
Operational groups are audited with counts, rates and Wilson recall intervals. No demographic fairness claim is possible. See `Bias_Fairness_Analysis.md`.

## 17. Ethical AI considerations
Respect dataset terms, protect network telemetry, retain human review, disclose AI assistance and avoid automatic enforcement from uncalibrated model scores.

## 18. Limitations
Historical controlled collection; possible environment shortcuts; limited rare-attack support; no temporal or cross-network validation; near-duplicate/session dependence; bounded hyperparameter search; uncalibrated probabilities; uncertain equivalence to live flow extractors. Exact decontamination changes the test population. Binary classification does not demonstrate unknown-attack discovery or attack-category classification.

## 19. Deployment considerations
The Streamlit demo loads a trusted local artifact and accepts precomputed feature CSVs. Matching feature definitions, causal aggregation, secure serving, resource limits and operational monitoring are prerequisites for any further deployment. It is not a standalone production IDS.

## 20. Recommendations
Proceed with an offline/shadow-mode evaluation on recent local telemetry. Measure per-service alert workload and missed incidents, test score calibration, and approve thresholds with SOC stakeholders. Use an independent new holdout for subsequent model changes.

## 21. Conclusion
The experiment produces reproducible benchmark evidence for a supervised detection prototype. Performance targets and operational constraints are evaluated honestly; laboratory results do not justify autonomous production blocking.

## 22. References
1. [UNSW-NB15 official dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset). Dataset access and academic-use terms; accessed 15 September 2026.
2. Moustafa, N. and Slay, J. (2015). [UNSW-NB15: a comprehensive data set for network intrusion detection systems](https://ieeexplore.ieee.org/document/7348942). MilCIS. DOI: 10.1109/MilCIS.2015.7348942.
3. Moustafa, N. and Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set. Information Security Journal. See the official dataset page for the publisher link.
4. Moustafa et al. (2017). Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks. IEEE Transactions on Big Data. Linked from the official dataset page.
5. Moustafa et al. (2017). Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models. Data Analytics and Decision Support for Cybersecurity. Linked from the official dataset page.
6. Sarhan, Layeghy, Moustafa and Portmann. NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems. [Author paper](https://arxiv.org/abs/2011.09144).
7. [Public download mirror](https://github.com/jamshaid120/UNSW_NB15-Complete-dataset). Exact source URLs, local names and SHA-256 hashes are recorded in `data_provenance.json`.
8. [Scikit-learn documentation](https://scikit-learn.org/stable/), [XGBoost documentation](https://xgboost.readthedocs.io/), [SHAP documentation](https://shap.readthedocs.io/). Software references; executed versions are in `models/metadata.json`.
