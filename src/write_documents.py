"""Build narrative artifacts from computed evidence; never synthesize metrics."""
import json
import pandas as pd
from src.data_loader import ROOT

R=ROOT/'reports'
SOURCE='https://research.unsw.edu.au/projects/unsw-nb15-dataset'
REFERENCES=f'''1. [UNSW-NB15 official dataset page]({SOURCE}). Dataset access and academic-use terms; accessed 15 September 2026.
2. Moustafa, N. and Slay, J. (2015). [UNSW-NB15: a comprehensive data set for network intrusion detection systems](https://ieeexplore.ieee.org/document/7348942). MilCIS. DOI: 10.1109/MilCIS.2015.7348942.
3. Moustafa, N. and Slay, J. (2016). The evaluation of Network Anomaly Detection Systems: Statistical analysis of the UNSW-NB15 data set and the comparison with the KDD99 data set. Information Security Journal. See the official dataset page for the publisher link.
4. Moustafa et al. (2017). Novel geometric area analysis technique for anomaly detection using trapezoidal area estimation on large-scale networks. IEEE Transactions on Big Data. Linked from the official dataset page.
5. Moustafa et al. (2017). Big data analytics for intrusion detection system: statistical decision-making using finite dirichlet mixture models. Data Analytics and Decision Support for Cybersecurity. Linked from the official dataset page.
6. Sarhan, Layeghy, Moustafa and Portmann. NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems. [Author paper](https://arxiv.org/abs/2011.09144).
7. [Public download mirror](https://github.com/jamshaid120/UNSW_NB15-Complete-dataset). Exact source URLs, local names and SHA-256 hashes are recorded in `data_provenance.json`.
8. [Scikit-learn documentation](https://scikit-learn.org/stable/), [XGBoost documentation](https://xgboost.readthedocs.io/), [SHAP documentation](https://shap.readthedocs.io/). Software references; executed versions are in `models/metadata.json`.
'''


def write(path,text):
    (ROOT/path).write_text(text.strip()+'\n',encoding='utf-8')


def md(frame):
    return frame.to_markdown(index=False,floatfmt='.4f')


def write_all(stats,meta,shap):
    comparison=pd.read_csv(R/'model_comparison.csv')
    validation=pd.read_csv(R/'validation_comparison.csv')
    splits=json.loads((R/'split_manifest.json').read_text())
    audit=pd.read_csv(R/'subgroup_audit.csv')
    best=comparison[comparison.model==meta['model']].iloc[0]
    attacks=audit[(audit.group_type=='attack_cat') & (audit.group!='Normal')].sort_values(['recall','n'])
    false_alerts=audit[audit.group_type=='service'].sort_values('fp',ascending=False)
    importance=pd.read_csv(R/'shap_importance.csv')
    ablation=pd.read_csv(R/'feature_ablation.csv')
    top=', '.join(importance.feature.head(5))
    summary=f'''The validation-selected model is **{meta['model']}**, with operating threshold **{meta['threshold']:.2f}**.
On the primary test set of **{int(best.n):,} flows**, attack recall is **{best.recall:.2%}**, precision **{best.precision:.2%}**, F1 **{best.f1:.4f}**, ROC-AUC **{best.roc_auc:.4f}**, and average precision **{best.pr_auc:.4f}**.
The model misses **{int(best.fn):,} attacks** (FNR **{best.fnr:.2%}**) and flags **{int(best.fp):,} benign flows** (FPR **{best.fpr:.2%}**).
These are historical benchmark results, not measured production effectiveness.'''
    problem='''# Problem statement

## Security problem and business value
Security Operations Centers (SOCs) face more network telemetry than analysts can inspect manually. A supervised model can prioritize flows for investigation, but missed attacks expose systems to harm and false alerts consume analyst time. The proposed system provides decision support to analysts who can combine model output with endpoint, identity and asset context.

## Research question and task
Can supervised machine-learning models distinguish malicious from benign network flows while maintaining an operationally acceptable balance of missed attacks and false alerts? This is binary classification: `label=0` denotes benign and `label=1` denotes attack. It is not an experiment demonstrating detection of genuinely novel attacks, nor does binary classification identify an attack family. `attack_cat` remains available only for explanation and subgroup analysis.

## Measurable success criteria
Research targets are F1 >= 0.90 and ROC-AUC >= 0.95, where achievable. Report attack recall, precision, average precision, FPR and FNR regardless of target attainment. A provisional validation operating constraint asks for recall >= 0.95 and FPR <= 0.05; maximize F1 among feasible thresholds. If no threshold is feasible, explicitly report failure and use maximum validation F1 as an exploratory operating point. These research constraints need stakeholder approval and local workload validation before operational use.

## Lifecycle and constraints
Frame the decision; acquire and validate data; isolate evaluation data; explore training data; fit preprocessing inside cross-validation; tune and compare models; freeze a validation threshold; evaluate once; explain decisions; audit errors; deploy a local prototype; propose monitoring and retraining. Accuracy alone can hide missed attacks and alert burden. Feature extraction must be available at decision time. Flow completion introduces latency; aggregate connection counters require historical context. No packet-capture or live blocking integration is included.

## Risks
False negatives can delay investigation. False positives can cause alert fatigue. Scores require calibration before probability-sensitive decisions. Explainability supports inspection but does not establish causality. Historical laboratory traffic differs from current deployment environments. Concept drift, changing service distributions and attacker adaptation can reduce performance; test on recent local telemetry and retain human review.
'''
    write('reports/problem_statement.md',problem)
    dataset=f'''# Dataset documentation

## Source and provenance
Use the [UNSW-NB15 dataset]({SOURCE}), obtained through the public mirror because the university download redirected to sign-in. The original project describes traffic collected in a controlled cyber range, with benign activity and generated attack behavior. Its public partitions contain 175,341 training and 82,332 testing records.

**Mirror naming discrepancy:** the mirror file named `UNSW_NB15_testing-set.csv` contains the 175,341-row partition, while its `training` file contains 82,332 rows. The downloader maps by these published row counts and records source filenames, URLs and SHA-256 checksums. This verifies internal consistency, not byte-level authenticity against the inaccessible official download. Retain this limitation in comparisons with published work.

## Observed summary

{md(stats)}

The partition CSV schema has 45 columns: 42 original predictors, an identifier, the binary label, and attack category. Predictor categories are protocol, service and state; the remaining original predictors are numerical. These partition features differ from the full raw-flow dictionary. Aliases are mapped in `src/reporting.py` and the delivered dictionary includes six engineered features.

## Quality controls
See `missing_values.csv`, `feature_types.csv`, `numeric_summary.csv`, `negative_values.csv`, and `attack_distribution.csv` for computed per-column evidence. A literal service `-` is retained as a legitimate unspecified service, not declared missing. Infinite numeric values become missing. No benign/attack label is imputed. Category labels are stripped of surrounding whitespace. Zero packet/duration denominators produce missing engineered ratios rather than infinity.

## Duplicate policy and split sizes

```json
{json.dumps(splits,indent=2)}
```

Predictor-identical development rows are deduplicated irrespective of ID. Ambiguous signatures with conflicting binary labels are removed and counted. Development data are split 80/20 with stratification and seed 42. All preprocessing and CV use only the training side. For the primary test analysis, duplicates/conflicting signatures are removed within the published test and signatures seen in development are excluded. This changes the evaluation population; `published_test_secondary.json` separately retains the entire published test population for transparency. No test outcomes determine the model or threshold. Exact hashes do not catch near-duplicates or shared capture sessions, and no trustworthy session/time key is available in these partition CSVs.

## Data dictionary and terms
`data_dictionary.csv` records feature name, observed type, feature category, description, modeling role, preprocessing and possible security meaning. It uses the mirrored source dictionary with explicit aliases. Do not assume exact equivalence between these fields and arbitrary NetFlow exports.
The dataset remains under its authors' terms: academic research is permitted; commercial use requires agreement with the authors. The repository code license does not relicense the dataset.
'''
    write('reports/dataset_documentation.md',dataset)
    assoc=pd.read_csv(R/'target_association.csv').head(6)
    eda=f'''# EDA and feature engineering

## Scope
All distribution, association and feature-design analysis uses only the development training partition. Published partition totals are used for data validation, not feature selection. Figures show observed data and tables are regenerated by executed code.

## Class balance and traffic categories
Training includes {splits['training']:,} flows after duplicate handling. Class and attack-category plots show representation; protocol, service and state charts display both counts and attack fractions. Small category attack rates are unstable and may reflect the collection setup rather than transportable rules. Class weighting is a tuning option; SMOTE is not used because interpolating mixed protocol and aggregate-count features is hard to justify.

## Numeric relationships
`feature_distributions.png` compares log(1+x) distributions of duration, source/destination bytes, packets, rate, TTL and TCP round-trip time. `byte_relationship.png` examines bidirectional volume; `correlations.png` measures Spearman relationships. Log axes aid visualization only: the fitted pipeline uses original numeric values followed by standard scaling. Highly correlated source/destination statistics and aggregate counts may divide attribution among related features.

Top training target associations (association is not causality):

{md(assoc)}

TTL and TCP sequence fields can encode operating-system or collection-environment characteristics. They are not label annotations, but high reliance is a transferability concern. Future evaluation should test their removal on an independent validation protocol before a new final test.

## Outliers and missing values
`outlier_analysis.csv` quantifies values outside 1.5 IQR fences. These are flags, not deletions: extreme traffic may be the attack signal. No values are removed merely for being extreme. Finite-value checks and zero-denominator handling prevent numerical failures. Numeric medians, categorical modes, scaling and rare-category encoding are fitted inside each CV training fold; the final pipeline refits on development training only. Unseen categories map to the learned infrequent bucket where available or an all-zero encoding otherwise. The source describes is_ftp_login as binary; any values outside that description must be treated as a dataset discrepancy, not silently relabeled.

## Engineered features
Total bytes and packets describe traffic volume. Bytes per packet approximates flow size structure. Source byte share and packet share measure directionality without unstable division by destination-only counts. Bytes per second relates volume to duration. All formulas and zero handling are in `data_dictionary.csv` and `src/features.py`; transformations do not inspect labels.

## Feature importance and ablation
SHAP importance is used for feature analysis; no test-driven feature selection is performed. The original-feature ablation refits the validation-selected model family with the same hyperparameters and compares at 0.50 on validation data. It is a limited conditional comparison, not a retuned competition, and does not change the frozen winner.

{md(ablation[['features','precision','recall','f1','roc_auc','pr_auc']])}
'''
    write('reports/EDA_Feature_Engineering_Report.md',eda)
    method=f'''## Modeling methodology
Three model families are required: Logistic Regression as an interpretable linear baseline; Random Forest for nonlinear interactions; and XGBoost for boosted decision trees. Each is wrapped in the same sklearn-compatible feature and preprocessing pipeline. Standard scaling is necessary for the linear model and harmless for tree splits. No target/attack category or row ID is supplied to any estimator.

RandomizedSearchCV uses {meta['iterations']} candidate configurations per family, three stratified folds, seed 42 and average precision scoring on a fixed stratified subset of {meta['search_rows']:,} training rows. This bounded search is computationally practical but not exhaustive. Winning hyperparameters are refit on all {splits['training']:,} development training rows; validation remains untouched by fitting. Weighting options are compared without SMOTE. Search results include train/CV scores and durations in `tuning_*.csv`. Training time includes tuning and full refit; inference time is a single batch measurement including preprocessing, hardware-dependent and not a production latency benchmark.

Model selection ranks validation average precision (area under the precision-recall step function). In this repository the column `pr_auc` means average precision, not trapezoidal integration. Thresholds from 0.01 through 0.99 are evaluated on validation only. Among thresholds meeting recall >= 0.95 and FPR <= 0.05, choose the highest F1, then lowest FPR, then highest threshold. If infeasible, fall back to highest validation F1 and disclose the failure. Constraint met for the selected model: **{meta['constraint_met']}**. The frozen threshold is **{meta['threshold']:.2f}**. The model is not refit on validation, so its threshold applies to the exact saved estimator. The shared validation set is used for two selections, which can make validation performance optimistic; the final test provides a separate check.
'''
    evalreport=f'''# Model evaluation

{summary}

{method}

## Validation comparison

{md(validation[['model','cv_pr_auc','pr_auc','f1','threshold','constraint_met']])}

## Final primary test comparison
All thresholds were frozen on validation. This table describes final evaluation; it is not a new model-selection step.

{md(comparison[['model','accuracy','precision','recall','f1','roc_auc','pr_auc','fpr','fnr','training_seconds','inference_seconds']])}

The model with the highest observed test recall is **{comparison.loc[comparison.recall.idxmax(),'model']}**. The lowest observed FPR belongs to **{comparison.loc[comparison.fpr.idxmin(),'model']}**. The recommendation remains the validation-selected **{meta['model']}** for an educational prototype, subject to the limitations below. `model_comparison_threshold_050.csv` supplies a common-threshold comparison so each model's threshold trade-off is visible.

## Security implications
The selected model generates {int(best.fp):,} false alerts and misses {int(best.fn):,} attack flows in this test. False alerts consume investigation time; missed attack flows represent coverage gaps, but a flow is not the same unit as an incident. Lowering thresholds generally catches more attacks while increasing false alerts. The lowest-recall attack category is **{attacks.iloc[0]['group']}**; review support and confidence intervals before generalizing from rare categories.

## Targets and recommendation
F1 target >= 0.90 achieved: **{bool(best.f1>=.90)}**. ROC-AUC target >= 0.95 achieved: **{bool(best.roc_auc>=.95)}**. Validation constraint attainment does not guarantee the same constraint on test or live data. Use an offline or shadow-mode trial with analyst review, never automatic blocking from this prototype.

## Operational workload scenario
If a future population retained this test FPR, every 10,000 benign flows would produce approximately {best.fpr*10000:,.0f} false alerts. This is a conditional arithmetic illustration, not a traffic-volume forecast. Deployment precision also depends on attack prevalence. Expected alerts for N flows and attack prevalence p are N[p*recall + (1-p)*FPR]. Analyst effort equals alerts times average review time; neither actual costs nor savings are available, so no monetary ROI is claimed.

## Limits of the estimate
The primary population excludes exact development overlaps and repeated test flows, which affects prevalence and comparability. See the secondary published-test metrics. No temporal, cross-network or current-attack holdout exists here. Scores are uncalibrated. Adjacent flows may be correlated; binomial uncertainty intervals for groups do not address that dependence.
'''
    write('reports/Model_Evaluation_Report.md',evalreport)
    local='\n\n'.join(f"### {case['case'].replace('_',' ').title()}\n\nRow {case['row_id']}, category {case['attack_cat']}, attack score {case['probability']:.4f}. Strongest contributions: "+'; '.join(f"{v['feature']} ({v['shap']:+.3f})" for v in case['strongest'])+'.' for case in shap['local'])
    explain=f'''# Explainability

SHAP explains **{shap['model']}**, the best tree family by validation average precision. The selected deployment prototype model is {meta['model']}. Global plots use {shap['random_sample_size']} deterministic random primary-test examples; local examples are the first available member of each prediction outcome. Local examples illustrate behavior and are not representative estimates.

## Global behavior
Largest mean absolute contributions: {top}. `shap_beeswarm.png` shows direction and spread, while `shap_importance.png` shows magnitude. Positive contributions push toward attack and negative contributions toward benign, measured in **{shap['units'].lower()}** for this explainer. Contributions refer to transformed inputs; numeric colors therefore reflect standardized values, and one-hot colors indicate category activation. Magnitude alone gives no direction; inspect the beeswarm or local signs.

TTL, state and context-count features can reflect capture topology, device defaults or traffic-generator behavior. Dominance of these variables is a reason for transferability testing and feature ablation, not evidence that all high-TTL flows are malicious. `id`, `label` and `attack_cat` are explicitly absent from the model schema. Correlated features can share attribution, and SHAP is neither causal proof nor a security rule.

## Local explanations
{local}

Unavailable outcome classes: {shap['missing_cases']}. No synthetic example is substituted for an unavailable error class. Waterfall plots are stored as `figures/shap_<outcome>.png`; complete signed contributions for the displayed leading features are in `shap_details.json`.
'''
    write('reports/Explainability_Report.md',explain)
    bias=f'''# Operational performance and bias audit

## Scope: operational groups, not demographic fairness
This dataset supplies no protected demographic attributes. No race, gender, age or socioeconomic variables are invented or inferred. The analysis audits performance across observed protocol, service, state and attack category. It supports **no demographic fairness conclusion**.

## Attack-category coverage

{md(attacks[['group','n','recall','fnr','fn','recall_ci_low','recall_ci_high']])}

Recall intervals use the Wilson 95% formula and assume independent flows. Shared capture conditions may violate independence. Attack-only groups have no benign denominator, so FPR is undefined (NaN), and benign-only groups have undefined recall/FNR. Precision is also undefined if the model produces no positive predictions. These are deliberately not replaced with zero. Attack-only precision can be trivially 1 and is not a useful service-alert measure.

## Services generating false alerts

{md(false_alerts[['group','n','precision','recall','fpr','fnr','fp']].head(10))}

`subgroup_audit.csv` includes all observed groups and sample counts, precision, recall, F1, FPR, FNR and confusion counts. Small samples demand caution; rank large service groups by total false alerts for workload and inspect rates for disproportionate error burden.

## Bias sources and mitigations
Collection-environment bias: controlled traffic can reward shortcuts tied to topology. Mitigation: independent networks, device diversity and feature ablation using development data.

Temporal bias and concept drift: historical attack behavior may not match current telemetry. Mitigation: time-separated validation, recent labeled data, monitoring of feature distributions and delayed-label recall, and versioned retraining.

Representation bias and imbalance: rare attack categories have little statistical support. Mitigation: collect additional authentic examples and report per-category recall with uncertainty. Weighting and threshold selection alone do not guarantee minority coverage.

Overfitting and leakage: exact duplicate removal, explicit predictor exclusions and training-only transforms reduce known pathways. Near-duplicates, shared sessions and aggregate fields with uncertain online availability remain concerns. Mitigation: group/time splits where provenance allows and a causal feature extractor.

Adversarial adaptation: actors can change behavior over time. Mitigation: combine the prototype with signature systems, endpoint telemetry and analyst review; assess defensive robustness using approved recorded datasets. This project does not implement attack generation.

False-positive fatigue: aggregate alerts by incident and asset context, set a measured workload budget, review service-specific errors and monitor calibration. Establish rollback and human approval before any response action.

## Governance and ethics
Network metadata can still be sensitive. Minimize retention and identifiers, restrict access to uploaded flows, keep the demo local, and avoid publishing operational telemetry. Document data terms, model version and decision threshold. Human researchers must validate methodology and claims, especially if the project is reused outside this academic setting.
'''
    write('reports/Bias_Fairness_Analysis.md',bias)
    final=f'''# Machine Learning-Based Network Intrusion Detection and Anomaly Classification

## 1. Executive summary
{summary}

## 2. Introduction
This study tests supervised binary classification of recorded network flows. It delivers a reproducible comparison and an analyst-facing research demo.

## 3. Problem statement
{problem.split('## Security problem and business value')[1].split('## Research question')[0]}

## 4. Business and security context
Prioritizing flow review can help a SOC manage telemetry volume. False negatives create detection gaps; false positives create workload. Neither financial benefits nor production effectiveness are measured here.

## 5. Research question
Can supervised models distinguish attacks from benign flows while meeting transparent detection and alert-rate objectives? F1 >= 0.90 and ROC-AUC >= 0.95 are targets, not promised outcomes.

## 6. Dataset
UNSW-NB15 public partitions; downloaded from a documented mirror with filename normalization. Source authentication is limited by inaccessible official files. See `dataset_documentation.md` and `data_provenance.json`.

## 7. Data understanding
{md(stats)}

## 8. Data preprocessing
Deduplicate development predictor signatures, remove conflicting binary labels, stratify development into training and validation, isolate a cleaned test population, and fit transformations only inside training. Exclude label, attack category and ID. Median/mode imputation, rare-category one-hot encoding and numeric scaling live in persisted pipelines.

## 9. Exploratory data analysis
Training figures examine balance, categories, byte/packet volume, timing, TTL, TCP properties and correlations. IQR fences flag extremes but do not delete them. See `EDA_Feature_Engineering_Report.md` for interpretation and tables.

## 10. Feature engineering
Add six transparent volume, directionality and rate features with safe division. A conditional validation ablation is recorded in `feature_ablation.csv`.

## 11. Modeling methodology
{method.replace('## Modeling methodology','')}

## 12. Model results
{md(comparison[['model','accuracy','precision','recall','f1','roc_auc','pr_auc','fpr','fnr']])}

## 13. Model comparison
Validation selects {meta['model']}; final-test results characterize all three families without reopening model selection. Common 0.50 threshold results and timings are provided separately. See `Model_Evaluation_Report.md`.

## 14. Explainability
SHAP explains {shap['model']} on {shap['random_sample_size']} sampled flows and four available outcome cases. Leading absolute attributions are {top}. Signed explanations are in `Explainability_Report.md`; these are model associations, not causal conclusions.

## 15. Error analysis
The selected model misses {int(best.fn):,} attacks and flags {int(best.fp):,} benign flows. Lowest attack-category recall is in {attacks.iloc[0]['group']}; inspect counts and uncertainty before interpreting rare groups.

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
{REFERENCES}
'''
    write('reports/Final_Project_Report.md',final)
    write('reports/References.md','# References\n\n'+REFERENCES)
    make_presentations(meta,best,summary,top,attacks)
    make_readme(meta,best,summary,comparison)
    from src.build_notebooks import build
    build()


def make_presentations(meta,best,summary,top,attacks):
    technical=[
        ('Network intrusion detection','Machine Learning-Based Network Intrusion Detection and Anomaly Classification\nUNSW-NB15 · Binary supervised classification','Project title and concise workflow','Introduce the research question and distinguish binary detection from attack-family classification.'),
        ('Problem and objectives','Prioritize suspicious flows for SOC analysts.\nResearch targets: F1 >= 0.90; ROC-AUC >= 0.95.\nReport missed attacks and false-alert burden.','False-positive versus false-negative decision table','Targets are provisional and are assessed against executed evidence. A flow alert is not an incident.'),
        ('Dataset and evaluation population','UNSW-NB15: 175,341 training and 82,332 published test rows.\nDocumented mirror; source filenames were reversed.\nDevelopment and primary test deduplicated; overlaps excluded.','reports/dataset_summary.csv','Explain the provenance limitation and show split_manifest.json. Do not equate cleaned test results with an untouched published split. Source: '+SOURCE),
        ('Training-data EDA','Compare class balance and attack categories.\nExamine traffic volume, timing and protocol associations.\nRetain plausible extremes as potential signals.','figures/feature_distributions.png','All exploratory plots use development training data. Log transforms in plots aid reading; numeric model inputs are standardized without a log transform.'),
        ('Preprocessing and feature engineering','Exclude id, label and attack_cat.\nTrain-fitted imputation, scaling and rare-category one-hot encoding.\nAdd volume, packet-size, directionality and rate features.','Six formulas from reports/data_dictionary.csv','Undefined ratios become missing and receive a training-fitted median. Explain the original-feature validation ablation and its limited scope.'),
        ('Modeling approach','Logistic Regression · Random Forest · XGBoost\nThree-fold stratified CV; four sampled configurations per family.\nTune on 30,000 training rows; refit on full development training.','reports/validation_comparison.csv','Average precision is the selection score. Validation selects the family and operating threshold; test predictions occur after those decisions are frozen.'),
        ('Model comparison',f"Selected: {meta['model']}, threshold {meta['threshold']:.2f}.\nTest recall {best.recall:.2%}; precision {best.precision:.2%}.\nF1 {best.f1:.4f}; ROC-AUC {best.roc_auc:.4f}; AP {best.pr_auc:.4f}.",'figures/model_comparison.png','Show all model rows from model_comparison.csv. Training time includes tuning and refit. PR-AUC is implemented as average precision. Source: executed project outputs.'),
        ('Explainability',f'Tree SHAP explains global and local predictions.\nLeading absolute attributions: {top}.\nInspect collection-environment proxies.','figures/shap_beeswarm.png','Positive SHAP pushes toward attack; negative pushes toward benign. Inspect TP, TN, FP and FN waterfalls. Attributions are not causal explanations.'),
        ('Error and operational bias audit',f"{int(best.fn):,} missed attack flows; {int(best.fp):,} false alerts.\nLowest attack-category recall: {attacks.iloc[0]['group']}.\nAudit protocol, service, state and attack category.",'reports/subgroup_audit.csv','Show support and Wilson recall intervals. FPR is undefined for attack-only groups. No demographic attributes are available and no demographic fairness claim is made.'),
        ('Prototype deployment','Feature CSV → saved preprocessing/model → frozen threshold → analyst review\nStreamlit accepts examples and uploaded flow features.\nNo live packet extraction or automatic blocking.','README architecture diagram','Model scores are uncalibrated. Match flow definitions and aggregation windows before any local shadow-mode trial.'),
        ('Limitations','Historical controlled collection and possible topology shortcuts.\nNo temporal or cross-network holdout; limited rare-attack support.\nBounded search and near-duplicate/session dependence.','Limitations with corresponding validation actions','Changing the model after test inspection requires an independent new holdout. Source identity is verified internally, not against an official checksum.'),
        ('Conclusions and next evaluation',f"Validation selected {meta['model']}.\nUse the prototype for education and analyst-reviewed experiments.\nValidate on recent local telemetry before any operational recommendation.",'figures/threshold_tradeoff.png','Summarize achieved and unmet criteria from Model_Evaluation_Report.md. Generative AI assisted implementation and drafting; executed code produced metrics. Human research review remains required.')]
    executive=[
        ('Network detection research','A decision-support prototype for security analysts','Minimal title and project name','This capstone evaluates recorded network flows; it is not a production protection claim.'),
        ('The security problem','Analysts must find suspicious activity in large telemetry volumes.\nMissed attacks create risk.\nFalse alerts consume investigation capacity.','False alerts and missed attacks comparison','Avoid assigning unsupported dollar losses or current threat prevalence.'),
        ('Monitoring workload','Network services and traffic patterns differ.\nRules and model alerts need asset and incident context.\nA single score cannot replace investigation.','Example flow-to-incident review process','Explain that several flow alerts can belong to one incident and require aggregation.'),
        ('Proposed ML solution','Use historical labeled flows to learn detection patterns.\nCompare three model approaches.\nPresent suspicious flows for human review.','README architecture diagram','UNSW-NB15 is a historical research benchmark. Source: '+SOURCE),
        ('Measured detection performance',f"Attack recall: {best.recall:.2%}.\nAlert precision: {best.precision:.2%}.\nMissed attack flows: {int(best.fn):,}.",'figures/confusion_'+meta['model'].lower().replace(' ','_')+'.png','Results describe the cleaned benchmark test population. They do not forecast detection in the organization. Source: reports/model_comparison.csv.'),
        ('Potential security value','Prioritize flow review and make decisions inspectable.\nMeasure analyst effort and incident coverage in a pilot.\nFinancial ROI has not been measured.','Pilot measurement table: alerts, review time, incidents, misses','A defensible ROI requires local traffic, review time, staffing cost and incident outcomes. No savings estimate is invented.'),
        ('False-alert trade-off',f"At threshold {meta['threshold']:.2f}: FPR {best.fpr:.2%}; FNR {best.fnr:.2%}.\nAbout {best.fpr*10000:,.0f} alerts per 10,000 benign flows if the test FPR transfers.\nThreshold changes affect both workload and coverage.",'figures/threshold_tradeoff.png','The 10,000-flow illustration is conditional arithmetic, not a volume forecast. Threshold curves use validation data; observed test rates may differ.'),
        ('Explainability and governance','Inspect reasons for predictions and systematic errors.\nAudit services and attack categories.\nKeep analysts accountable for response decisions.','figures/shap_importance.png','Operational auditing is not demographic fairness. Protect uploaded metadata and keep the prototype local.'),
        ('Deployment recommendation','Start with offline or shadow-mode evaluation.\nUse recent local telemetry and a defined alert budget.\nRequire calibration, monitoring and rollback before expansion.','Pilot gates: validate → measure → review → decide','This model is an educational/research prototype and should not be used as a standalone production intrusion-detection system.'),
        ('Key takeaways',f"{meta['model']} was selected using validation evidence.\nBenchmark performance has visible errors and coverage gaps.\nThe next decision is whether a controlled local pilot is justified.",'Three concise takeaways with limitations footnote','Do not interpret strong aggregate results as complete protection. Review rare-attack failures and local false-alert capacity before setting an operating threshold.')]
    for audience,slides in [('technical',technical),('executive',executive)]:
        content=f'# {audience.title()} presentation content\n\n{len(slides)} slides. Metrics come from the executed run; source files are named below.\n'
        for i,(title,body,visual,notes) in enumerate(slides,1):
            content+=f'\n## Slide {i} — {title}\n\n### Main content\n\n'+ '\n'.join('- '+line for line in body.splitlines())+f'\n\n### Recommended visual\n\n{visual}\n\n### Speaker notes\n\n{notes}\n'
        write(f'presentations/{audience}_presentation_content.md',content)


def make_readme(meta,best,summary,comparison):
    readme=fr'''# Machine Learning-Based Network Intrusion Detection and Anomaly Classification

Graduate AI/ML capstone · UNSW-NB15 · Reproducible binary detection · Analyst review

## Executive summary
{summary}

## Problem statement
Prioritize network-flow investigation while quantifying missed attacks and false-alert workload. The research asks whether supervised classifiers can distinguish benign and malicious flows. The primary target is `label`; `attack_cat` is reserved for operational subgroup auditing. This project does not demonstrate novel-attack discovery or multiclass anomaly classification.

## Dataset
[UNSW-NB15 official source]({SOURCE}). The local training/test partitions contain 175,341/82,332 rows before cleaning. The university download redirected to sign-in, so a public mirror was used. Its reversed filenames are normalized by row count; `reports/data_provenance.json` records exact URLs and SHA-256 hashes. Official checksum equivalence is not established. See [dataset documentation](reports/dataset_documentation.md), [dictionary](reports/data_dictionary.csv), and [download instructions](data/raw/README.md).

## Architecture and workflow

```mermaid
flowchart TD
    A[Network Dataset] --> B[Data Validation]
    B --> S[Duplicate handling and fixed splits]
    S --> C[EDA on training data]
    C --> D[Training-only Preprocessing]
    D --> E[Feature Engineering]
    E --> F[Model Training and CV]
    F --> G[Validation Model Comparison and Threshold]
    G --> H[Frozen Test Evaluation]
    H --> I[Explainability]
    I --> J[Bias and Error Audit]
    J --> K[Deployment Prototype]
```

## Installation
Python 3.13; tested interpreter and library versions are recorded in `models/metadata.json`. Create an isolated environment from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

On Linux/macOS use `.venv/bin/python` in place of the Windows interpreter path. `requirements.txt` pins direct dependencies; `requirements-lock.txt` records the executed full dependency set. Alternatively use `conda env create -f environment.yml`. Deterministic seeds reduce variation; hardware and parallel floating-point behavior can still vary.

## Usage and reproduction
Run from the repository root. The download helper needs internet access and preserves the same dataset; it does not substitute another dataset.

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe scripts/execute_notebooks.py
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```

`python -m src.train --iterations 4 --search-rows 30000` is the default experiment. Three-fold CV tunes on a fixed stratified training subset; refit uses all development training rows. All fitted model files and processed splits are regenerated. `python -m src.reporting` regenerates charts, SHAP, reports and notebooks from saved models without retraining. Notebook execution reads the saved experiment and performs additional checks; it does not repeat expensive searches. Training itself is executed by `src.train`.

## Model comparison
Validation selects the model by average precision and determines its operating threshold. Final-test outcomes do not change that choice. Primary test data exclude exact duplicate/conflicting signatures and development overlaps; secondary full-published-test results are separate.

{md(comparison[['model','accuracy','precision','recall','f1','roc_auc','pr_auc','fpr','fnr']])}

`pr_auc` is average precision. All rates refer to attack=1. Undefined subgroup metrics remain NaN. The comparison uses each model's validation-selected threshold; a 0.50 comparison is also supplied.

## Main findings and explainability
The selected model is {meta['model']} at threshold {meta['threshold']:.2f}. F1 >= 0.90 achieved: {bool(best.f1>=.90)}; ROC-AUC >= 0.95 achieved: {bool(best.roc_auc>=.95)}. Inspect the [evaluation report](reports/Model_Evaluation_Report.md) for error counts and threshold-constraint attainment.

![Model comparison](figures/model_comparison.png)

Tree SHAP provides global importance, a beeswarm and local explanations for correct attacks, correct benign flows, false positives and false negatives. See [explainability](reports/Explainability_Report.md). Attributions are model associations; correlated features and environment proxies limit interpretation.

## Ethical AI and operational bias
Protocol, service, state and attack category are legitimate operational groups. The dataset lacks protected demographic attributes, so no demographic fairness conclusion is made. The [audit](reports/Bias_Fairness_Analysis.md) reports sample counts, errors and recall uncertainty. Keep analysts in the loop, minimize telemetry exposure and respect dataset terms. [Generative AI usage](reports/Generative_AI_Usage.md) distinguishes AI assistance from executed experimental evidence.

## Limitations
Historical laboratory collection, possible topology shortcuts, rare-category uncertainty, uncalibrated probabilities, bounded tuning and no temporal/cross-network holdout limit deployment claims. Exact deduplication does not eliminate shared-session or near-duplicate dependence. Aggregate context features need matching causal extraction. The mirror cannot be authenticated against official checksums. Benchmark prevalence differs from real SOC traffic.

## Repository structure

```text
app/             Streamlit demo and example flow CSV
data/raw/        Downloaded partitions (ignored by Git)
data/processed/  Reproducible fixed splits (ignored by Git)
src/             Validation, features, pipelines, training, evaluation, reporting
notebooks/       Six executed research notebooks
models/          Three models, final model and metadata (binaries ignored)
figures/         EDA, evaluation and SHAP PNGs
reports/         Narrative reports, computed tables, provenance and rubric audit
presentations/   Technical (12 slides) and executive (10 slides) content with notes
scripts/         Download and notebook execution helpers
tests/           Pipeline, feature, prediction and metric tests
```

## Deployment instructions
Run Streamlit locally using the command above. Edit an example or upload a CSV containing the full 42-feature schema; label and identifiers are not needed. The app displays Benign/Attack, an uncalibrated attack score, threshold and interpretation. Use only trusted locally generated joblib artifacts. Missing model/data files receive an actionable message. This model is an educational/research prototype and should not be used as a standalone production intrusion-detection system.

A production proposal would require a matching feature extractor, privacy controls, authentication, calibration, recent independent evaluation, monitoring and rollback. No public hosting or automatic blocking has been configured.

## Reproducibility and verification
Seed 42; fixed published partition mapping; duplicate manifest; input SHA-256 hashes; pinned dependencies; saved CV configurations; persisted preprocessing/model/threshold; full test predictions and subgroup counts. See [verification](reports/Verification.md) and [rubric audit](reports/Rubric_Audit.md). Raw data and model binaries are intentionally excluded from Git because of size and data terms; regenerate them with the commands above. Project repository: [iamkenichi/AI_Capstone_Network_Intrusion_Detection](https://github.com/iamkenichi/AI_Capstone_Network_Intrusion_Detection). The original repository history is preserved.

## License and references
MIT applies only to original repository code and documentation. UNSW-NB15 remains under its authors' academic-use terms. This does not grant commercial dataset rights. See [references](reports/References.md) and the official dataset page for the requested dataset citations.
'''
    write('README.md',readme)
