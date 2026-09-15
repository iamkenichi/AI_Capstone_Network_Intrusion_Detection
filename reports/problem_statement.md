# Problem statement

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
