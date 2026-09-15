# Executive presentation content

10 slides. Metrics come from the executed run; source files are named below.

## Slide 1 — Network detection research

### Main content

- A decision-support prototype for security analysts

### Recommended visual

Minimal title and project name

### Speaker notes

This capstone evaluates recorded network flows; it is not a production protection claim.

## Slide 2 — The security problem

### Main content

- Analysts must find suspicious activity in large telemetry volumes.
- Missed attacks create risk.
- False alerts consume investigation capacity.

### Recommended visual

False alerts and missed attacks comparison

### Speaker notes

Avoid assigning unsupported dollar losses or current threat prevalence.

## Slide 3 — Monitoring workload

### Main content

- Network services and traffic patterns differ.
- Rules and model alerts need asset and incident context.
- A single score cannot replace investigation.

### Recommended visual

Example flow-to-incident review process

### Speaker notes

Explain that several flow alerts can belong to one incident and require aggregation.

## Slide 4 — Proposed ML solution

### Main content

- Use historical labeled flows to learn detection patterns.
- Compare three model approaches.
- Present suspicious flows for human review.

### Recommended visual

README architecture diagram

### Speaker notes

UNSW-NB15 is a historical research benchmark. Source: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## Slide 5 — Measured detection performance

### Main content

- Attack recall: 97.20%.
- Alert precision: 68.07%.
- Missed attack flows: 532.

### Recommended visual

figures/confusion_xgboost.png

### Speaker notes

Results describe the cleaned benchmark test population. They do not forecast detection in the organization. Source: reports/model_comparison.csv.

## Slide 6 — Potential security value

### Main content

- Prioritize flow review and make decisions inspectable.
- Measure analyst effort and incident coverage in a pilot.
- Financial ROI has not been measured.

### Recommended visual

Pilot measurement table: alerts, review time, incidents, misses

### Speaker notes

A defensible ROI requires local traffic, review time, staffing cost and incident outcomes. No savings estimate is invented.

## Slide 7 — False-alert trade-off

### Main content

- At threshold 0.49: FPR 25.64%; FNR 2.80%.
- About 2,564 alerts per 10,000 benign flows if the test FPR transfers.
- Threshold changes affect both workload and coverage.

### Recommended visual

figures/threshold_tradeoff.png

### Speaker notes

The 10,000-flow illustration is conditional arithmetic, not a volume forecast. Threshold curves use validation data; observed test rates may differ.

## Slide 8 — Explainability and governance

### Main content

- Inspect reasons for predictions and systematic errors.
- Audit services and attack categories.
- Keep analysts accountable for response decisions.

### Recommended visual

figures/shap_importance.png

### Speaker notes

Operational auditing is not demographic fairness. Protect uploaded metadata and keep the prototype local.

## Slide 9 — Deployment recommendation

### Main content

- Start with offline or shadow-mode evaluation.
- Use recent local telemetry and a defined alert budget.
- Require calibration, monitoring and rollback before expansion.

### Recommended visual

Pilot gates: validate → measure → review → decide

### Speaker notes

This model is an educational/research prototype and should not be used as a standalone production intrusion-detection system.

## Slide 10 — Key takeaways

### Main content

- XGBoost was selected using validation evidence.
- Benchmark performance has visible errors and coverage gaps.
- The next decision is whether a controlled local pilot is justified.

### Recommended visual

Three concise takeaways with limitations footnote

### Speaker notes

Do not interpret strong aggregate results as complete protection. Review rare-attack failures and local false-alert capacity before setting an operating threshold.
