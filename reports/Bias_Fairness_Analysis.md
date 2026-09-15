# Operational performance and bias audit

## Scope: operational groups, not demographic fairness
This dataset supplies no protected demographic attributes. No race, gender, age or socioeconomic variables are invented or inferred. The analysis audits performance across observed protocol, service, state and attack category. It supports **no demographic fairness conclusion**.

## Attack-category coverage

| group          |    n |   recall |    fnr |   fn |   recall_ci_low |   recall_ci_high |
|:---------------|-----:|---------:|-------:|-----:|----------------:|-----------------:|
| Fuzzers        | 4335 |   0.8957 | 0.1043 |  452 |          0.8863 |           0.9045 |
| Analysis       |  294 |   0.9422 | 0.0578 |   17 |          0.9094 |           0.9636 |
| Shellcode      |  365 |   0.9918 | 0.0082 |    3 |          0.9761 |           0.9972 |
| Exploits       | 7085 |   0.9927 | 0.0073 |   52 |          0.9904 |           0.9944 |
| DoS            | 1186 |   0.9949 | 0.0051 |    6 |          0.9890 |           0.9977 |
| Reconnaissance | 2224 |   0.9996 | 0.0004 |    1 |          0.9975 |           0.9999 |
| Generic        | 3381 |   0.9997 | 0.0003 |    1 |          0.9983 |           0.9999 |
| Worms          |   43 |   1.0000 | 0.0000 |    0 |          0.9180 |           1.0000 |
| Backdoor       |   70 |   1.0000 | 0.0000 |    0 |          0.9480 |           1.0000 |

Recall intervals use the Wilson 95% formula and assume independent flows. Shared capture conditions may violate independence. Attack-only groups have no benign denominator, so FPR is undefined (NaN), and benign-only groups have undefined recall/FNR. Precision is also undefined if the model produces no positive predictions. These are deliberately not replaced with zero. Attack-only precision can be trivially 1 and is not a useful service-alert measure.

## Services generating false alerts

| group    |     n |   precision |   recall |      fpr |    fnr |   fp |
|:---------|------:|------------:|---------:|---------:|-------:|-----:|
| -        | 34118 |      0.5558 |   0.9486 |   0.2980 | 0.0514 | 7298 |
| http     |  7891 |      0.8010 |   0.9957 |   0.2747 | 0.0043 | 1027 |
| ftp      |  1291 |      0.6135 |   0.9647 |   0.4343 | 0.0353 |  327 |
| radius   |     6 |      0.6667 |   1.0000 |   1.0000 | 0.0000 |    2 |
| dns      |  5848 |      0.9997 |   1.0000 |   0.0003 | 0.0000 |    1 |
| ftp-data |  1177 |      1.0000 |   1.0000 |   0.0000 | 0.0000 |    0 |
| dhcp     |    13 |      1.0000 |   1.0000 | nan      | 0.0000 |    0 |
| irc      |     5 |      1.0000 |   1.0000 | nan      | 0.0000 |    0 |
| pop3     |   381 |      1.0000 |   1.0000 | nan      | 0.0000 |    0 |
| smtp     |  1759 |      1.0000 |   1.0000 |   0.0000 | 0.0000 |    0 |

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
