# Dataset documentation

## Source and provenance
Use the [UNSW-NB15 dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset), obtained through the public mirror because the university download redirected to sign-in. The original project describes traffic collected in a controlled cyber range, with benign activity and generated attack behavior. Its public partitions contain 175,341 training and 82,332 testing records.

**Mirror naming discrepancy:** the mirror file named `UNSW_NB15_testing-set.csv` contains the 175,341-row partition, while its `training` file contains 82,332 rows. The downloader maps by these published row counts and records source filenames, URLs and SHA-256 checksums. This verifies internal consistency, not byte-level authenticity against the inaccessible official download. Retain this limitation in comparisons with published work.

## Observed summary

| partition          |   rows |   columns |   missing_cells |   exact_duplicates |   predictor_duplicates |   benign |   attack |
|:-------------------|-------:|----------:|----------------:|-------------------:|-----------------------:|---------:|---------:|
| Published training | 175341 |        45 |               0 |                  0 |                  74301 |    56000 |   119341 |
| Published test     |  82332 |        45 |               0 |                  0 |                  28386 |    37000 |    45332 |
| Development train  |  80648 |        45 |               0 |                  0 |                      0 |    41328 |    39320 |

The partition CSV schema has 45 columns: 42 original predictors, an identifier, the binary label, and attack category. Predictor categories are protocol, service and state; the remaining original predictors are numerical. These partition features differ from the full raw-flow dictionary. Aliases are mapped in `src/reporting.py` and the delivered dictionary includes six engineered features.

## Quality controls
See `missing_values.csv`, `feature_types.csv`, `numeric_summary.csv`, `negative_values.csv`, and `attack_distribution.csv` for computed per-column evidence. A literal service `-` is retained as a legitimate unspecified service, not declared missing. Infinite numeric values become missing. No benign/attack label is imputed. Category labels are stripped of surrounding whitespace. Zero packet/duration denominators produce missing engineered ratios rather than infinity.

## Duplicate policy and split sizes

```json
{
  "raw_train": 175341,
  "raw_test": 82332,
  "clean_development": 100811,
  "training": 80648,
  "validation": 20163,
  "primary_test": 52738,
  "removed_development": 74530,
  "conflicting_development_signatures": 229,
  "removed_test_duplicates_or_conflicts": 28392,
  "conflicting_test_signatures": 6,
  "test_overlap_removed": 1202,
  "seed": 42
}
```

Predictor-identical development rows are deduplicated irrespective of ID. Ambiguous signatures with conflicting binary labels are removed and counted. Development data are split 80/20 with stratification and seed 42. All preprocessing and CV use only the training side. For the primary test analysis, duplicates/conflicting signatures are removed within the published test and signatures seen in development are excluded. This changes the evaluation population; `published_test_secondary.json` separately retains the entire published test population for transparency. No test outcomes determine the model or threshold. Exact hashes do not catch near-duplicates or shared capture sessions, and no trustworthy session/time key is available in these partition CSVs.

## Data dictionary and terms
`data_dictionary.csv` records feature name, observed type, feature category, description, modeling role, preprocessing and possible security meaning. It uses the mirrored source dictionary with explicit aliases. Do not assume exact equivalence between these fields and arbitrary NetFlow exports.
The dataset remains under its authors' terms: academic research is permitted; commercial use requires agreement with the authors. The repository code license does not relicense the dataset.
