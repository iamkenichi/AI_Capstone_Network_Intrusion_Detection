# Generative AI Usage Disclosure

## How Generative AI was used

OpenAI Codex was used as an implementation assistant during this capstone. It supported project scaffolding, Python code generation and refactoring, debugging, test construction, documentation drafting, and visualization suggestions. Generative AI was not used to invent model results, dataset statistics, confusion-matrix counts, threshold curves, or SHAP outputs.

The measured results in this repository were produced by executed Python code on the UNSW-NB15 dataset. The final model family and operating threshold were selected using validation evidence before inspection of the final test results. Final responsibility for methodology, interpretation, disclosure, and submission remains with the human author.

## Examples of AI-assisted implementation that were verified by execution

### Example 1 - Target-leakage protection

The project excludes the row identifier, the binary target, and the attack-category annotation from predictors. In particular, `attack_cat` is retained only for EDA and subgroup/error analysis because using it to predict `label` would leak target information.

```python
# src/data_loader.py
CATEGORICAL = ['proto', 'service', 'state']
EXCLUDED = ['id', 'label', 'attack_cat']


def predictors(frame):
    out = frame.drop(columns=EXCLUDED, errors='ignore').copy()
    for name in CATEGORICAL:
        out[name] = out[name].map(
            lambda x: str(x).strip() if pd.notna(x) else np.nan
        )
    return out.replace([np.inf, -np.inf], np.nan)
```

### Example 2 - Validation-only threshold analysis

Codex assisted with the threshold-analysis code, but the selected threshold is produced from executed validation data. The research constraint is explicitly checked rather than assumed to be feasible.

```python
# src/evaluate.py
def threshold_table(y, p):
    return pd.DataFrame([
        dict(threshold=float(t), **metrics(y, p, t))
        for t in np.linspace(0.01, 0.99, 99)
    ])


def choose_threshold(table):
    candidates = table[(table.recall >= 0.95) & (table.fpr <= 0.05)]
    feasible = not candidates.empty
    if not feasible:
        candidates = table
    best = candidates.sort_values(
        ['f1', 'fpr', 'threshold'],
        ascending=[False, True, False]
    ).iloc[0]
    return float(best.threshold), feasible
```

This process selected the XGBoost operating threshold of **0.49** from validation evidence. The stricter recall/FPR research constraint was not achievable, and that limitation is reported instead of being hidden.

### Example 3 - Executable tests for prediction safeguards

AI-assisted code was checked using `pytest`. Tests verify prediction round-tripping, missing-feature rejection, nonnegative numeric validation, subgroup metric behavior, and the explicit threshold fallback path.

```python
# tests/test_prediction.py
def test_threshold_constraint_fallback_is_explicit():
    threshold, feasible = choose_threshold(
        threshold_table([0, 0, 1, 1], [0.9, 0.9, 0.1, 0.1])
    )
    assert not feasible
    assert 0.01 <= threshold <= 0.99
```

## Examples included in the repository

- `app/example_flows.csv` contains example network-flow feature rows for the Streamlit research prototype.
- `notebooks/` contains the six research notebooks for data understanding, EDA, preprocessing/feature engineering, training, evaluation, and explainability/audit.
- `figures/` contains executed model and SHAP outputs used in the report and presentations.
- `tests/` contains executable validation tests for preprocessing, prediction, and the application.
- `demo/` contains the capstone demonstration video and its narration transcript.

## Verification and human oversight

The use of Generative AI was bounded by the following controls:

1. Dataset statistics and performance metrics are read from executed experiment outputs rather than written manually to meet desired targets.
2. `label`, `attack_cat`, and `id` are excluded from predictors to avoid direct target leakage.
3. Model-family selection and threshold selection use validation data; the final test population is reserved for evaluation.
4. The project reports unmet targets, including F1 below 0.90 and an operationally high false-positive rate.
5. SHAP explanations are described as model associations, not causal security rules.
6. No demographic fairness claims are made because the dataset lacks protected demographic attributes.

Generative AI therefore functioned as a coding and documentation assistant. It did not replace executed experimentation, methodological controls, or human scientific responsibility.
