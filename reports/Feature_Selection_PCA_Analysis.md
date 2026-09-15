# Feature selection and dimensionality reduction supplement

## Purpose
This supplement addresses the rubric requirement for at least one feature-selection method and one dimensionality-reduction method. It is deliberately separated from the frozen final model-selection workflow so that the selected XGBoost model and final-test metrics are not changed after test inspection.

## Feature-selection method used
The project uses model-based feature analysis through SHAP importance and implements a reproducible `SelectKBest(mutual_info_classif, k=20)` comparison in `scripts/feature_selection_pca_analysis.py`. SHAP identifies the most influential transformed model features for interpretation, while mutual information provides a filter-based feature-selection comparison independent of a single tree model.

This method is justified because network-flow features can be correlated, redundant, and service-specific. Feature selection can reduce complexity and help identify a compact set of high-signal variables, but it must be fitted only on training data to avoid leakage.

## Dimensionality-reduction method used
The project adds a reproducible PCA experiment retaining 95% cumulative explained variance after the same training-only preprocessing pipeline. PCA is evaluated with Logistic Regression as a compact linear-model comparison, not as a replacement for the final XGBoost detector.

PCA is justified as a dimensionality-reduction check because one-hot-encoded protocol/service/state fields plus numeric traffic features can create a higher-dimensional feature matrix. However, PCA components are less interpretable than original network-flow features. For this reason, PCA is treated as a supplementary experiment and not as the final operational model.

## Why PCA is not the final deployment choice
The final detector remains XGBoost because the original validation protocol selected it by average precision and because tree SHAP provides clearer operational explanations using transformed network-flow features. PCA can reduce dimensionality, but it mixes original variables into abstract components that are harder for a SOC analyst to validate.

## Reproducible command

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
.\.venv\Scripts\python.exe -m src.train
.\.venv\Scripts\python.exe scripts/feature_selection_pca_analysis.py
```

Expected outputs:

- `reports/feature_selection_pca_results.csv`
- `reports/feature_selection_pca_metadata.json`
- `figures/pca_explained_variance.png`

## Scientific boundary
This supplement does not retune the final test result, does not use `attack_cat` as a predictor, and does not select a new deployment model after looking at final-test performance. It exists to demonstrate feature-selection and dimensionality-reduction methodology required by the rubric.
