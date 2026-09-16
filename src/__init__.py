"""
Machine Learning-Based Network Intrusion Detection and Anomaly Classification.

Reusable source package for the capstone. Every notebook, test and the
Streamlit application import from here rather than redefining logic, so there
is exactly one implementation of each pipeline stage.

Modules
-------
config         Paths, seeds, column roles, leakage exclusions, plot styling.
data_loader    UNSW-NB15 download, integrity verification and loading.
preprocessing  Leak-free splitting and the sklearn ColumnTransformer pipeline.
features       Domain-motivated network-traffic feature engineering.
train          Model definitions, hyper-parameter search, persistence.
evaluate       Metrics, curves, threshold analysis, subgroup audit, SHAP.
predict        Inference API used by the tests and the Streamlit app.
"""

__version__ = "1.0.0"
__all__ = [
    "config",
    "data_loader",
    "preprocessing",
    "features",
    "train",
    "evaluate",
    "predict",
]
