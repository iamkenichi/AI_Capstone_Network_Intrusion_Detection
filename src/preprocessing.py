"""All learned transformations live inside the cross-validation pipeline."""
import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.features import NetworkFeatures


def build_pipeline(estimator, engineered=True):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
                        ('scale', StandardScaler())])
    categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=20,
                                 sparse_output=False, dtype=np.float32))])
    preprocessing = ColumnTransformer([
        ('numeric', numeric, make_column_selector(dtype_include=np.number)),
        ('categorical', categorical, make_column_selector(dtype_exclude=np.number))])
    return Pipeline([('features', NetworkFeatures(engineered)), ('preprocess', preprocessing),
                     ('model', estimator)])
