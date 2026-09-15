"""Inference uses the persisted schema and validation-selected threshold."""
import json
import joblib
import pandas as pd
from src.data_loader import ROOT, predictors, CATEGORICAL, EXCLUDED


def load_model():
    path = ROOT / 'models' / 'final_model.joblib'
    if not path.exists():
        raise FileNotFoundError('Train first: python -m src.train')
    # Only load trusted, locally generated joblib artifacts.
    return joblib.load(path), json.loads((ROOT / 'models' / 'metadata.json').read_text())


def predict(frame, model=None, metadata=None):
    if model is None:
        model, metadata = load_model()
    missing = set(metadata['features']) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing required network features: {sorted(missing)}')
    clean = predictors(frame)
    clean = clean[metadata['features']]
    for name in set(metadata['features']) - set(CATEGORICAL):
        clean[name] = pd.to_numeric(clean[name], errors='raise')
        if (clean[name].dropna() < 0).any():
            raise ValueError(f'{name} must be nonnegative.')
    p = model.predict_proba(clean)[:, 1]
    return pd.DataFrame({'prediction': ['Attack' if v >= metadata['threshold'] else 'Benign' for v in p],
        'attack_probability': p, 'threshold': metadata['threshold']}, index=frame.index)
