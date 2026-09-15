"""Validate published CSV schema and enforce predictor boundaries."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CATEGORICAL = ['proto', 'service', 'state']
EXCLUDED = ['id', 'label', 'attack_cat']


def load_partition(name):
    path = ROOT / 'data' / 'raw' / f'UNSW_NB15_{name}-set.csv'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path.name}. See data/raw/README.md for download instructions.')
    frame = pd.read_csv(path)
    required = set(CATEGORICAL + EXCLUDED + ['sbytes', 'dbytes', 'spkts', 'dpkts', 'dur'])
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(frame.columns))}')
    if frame.label.isna().any() or not set(frame.label.unique()).issubset({0, 1}):
        raise ValueError('label must be complete and binary (0 benign, 1 attack).')
    frame['attack_cat'] = frame.attack_cat.astype(str).str.strip()
    if ((frame.attack_cat.str.lower() == 'normal') != (frame.label == 0)).any():
        raise ValueError('attack_cat and label disagree about Normal records.')
    return frame


def predictors(frame):
    out = frame.drop(columns=EXCLUDED, errors='ignore').copy()
    for name in CATEGORICAL:
        out[name] = out[name].map(lambda x: str(x).strip() if pd.notna(x) else np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def fingerprints(frame):
    return pd.util.hash_pandas_object(predictors(frame), index=False)


def deduplicate(frame):
    # Keep the first observation of identical predictors, regardless of row ID.
    hashes = fingerprints(frame)
    conflict = frame.groupby(hashes.to_numpy()).label.nunique().gt(1)
    if conflict.any():
        # Ambiguous predictor signatures cannot support a deterministic label.
        frame = frame.loc[~hashes.isin(conflict[conflict].index)]
        hashes = fingerprints(frame)
    return frame.loc[~hashes.duplicated()].copy(), int(conflict.sum())
