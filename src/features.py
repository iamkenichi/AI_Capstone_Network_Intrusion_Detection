"""Deterministic flow features; no learned statistics or label access."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class NetworkFeatures(TransformerMixin, BaseEstimator):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X):
        out = X.copy()
        if self.enabled:
            out['total_bytes'] = out.sbytes + out.dbytes
            out['total_packets'] = out.spkts + out.dpkts
            # Undefined ratios become missing, then receive a TRAIN-fitted median.
            out['bytes_per_packet'] = out.total_bytes / out.total_packets.replace(0, np.nan)
            out['source_byte_share'] = out.sbytes / out.total_bytes.replace(0, np.nan)
            out['source_packet_share'] = out.spkts / out.total_packets.replace(0, np.nan)
            out['bytes_per_second'] = out.total_bytes / out.dur.replace(0, np.nan)
        return out.replace([np.inf, -np.inf], np.nan)

    def get_feature_names_out(self, input_features=None):
        names = list(self.feature_names_in_ if input_features is None else input_features)
        return np.asarray(names + (['total_bytes', 'total_packets', 'bytes_per_packet',
            'source_byte_share', 'source_packet_share', 'bytes_per_second'] if self.enabled else []), dtype=object)
