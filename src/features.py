"""
Domain-motivated feature engineering for network flow records.

Every engineered feature here is a **row-wise function of a single flow**. None
of them uses a statistic estimated from the data (no target encoding, no
dataset-level means, no group aggregates). That property matters twice over:

* **No target leakage.** A feature cannot leak the label if it never sees the
  label, and cannot leak across the split boundary if it never sees another row.
* **Deployable.** Each feature can be computed on one flow record in isolation
  by a sensor at inference time, which is exactly how an IDS must operate.

Because the transformer is stateless, ``fit`` is a no-op; it is still written as
a scikit-learn transformer so it can sit inside the same ``Pipeline`` as the
scaler and the estimator and be persisted as one object.

Numerical safety
----------------
Network data is full of legitimate zeros: 46.7% of UNSW-NB15 flows receive no
response at all (``dpkts == 0``), and 3,607 flows report ``dur == 0``. Every
ratio below is therefore computed against a guarded denominator, and ratios are
formed as ``a / (a + b)`` (bounded to [0, 1]) rather than ``a / b`` (unbounded,
explosive) wherever the semantics allow it. The module asserts on output that no
non-finite value was produced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config

#: Tiny constant guarding denominators that can legitimately be zero.
EPSILON: float = 1e-9

#: name -> (cybersecurity rationale). Mirrored into the EDA report and the data
#: dictionary so that every engineered column is documented in one place.
ENGINEERED_FEATURE_DOCS: dict[str, str] = {
    "flow_bytes_total": (
        "Total bytes carried in both directions. Separates bulk transfers and "
        "data exfiltration from the tiny probe flows typical of scanning."
    ),
    "flow_pkts_total": (
        "Total packets in both directions. A volume counterpart to byte count "
        "that is insensitive to payload size, so it isolates packet-rate attacks."
    ),
    "bytes_per_packet": (
        "Mean payload size across the whole flow. Floods and scans use minimal "
        "packets; exploit and shellcode deliveries push large payloads into few "
        "packets."
    ),
    "src_byte_ratio": (
        "Share of flow bytes sent by the source, in [0, 1]. Values near 1.0 mean "
        "the source talked and the destination barely answered - the signature "
        "of scanning, spoofed floods and one-way probing. Values near 0 indicate "
        "a download-shaped flow."
    ),
    "src_pkt_ratio": (
        "Packet-count analogue of src_byte_ratio, in [0, 1]. Robust when payload "
        "sizes vary, e.g. reconnaissance that provokes variable-length errors."
    ),
    "load_log_ratio": (
        "log1p(source bits/s) - log1p(destination bits/s). A signed, "
        "scale-stable measure of throughput asymmetry: strongly positive for "
        "outbound floods, strongly negative for large inbound responses."
    ),
    "jit_log_ratio": (
        "log1p(source jitter) - log1p(destination jitter). Machine-generated "
        "attack traffic is often unnaturally regular (low jitter) compared with "
        "human-driven sessions."
    ),
    "is_one_way": (
        "1 when the destination returned no packets at all. The single clearest "
        "indicator of unanswered probing, closed-port scanning, or traffic sent "
        "to a spoofed or non-existent host."
    ),
    "tcp_handshake_complete": (
        "1 when both SYN-ACK and ACK-data round-trip times are non-zero, i.e. a "
        "full TCP three-way handshake completed. Half-open SYN scans and SYN "
        "floods deliberately never reach this state."
    ),
    "tcp_seq_exchanged": (
        "1 when TCP base sequence numbers were observed in both directions, "
        "confirming a genuinely established bidirectional TCP session rather "
        "than a synthetic or aborted one."
    ),
    "both_win_advertised": (
        "1 when both endpoints advertised a non-zero TCP receive window. Absence "
        "signals a connection that never reached a data-transfer state."
    ),
    "is_zero_duration": (
        "1 when the flow's measured duration is exactly zero - a single-packet "
        "or sub-resolution event. Common for stateless UDP probes and for "
        "measurement artefacts that should be visible to the model rather than "
        "silently absorbed into a rate denominator."
    ),
    "service_unknown": (
        "1 when the flow analyser could not identify an application-layer "
        "service. Traffic on non-standard ports that resists classification is "
        "operationally suspicious in its own right - though note that in THIS "
        "corpus the univariate association runs the other way (identified "
        "services carry the higher attack rate), which is a property of the "
        "testbed's traffic mix rather than of real networks. See fig10."
    ),
    "src_loss_rate": (
        "Fraction of source packets that were retransmitted or lost, in [0, 1]. "
        "Elevated loss accompanies congestion-inducing floods and unstable "
        "exploit payloads."
    ),
    "dst_loss_rate": (
        "Fraction of destination packets retransmitted or lost, in [0, 1]. "
        "Signals a target under strain - a denial-of-service symptom."
    ),
}

#: Columns the transformer needs on input.
REQUIRED_COLUMNS: tuple[str, ...] = (
    "sbytes", "dbytes", "spkts", "dpkts", "sload", "dload",
    "sjit", "djit", "synack", "ackdat", "stcpb", "dtcpb",
    "swin", "dwin", "dur", "service", "sloss", "dloss",
)


def _safe_share(numerator: pd.Series, other: pd.Series) -> pd.Series:
    """Return ``numerator / (numerator + other)`` bounded to [0, 1], 0/0 -> 0.5.

    A 0/0 flow carried nothing in either direction, so neither side dominates;
    0.5 is the neutral, non-informative encoding rather than an arbitrary 0.
    """
    total = numerator + other
    return np.where(total > 0, numerator / (total + EPSILON), 0.5)


def add_engineered_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of ``frame`` with the engineered columns appended.

    Pure and row-wise: the output for a given row depends only on that row.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "Cannot engineer features; input is missing required flow columns: "
            f"{missing}. Expected a UNSW-NB15-shaped frame."
        )

    out = frame.copy()

    sbytes, dbytes = out["sbytes"].astype(float), out["dbytes"].astype(float)
    spkts, dpkts = out["spkts"].astype(float), out["dpkts"].astype(float)

    # --- Volume -------------------------------------------------------------
    out["flow_bytes_total"] = sbytes + dbytes
    out["flow_pkts_total"] = spkts + dpkts
    # spkts >= 1 for every record in UNSW-NB15, but guard anyway so the function
    # stays correct on arbitrary input (e.g. a hand-built row from the app).
    out["bytes_per_packet"] = out["flow_bytes_total"] / out["flow_pkts_total"].clip(lower=1.0)

    # --- Directionality -----------------------------------------------------
    out["src_byte_ratio"] = _safe_share(sbytes, dbytes)
    out["src_pkt_ratio"] = _safe_share(spkts, dpkts)
    out["load_log_ratio"] = (
        np.log1p(out["sload"].clip(lower=0)) - np.log1p(out["dload"].clip(lower=0))
    )
    out["jit_log_ratio"] = (
        np.log1p(out["sjit"].clip(lower=0)) - np.log1p(out["djit"].clip(lower=0))
    )
    out["is_one_way"] = (dpkts == 0).astype("int8")

    # --- TCP session state --------------------------------------------------
    out["tcp_handshake_complete"] = (
        (out["synack"] > 0) & (out["ackdat"] > 0)
    ).astype("int8")
    out["tcp_seq_exchanged"] = ((out["stcpb"] > 0) & (out["dtcpb"] > 0)).astype("int8")
    out["both_win_advertised"] = ((out["swin"] > 0) & (out["dwin"] > 0)).astype("int8")

    # --- Measurement / identification flags ---------------------------------
    out["is_zero_duration"] = (out["dur"] <= 0).astype("int8")
    out["service_unknown"] = (
        out["service"].astype(str).str.strip().str.lower()
        == config.SERVICE_MISSING_TOKEN
    ).astype("int8")

    # --- Reliability --------------------------------------------------------
    out["src_loss_rate"] = (out["sloss"].astype(float) / spkts.clip(lower=1.0)).clip(0, 1)
    out["dst_loss_rate"] = (out["dloss"].astype(float) / dpkts.clip(lower=1.0)).clip(0, 1)

    engineered = list(ENGINEERED_FEATURE_DOCS)
    block = out[engineered].to_numpy(dtype=float)
    if not np.isfinite(block).all():
        bad = [c for c in engineered if not np.isfinite(out[c].to_numpy(dtype=float)).all()]
        raise FloatingPointError(
            f"Engineered features produced non-finite values in: {bad}. "
            "This indicates an unguarded division; fix src/features.py."
        )
    return out


class NetworkFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Stateless scikit-learn transformer wrapping :func:`add_engineered_features`.

    Parameters
    ----------
    enabled:
        When ``False`` the transformer is a pass-through. This exists so the
        "with vs. without engineered features" ablation can be run by flipping
        one pipeline parameter instead of maintaining a second code path.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803 - sklearn API
        # Nothing is learned. Recorded only so sklearn considers this fitted.
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        if not self.enabled:
            return X.copy()
        return add_engineered_features(X)

    def get_feature_names_out(self, input_features=None):
        base = list(input_features if input_features is not None
                    else getattr(self, "feature_names_in_", []))
        if not self.enabled:
            return np.asarray(base, dtype=object)
        return np.asarray(base + list(ENGINEERED_FEATURE_DOCS), dtype=object)
