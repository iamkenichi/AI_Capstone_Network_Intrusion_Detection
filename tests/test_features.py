"""
Tests for the engineered network-security features.

The properties asserted here are the ones the feature design *claims*:
row-wise independence (hence no leakage), numerical safety on the degenerate
flows that make up nearly half the dataset, bounded ranges where the definition
says bounded, and correct semantics for the session-state flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import (
    ENGINEERED_FEATURE_DOCS,
    NetworkFeatureEngineer,
    add_engineered_features,
)
from tests.conftest import requires_data


# --------------------------------------------------------------------------- #
# Numerical safety
# --------------------------------------------------------------------------- #
def test_degenerate_flows_produce_finite_values(synthetic_flows: pd.DataFrame) -> None:
    """
    Zero-duration, zero-response flows must not produce NaN or inf.

    46.7% of UNSW-NB15 flows have `dpkts == 0` and 3,607 have `dur == 0`, so an
    unguarded division here would corrupt a large fraction of the design matrix
    rather than a rare edge case.
    """
    out = add_engineered_features(synthetic_flows)
    block = out[list(ENGINEERED_FEATURE_DOCS)].to_numpy(float)
    assert np.isfinite(block).all()


def test_all_zero_flow_is_handled() -> None:
    """The pathological all-zero record must still yield finite output."""
    from src.predict import REQUIRED_INPUT_COLUMNS

    record = {column: 0 for column in REQUIRED_INPUT_COLUMNS}
    record.update({"proto": "tcp", "service": "-", "state": "int"})
    out = add_engineered_features(pd.DataFrame([record]))

    block = out[list(ENGINEERED_FEATURE_DOCS)].to_numpy(float)
    assert np.isfinite(block).all()
    # 0/0 direction is genuinely undetermined and encodes as the neutral 0.5.
    assert out["src_byte_ratio"].iloc[0] == pytest.approx(0.5)
    assert out["src_pkt_ratio"].iloc[0] == pytest.approx(0.5)


def test_ratios_stay_within_their_declared_bounds(synthetic_flows: pd.DataFrame) -> None:
    out = add_engineered_features(synthetic_flows)
    for column in ("src_byte_ratio", "src_pkt_ratio", "src_loss_rate", "dst_loss_rate"):
        values = out[column].to_numpy(float)
        assert (values >= 0.0).all() and (values <= 1.0).all(), (
            f"{column} escaped [0, 1]: min={values.min()}, max={values.max()}")


def test_extreme_magnitudes_do_not_overflow(synthetic_flows: pd.DataFrame) -> None:
    """The 14 MB / 5.99 Gbit/s record must not overflow float64."""
    extreme = add_engineered_features(synthetic_flows.iloc[[2]])
    assert np.isfinite(extreme[list(ENGINEERED_FEATURE_DOCS)].to_numpy(float)).all()
    assert extreme["flow_bytes_total"].iloc[0] > 2.8e7


# --------------------------------------------------------------------------- #
# Semantics
# --------------------------------------------------------------------------- #
def test_flag_semantics(synthetic_flows: pd.DataFrame) -> None:
    """Each session-state flag must mean what its documentation says."""
    out = add_engineered_features(synthetic_flows)

    ordinary, one_way, _ = 0, 1, 2

    # Row 1 is the one-way probe: no response, no handshake, zero duration.
    assert out["is_one_way"].iloc[one_way] == 1
    assert out["is_one_way"].iloc[ordinary] == 0
    assert out["is_zero_duration"].iloc[one_way] == 1
    assert out["is_zero_duration"].iloc[ordinary] == 0
    assert out["service_unknown"].iloc[one_way] == 1
    assert out["service_unknown"].iloc[ordinary] == 0

    # Row 0 completed a handshake and exchanged TCP sequence numbers.
    assert out["tcp_handshake_complete"].iloc[ordinary] == 1
    assert out["tcp_handshake_complete"].iloc[one_way] == 0
    assert out["tcp_seq_exchanged"].iloc[ordinary] == 1
    assert out["tcp_seq_exchanged"].iloc[one_way] == 0
    assert out["both_win_advertised"].iloc[ordinary] == 1
    assert out["both_win_advertised"].iloc[one_way] == 0


def test_directionality_is_computed_correctly(synthetic_flows: pd.DataFrame) -> None:
    out = add_engineered_features(synthetic_flows)

    # Row 1 sent 200 bytes and received nothing -> ratio pinned at 1.0.
    assert out["src_byte_ratio"].iloc[1] == pytest.approx(1.0, abs=1e-6)
    # Row 0 sent 1400 of 4600 total bytes.
    assert out["src_byte_ratio"].iloc[0] == pytest.approx(1400 / 4600, abs=1e-6)
    assert out["flow_bytes_total"].iloc[0] == pytest.approx(4600.0)
    assert out["flow_pkts_total"].iloc[0] == pytest.approx(22.0)
    assert out["bytes_per_packet"].iloc[0] == pytest.approx(4600 / 22, abs=1e-6)


def test_loss_rates_are_proportions(synthetic_flows: pd.DataFrame) -> None:
    out = add_engineered_features(synthetic_flows)
    # Row 0: 1 of 12 source packets lost.
    assert out["src_loss_rate"].iloc[0] == pytest.approx(1 / 12, abs=1e-6)
    # Row 1: no destination packets at all -> guarded denominator, rate 0.
    assert out["dst_loss_rate"].iloc[1] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Row-wise independence (the no-leakage property)
# --------------------------------------------------------------------------- #
@requires_data
def test_features_are_row_wise_independent(small_corpus: pd.DataFrame) -> None:
    """
    A row's engineered values must not depend on any other row.

    This is what guarantees the transformer cannot leak information across the
    train/test boundary, and it is also what makes the features computable by a
    sensor on a single flow at inference time. Verified by shuffling the input
    and checking the output follows the same permutation.
    """
    sample = small_corpus.head(500).reset_index(drop=True)
    columns = list(ENGINEERED_FEATURE_DOCS)

    straight = add_engineered_features(sample)[columns]
    shuffled_index = sample.sample(frac=1.0, random_state=7).index
    shuffled = add_engineered_features(sample.loc[shuffled_index])[columns]

    pd.testing.assert_frame_equal(
        straight.loc[shuffled_index].reset_index(drop=True),
        shuffled.reset_index(drop=True),
    )


@requires_data
def test_single_row_matches_batch(small_corpus: pd.DataFrame) -> None:
    """Scoring one flow alone must equal scoring it inside a batch."""
    sample = small_corpus.head(64).reset_index(drop=True)
    columns = list(ENGINEERED_FEATURE_DOCS)
    batch = add_engineered_features(sample)[columns]

    for position in (0, 17, 63):
        alone = add_engineered_features(sample.iloc[[position]])[columns]
        np.testing.assert_allclose(
            alone.to_numpy(float), batch.iloc[[position]].to_numpy(float), rtol=1e-12)


# --------------------------------------------------------------------------- #
# Transformer API
# --------------------------------------------------------------------------- #
def test_transformer_is_stateless_and_sklearn_compatible(
        synthetic_flows: pd.DataFrame) -> None:
    engineer = NetworkFeatureEngineer()
    fitted = engineer.fit(synthetic_flows)
    assert fitted is engineer  # fit must return self

    out = engineer.transform(synthetic_flows)
    assert len(out.columns) == len(synthetic_flows.columns) + len(ENGINEERED_FEATURE_DOCS)

    names = engineer.get_feature_names_out(synthetic_flows.columns)
    assert list(names)[-len(ENGINEERED_FEATURE_DOCS):] == list(ENGINEERED_FEATURE_DOCS)


def test_disabled_transformer_is_a_passthrough(synthetic_flows: pd.DataFrame) -> None:
    engineer = NetworkFeatureEngineer(enabled=False).fit(synthetic_flows)
    out = engineer.transform(synthetic_flows)
    pd.testing.assert_frame_equal(out, synthetic_flows)
    assert not set(engineer.get_feature_names_out(synthetic_flows.columns)) & set(
        ENGINEERED_FEATURE_DOCS)


def test_missing_input_column_raises_a_useful_error(
        synthetic_flows: pd.DataFrame) -> None:
    broken = synthetic_flows.drop(columns=["dbytes", "synack"])
    with pytest.raises(ValueError, match="missing required flow columns"):
        add_engineered_features(broken)


def test_every_engineered_feature_is_documented() -> None:
    """No feature may reach the model without a written security rationale."""
    from src.predict import REQUIRED_INPUT_COLUMNS

    record = {column: 0 for column in REQUIRED_INPUT_COLUMNS}
    record.update({"proto": "tcp", "service": "-", "state": "int"})
    out = add_engineered_features(pd.DataFrame([record]))
    produced = set(out.columns) - set(REQUIRED_INPUT_COLUMNS)

    assert produced == set(ENGINEERED_FEATURE_DOCS)
    for name, text in ENGINEERED_FEATURE_DOCS.items():
        assert len(text) > 60, f"{name} needs a real explanation, not a label."
