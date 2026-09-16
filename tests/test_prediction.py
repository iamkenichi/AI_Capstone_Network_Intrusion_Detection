"""
Tests for the inference API - the code path the Streamlit app and any batch
scoring job actually run.

Covers input validation (the errors a user will hit), risk banding, and an
end-to-end round trip through the persisted model. The round-trip tests skip
cleanly when no model has been trained yet, so the suite is runnable on a fresh
clone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, predict
from tests.conftest import requires_data, requires_model


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def test_validate_input_accepts_a_well_formed_frame(
        synthetic_flows: pd.DataFrame) -> None:
    out = predict.validate_input(synthetic_flows)
    assert list(out.columns) == list(predict.REQUIRED_INPUT_COLUMNS)
    assert len(out) == len(synthetic_flows)


def test_validate_input_reorders_columns(synthetic_flows: pd.DataFrame) -> None:
    """Column order in the caller's CSV must not matter."""
    shuffled = synthetic_flows[list(reversed(synthetic_flows.columns))]
    out = predict.validate_input(shuffled)
    assert list(out.columns) == list(predict.REQUIRED_INPUT_COLUMNS)


def test_validate_input_names_the_missing_columns(
        synthetic_flows: pd.DataFrame) -> None:
    """The error must say exactly what is missing, not fail deep in sklearn."""
    broken = synthetic_flows.drop(columns=["sttl", "dbytes", "ct_srv_src"])
    with pytest.raises(ValueError) as excinfo:
        predict.validate_input(broken)

    message = str(excinfo.value)
    assert "missing" in message.lower()
    for column in ("sttl", "dbytes", "ct_srv_src"):
        assert column in message


def test_validate_input_rejects_empty_and_non_numeric(
        synthetic_flows: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="zero rows"):
        predict.validate_input(synthetic_flows.head(0))

    # Cast the column first: assigning a string into an int64 column is itself
    # deprecated in pandas, and the behaviour under test is what happens when a
    # user's CSV genuinely contains text in a numeric field.
    corrupted = synthetic_flows.copy()
    corrupted["sbytes"] = corrupted["sbytes"].astype(object)
    corrupted.loc[0, "sbytes"] = "not-a-number"
    with pytest.raises(ValueError, match="must be numeric"):
        predict.validate_input(corrupted)


def test_validate_input_normalises_categorical_case(
        synthetic_flows: pd.DataFrame) -> None:
    messy = synthetic_flows.copy()
    messy["proto"] = "  TCP  "
    messy["state"] = "FIN"
    out = predict.validate_input(messy)
    assert (out["proto"] == "tcp").all()
    assert (out["state"] == "fin").all()


# --------------------------------------------------------------------------- #
# Risk banding
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.99, "Critical"), (0.90, "Critical"), (0.89, "High"), (0.70, "High"),
     (0.69, "Medium"), (0.40, "Medium"), (0.39, "Low"), (0.00, "Low")],
)
def test_risk_band_boundaries(probability: float, expected: str) -> None:
    band, action = predict.risk_band(probability)
    assert band == expected
    assert action, "every band must carry a recommended action"


def test_risk_bands_are_ordered_and_exhaustive() -> None:
    floors = [floor for floor, _, _ in predict.RISK_BANDS]
    assert floors == sorted(floors, reverse=True), "bands must descend"
    assert floors[-1] == 0.0, "the lowest band must catch every probability"
    assert len({band for _, band, _ in predict.RISK_BANDS}) == len(predict.RISK_BANDS)


# --------------------------------------------------------------------------- #
# End-to-end inference
# --------------------------------------------------------------------------- #
@requires_model
def test_model_loads_with_its_frozen_threshold() -> None:
    model = predict.load_model()
    assert model.pipeline is not None
    assert 0.0 < model.threshold < 1.0
    assert model.metadata.get("selected_on") == "validation split", (
        "the deployment manifest must record that selection used validation data")


@requires_model
def test_predict_returns_a_complete_result(synthetic_flows: pd.DataFrame) -> None:
    result = predict.predict(synthetic_flows)

    assert len(result) == len(synthetic_flows)
    for column in ("attack_probability", "prediction", "verdict",
                   "risk_band", "recommended_action", "threshold"):
        assert column in result.columns

    probability = result["attack_probability"].to_numpy(float)
    assert np.isfinite(probability).all()
    assert ((probability >= 0.0) & (probability <= 1.0)).all()
    assert set(result["prediction"].unique()) <= {0, 1}
    assert set(result["verdict"].unique()) <= {"ATTACK", "BENIGN"}


@requires_model
def test_prediction_agrees_with_threshold(synthetic_flows: pd.DataFrame) -> None:
    """The binary verdict must be exactly `probability >= threshold`."""
    result = predict.predict(synthetic_flows, threshold=0.42)
    expected = (result["attack_probability"] >= 0.42).astype(int)
    pd.testing.assert_series_equal(
        result["prediction"], expected, check_names=False, check_dtype=False)


@requires_model
def test_prediction_is_deterministic(synthetic_flows: pd.DataFrame) -> None:
    first = predict.predict(synthetic_flows)
    second = predict.predict(synthetic_flows)
    np.testing.assert_array_equal(
        first["attack_probability"].to_numpy(), second["attack_probability"].to_numpy())


@requires_model
def test_batch_and_single_scoring_agree(synthetic_flows: pd.DataFrame) -> None:
    """No training/serving skew between the batch and single-record paths."""
    batch = predict.predict(synthetic_flows)
    for position in range(len(synthetic_flows)):
        single = predict.predict_one(synthetic_flows.iloc[position].to_dict())
        assert single["attack_probability"] == pytest.approx(
            batch["attack_probability"].iloc[position], abs=1e-9)


@requires_model
def test_unseen_protocol_does_not_crash_inference(
        synthetic_flows: pd.DataFrame) -> None:
    novel = synthetic_flows.copy()
    novel["proto"] = "some-protocol-invented-in-2030"
    novel["service"] = "unheard-of-service"
    result = predict.predict(novel)
    assert np.isfinite(result["attack_probability"].to_numpy(float)).all()


@requires_model
@requires_data
def test_model_beats_the_majority_class_baseline() -> None:
    """
    A real accuracy check on held-out data.

    Not a smoke test: it asserts the deployed model genuinely outperforms
    always-predicting-the-majority-class on the test split, which is the
    minimum bar for the artefact being worth shipping at all.
    """
    test_path = config.PROCESSED_DIR / "test.parquet"
    if not test_path.exists():
        pytest.skip("data/processed/test.parquet not present; run src.preprocessing")

    test = pd.read_parquet(test_path).sample(
        n=4000, random_state=config.RANDOM_STATE)
    result = predict.predict(test)

    truth = test[config.TARGET].to_numpy(int)
    predicted = result["prediction"].to_numpy(int)

    accuracy = float((predicted == truth).mean())
    majority = float(max(truth.mean(), 1 - truth.mean()))
    recall = float(predicted[truth == 1].mean())

    assert accuracy > majority + 0.10, (
        f"accuracy {accuracy:.4f} barely beats the {majority:.4f} majority baseline")
    assert recall > 0.70, f"attack recall {recall:.4f} is too low to be useful"


@requires_model
def test_explanation_is_available_for_the_deployed_model(
        synthetic_flows: pd.DataFrame) -> None:
    """A deployed detector must be able to justify an individual alert."""
    explanation = predict.explain_one(synthetic_flows.iloc[0].to_dict(), top_n=6)
    if explanation is None:
        pytest.skip("deployed model is not a tree ensemble; SHAP not applicable")

    assert len(explanation) <= 6
    assert set(explanation.columns) >= {"feature", "value", "shap_value", "direction"}
    assert np.isfinite(explanation["shap_value"].to_numpy(float)).all()
    assert explanation["shap_value"].abs().is_monotonic_decreasing


@requires_model
def test_feature_reference_covers_every_input_column() -> None:
    """The app's form must be able to render a control for every model input."""
    reference = predict.feature_reference()
    covered = set(reference["numeric"]) | set(reference["categorical"])
    assert covered == set(predict.REQUIRED_INPUT_COLUMNS)

    for name, example in reference["examples"].items():
        missing = set(predict.REQUIRED_INPUT_COLUMNS) - set(example)
        assert not missing, f"example '{name}' is missing {missing}"
