"""
Tests that the SHAP explanation describes the model it claims to describe.

This matters more than it looks. SHAP's TreeExplainer returns a different array
shape for different estimator families - XGBoost gives ``(n, features)`` for a
single logit, scikit-learn and LightGBM can give ``(n, features, 2)`` or a list
per class. If the class axis is picked up wrongly the explanation still *renders*
perfectly well; it simply describes the benign class while the report says it
describes attacks. Nothing downstream would catch that, which is why the
additivity identity is asserted here rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src import explain, predict, preprocessing
from tests.conftest import requires_data, requires_model


@pytest.fixture(scope="module")
def explanation_bundle():
    """A small SHAP explanation computed on real held-out flows."""
    model = predict.load_model()
    _, _, test, _ = preprocessing.split_published(verbose=False)
    X_test, _ = preprocessing.split_xy(test)
    rows = X_test.head(200)
    explanation, design = explain.compute_shap(model.pipeline, rows, sample_size=None)
    return model, rows, explanation, design


@requires_data
@requires_model
def test_shap_values_are_two_dimensional_after_class_selection(explanation_bundle) -> None:
    _, rows, explanation, _ = explanation_bundle
    values = np.asarray(explanation.values)
    assert values.ndim == 2, (
        f"expected (n_rows, n_features) after selecting the attack class, got {values.shape}")
    assert values.shape[0] == len(rows)
    assert values.shape[1] == len(explanation.feature_names)


@requires_data
@requires_model
def test_shap_values_satisfy_additivity(explanation_bundle) -> None:
    """base_value + sum(contributions) must reproduce the model's raw margin.

    This is the property that fails loudly if the class axis is mishandled.
    """
    model, _, explanation, design = explanation_bundle
    estimator = model.pipeline.named_steps[list(model.pipeline.named_steps)[-1]]

    if hasattr(estimator, "predict") and "lightgbm" in type(estimator).__module__:
        raw = estimator.predict(design, raw_score=True)
    elif hasattr(estimator, "predict"):
        raw = estimator.predict(design, output_margin=True)
    else:
        pytest.skip(f"no raw-margin accessor for {type(estimator).__name__}")

    reconstructed = np.asarray(explanation.base_values).ravel() + \
        np.asarray(explanation.values).sum(axis=1)
    np.testing.assert_allclose(reconstructed, raw, rtol=1e-5, atol=1e-6)


@requires_data
@requires_model
def test_explanation_is_oriented_to_the_attack_class(explanation_bundle) -> None:
    """A larger total contribution must mean a higher probability of ATTACK.

    If the benign-class axis were selected, this correlation would be negative
    and every directional statement in the reports would be backwards.
    """
    model, rows, explanation, _ = explanation_bundle
    total = np.asarray(explanation.values).sum(axis=1)
    proba = model.pipeline.predict_proba(rows)[:, 1]
    correlation = float(np.corrcoef(total, proba)[0, 1])
    assert correlation > 0.5, (
        f"SHAP totals correlate {correlation:.3f} with attack probability; "
        "the explanation appears to describe the wrong class")


@requires_data
@requires_model
def test_global_importance_shares_are_normalised_over_all_features(
    explanation_bundle,
) -> None:
    """Shares are a fraction of the WHOLE feature set, not of the rows returned.

    `global_importance` truncates to the top n for display, so the returned
    shares deliberately sum to less than one - that remainder is the impact
    carried by the features not shown, and the reports rely on the shares
    meaning "of total attributed impact" rather than "of what is displayed".
    """
    _, _, explanation, _ = explanation_bundle
    n_features = len(explanation.feature_names)

    everything = explain.global_importance(explanation, top_n=n_features)
    assert len(everything) == n_features
    assert (everything["mean_abs_shap"] >= 0).all()
    assert everything["share_of_total"].sum() == pytest.approx(1.0, abs=1e-9)
    assert everything["mean_abs_shap"].is_monotonic_decreasing, "not ranked by importance"

    truncated = explain.global_importance(explanation, top_n=10)
    assert len(truncated) == 10
    assert truncated["share_of_total"].sum() < 1.0
    # The same feature must carry the same share whether or not it was truncated.
    assert truncated.loc[0, "share_of_total"] == pytest.approx(
        everything.loc[0, "share_of_total"], abs=1e-12)
