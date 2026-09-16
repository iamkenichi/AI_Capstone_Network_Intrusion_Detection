"""
Tests for splitting and the preprocessing pipeline.

These are the tests that matter most in this project, because the claims they
verify - no target leakage, no preprocessing leakage, no duplicate leakage - are
exactly the claims a reader has to take on trust everywhere else. Each one
asserts a property that would silently inflate the reported metrics if it broke.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, preprocessing
from src.features import ENGINEERED_FEATURE_DOCS
from tests.conftest import requires_data


# --------------------------------------------------------------------------- #
# Leakage control
# --------------------------------------------------------------------------- #
@requires_data
def test_split_xy_removes_every_leakage_column(small_corpus: pd.DataFrame) -> None:
    """`attack_cat`, `label`, `id` and `partition` must never reach the model."""
    X, y = preprocessing.split_xy(small_corpus)

    for forbidden in (*config.LEAKAGE_COLUMNS, "partition"):
        assert forbidden not in X.columns, (
            f"'{forbidden}' survived into the feature matrix - this is target leakage."
        )
    assert y.name == config.TARGET
    assert set(y.unique()) <= {0, 1}
    assert len(X) == len(y)


@requires_data
def test_attack_cat_determines_label(small_corpus: pd.DataFrame) -> None:
    """
    Document the reason attack_cat is banned as a predictor.

    If this ever fails, the leakage justification in the reports is wrong and
    must be revisited - so the assumption is asserted rather than assumed.
    """
    is_normal = small_corpus[config.ATTACK_CAT].str.lower() == "normal"
    assert (is_normal == (small_corpus[config.TARGET] == 0)).all()


@requires_data
def test_transformed_matrix_contains_no_target_information(
        small_corpus: pd.DataFrame) -> None:
    """No transformed column may be perfectly correlated with the target."""
    X, y = preprocessing.split_xy(small_corpus)
    pipeline = preprocessing.build_feature_pipeline()
    design = pipeline.fit_transform(X, y)
    names = preprocessing.transformed_feature_names(pipeline)

    frame = pd.DataFrame(design, columns=names)
    correlations = frame.corrwith(pd.Series(y.to_numpy(), index=frame.index)).abs()
    worst = correlations.dropna().max()
    assert worst < 0.999, (
        f"Feature '{correlations.idxmax()}' correlates {worst:.4f} with the target, "
        "which indicates leakage rather than signal."
    )


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #
@requires_data
def test_splits_are_disjoint_and_correctly_sized(small_corpus: pd.DataFrame) -> None:
    train, val, test = preprocessing.split_corpus(small_corpus, verbose=False)

    total = len(small_corpus)
    assert len(train) + len(val) + len(test) == total

    # 60/20/20, allowing for integer rounding on a small sample.
    assert len(test) / total == pytest.approx(config.TEST_SIZE, abs=0.01)
    assert len(val) / total == pytest.approx(config.VAL_SIZE, abs=0.01)

    # Disjoint by content, not merely by index: reset_index() has already been
    # applied, so positional indices would collide and prove nothing.
    def fingerprint(frame: pd.DataFrame) -> set[tuple]:
        return set(map(tuple, frame[["id", "partition"]].itertuples(index=False)))

    assert not fingerprint(train) & fingerprint(test)
    assert not fingerprint(train) & fingerprint(val)
    assert not fingerprint(val) & fingerprint(test)


@requires_data
def test_split_is_stratified_by_attack_family(small_corpus: pd.DataFrame) -> None:
    """Every attack family must appear in all three splits, in similar proportion."""
    train, val, test = preprocessing.split_corpus(small_corpus, verbose=False)

    families = set(small_corpus[config.ATTACK_CAT].unique())
    for name, part in (("train", train), ("val", val), ("test", test)):
        assert set(part[config.ATTACK_CAT].unique()) == families, (
            f"{name} split is missing attack families - stratification failed.")

    reference = small_corpus[config.TARGET].mean()
    for part in (train, val, test):
        assert part[config.TARGET].mean() == pytest.approx(reference, abs=0.01)


@requires_data
def test_split_is_reproducible(small_corpus: pd.DataFrame) -> None:
    """The same seed must produce byte-identical splits."""
    first = preprocessing.split_corpus(small_corpus, verbose=False)
    second = preprocessing.split_corpus(small_corpus, verbose=False)
    for a, b in zip(first, second):
        pd.testing.assert_frame_equal(a, b)


@requires_data
def test_deduplication_removes_cross_split_duplicates() -> None:
    """
    After deduplication no feature vector may appear in more than one split.

    This is the property that makes the reported test metrics an estimate of
    generalisation rather than partly a measure of memorisation.
    """
    corpus, report = preprocessing.prepare_corpus(deduplicate=True, verbose=False)
    predictors = [c for c in corpus.columns
                  if c not in (*config.LEAKAGE_COLUMNS, "partition")]

    assert not corpus.duplicated(subset=predictors).any()
    assert report["rows_final"] < report["rows_loaded"]
    assert report["duplicate_rows"] > 0

    train, val, test = preprocessing.split_corpus(corpus, verbose=False)
    train_vectors = set(map(tuple, train[predictors].itertuples(index=False)))
    test_vectors = set(map(tuple, test[predictors].itertuples(index=False)))
    assert not train_vectors & test_vectors, (
        "Identical feature vectors appear in both train and test after "
        "deduplication - the split is leaking.")


# --------------------------------------------------------------------------- #
# The preprocessing pipeline itself
# --------------------------------------------------------------------------- #
@requires_data
def test_scaler_is_fitted_on_training_data_only(small_corpus: pd.DataFrame) -> None:
    """
    The fitted scaler's statistics must match the training split exactly.

    If validation or test rows had leaked into `fit`, the learned means would
    differ from the training-split means.
    """
    train, val, _ = preprocessing.split_corpus(small_corpus, verbose=False)
    X_train, y_train = preprocessing.split_xy(train)

    pipeline = preprocessing.build_feature_pipeline()
    pipeline.fit(X_train, y_train)

    engineered = pipeline.named_steps["engineer"].transform(X_train)
    column_transformer = pipeline.named_steps["preprocess"]
    scaler = column_transformer.named_transformers_["num"]
    linear_columns = preprocessing.feature_columns()["linear"]

    expected = engineered[linear_columns].to_numpy(float).mean(axis=0)
    np.testing.assert_allclose(scaler.mean_, expected, rtol=1e-9, atol=1e-9)

    # And transforming the validation split must not mutate the fitted state.
    before = scaler.mean_.copy()
    pipeline.transform(preprocessing.split_xy(val)[0])
    np.testing.assert_array_equal(scaler.mean_, before)


@requires_data
def test_pipeline_output_is_finite_and_correctly_shaped(
        small_corpus: pd.DataFrame) -> None:
    X, y = preprocessing.split_xy(small_corpus)
    pipeline = preprocessing.build_feature_pipeline()
    design = np.asarray(pipeline.fit_transform(X, y))

    assert design.shape[0] == len(X)
    assert np.isfinite(design).all(), "Preprocessing produced NaN or infinite values."
    assert design.shape[1] == len(preprocessing.transformed_feature_names(pipeline))


@requires_data
def test_unseen_category_does_not_break_inference(small_corpus: pd.DataFrame) -> None:
    """
    A protocol never seen during training must be absorbed, not raise.

    This is the failure mode that takes a deployed detector offline the first
    time an unusual protocol crosses the wire.
    """
    train, val, _ = preprocessing.split_corpus(small_corpus, verbose=False)
    X_train, y_train = preprocessing.split_xy(train)
    X_val, _ = preprocessing.split_xy(val)

    pipeline = preprocessing.build_feature_pipeline()
    pipeline.fit(X_train, y_train)

    novel = X_val.head(20).copy()
    novel["proto"] = "quic-over-carrier-pigeon"
    novel["service"] = "definitely-not-a-real-service"
    novel["state"] = "zzz"

    design = np.asarray(pipeline.transform(novel))
    assert design.shape == (20, len(preprocessing.transformed_feature_names(pipeline)))
    assert np.isfinite(design).all()


@requires_data
def test_column_groups_cover_every_modelled_column(small_corpus: pd.DataFrame) -> None:
    """No predictor may be silently dropped by falling through every branch."""
    X, _ = preprocessing.split_xy(small_corpus)
    engineered = preprocessing.build_feature_pipeline().named_steps["engineer"].fit_transform(X)

    groups = preprocessing.feature_columns()
    assigned = set(sum(groups.values(), []))
    missing = set(engineered.columns) - assigned

    assert not missing, (
        f"These columns are produced but never consumed by the ColumnTransformer: "
        f"{sorted(missing)}. Either assign them to a group or document the omission.")


def test_ablation_switches_change_the_column_set() -> None:
    """The ablation switches must genuinely alter what the model sees."""
    full = preprocessing.feature_columns()
    no_ttl = preprocessing.feature_columns(exclude=config.TTL_FEATURES)
    no_engineered = preprocessing.feature_columns(include_engineered=False)

    for feature in config.TTL_FEATURES:
        assert feature in sum(full.values(), [])
        assert feature not in sum(no_ttl.values(), [])

    assert not (set(sum(no_engineered.values(), [])) & set(ENGINEERED_FEATURE_DOCS))
    assert set(ENGINEERED_FEATURE_DOCS) & set(sum(full.values(), []))


@requires_data
def test_pipeline_survives_a_joblib_round_trip(small_corpus: pd.DataFrame,
                                               tmp_path) -> None:
    """
    The fitted pipeline must persist and reload in a fresh process.

    This is not a formality. A pipeline that trains perfectly but cannot be
    pickled is undeployable, and the failure only surfaces at the very END of a
    long training run - after the expensive search has already completed. The
    usual cause is a lambda or local closure inside a transformer, so this test
    guards a real and expensive failure mode.
    """
    import joblib

    X, y = preprocessing.split_xy(small_corpus)
    pipeline = preprocessing.build_feature_pipeline()
    expected = np.asarray(pipeline.fit_transform(X, y))

    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    reloaded = joblib.load(path)

    np.testing.assert_allclose(np.asarray(reloaded.transform(X)), expected, rtol=1e-12)
    assert (preprocessing.transformed_feature_names(reloaded)
            == preprocessing.transformed_feature_names(pipeline))


def test_log1p_helpers_are_picklable_module_functions() -> None:
    """Guard the specific defect above: these must not become lambdas again."""
    import pickle

    for function in (preprocessing.safe_log1p, preprocessing.safe_expm1):
        assert function.__name__ != "<lambda>"
        assert pickle.loads(pickle.dumps(function)) is function

    values = np.array([0.0, 1.0, 1e9, -5.0])
    transformed = preprocessing.safe_log1p(values)
    assert np.isfinite(transformed).all()
    assert transformed[0] == 0.0                 # log1p(0) == 0
    assert transformed[3] == 0.0                 # negatives are clipped to 0
    np.testing.assert_allclose(
        preprocessing.safe_expm1(transformed[:3]), [0.0, 1.0, 1e9], rtol=1e-9)


@requires_data
def test_prepare_corpus_reports_known_defects() -> None:
    """The preparation report must surface the duplication problem, not hide it."""
    _, report = preprocessing.prepare_corpus(deduplicate=True, verbose=False)

    assert report["missing_values"] == 0
    assert 0.0 < report["duplicate_fraction"] < 1.0
    assert report["conflicting_feature_vectors"] >= 0
    assert set(report["class_balance"]) == {0, 1}
    assert sum(report["class_balance"].values()) == pytest.approx(1.0, abs=1e-3)
