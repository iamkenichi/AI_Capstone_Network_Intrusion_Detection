"""
Corpus preparation, leak-free splitting, and the shared preprocessing pipeline.

The two responsibilities are deliberately separated:

``prepare_corpus`` / ``split_corpus``
    Decide *which rows* go where. Everything that could leak information across
    the split boundary (deduplication) happens here, before the split.

``build_preprocessor`` / ``build_feature_pipeline``
    Decide *how columns are transformed*. Every statistic used (category
    frequencies, means, standard deviations) is learned inside ``fit``, so when
    the pipeline is fitted on the training split alone, the validation and test
    splits cannot influence it. This is the mechanism that makes the "fit on
    train only" rule structural rather than a convention someone must remember.

Split protocol
--------------
Two stratified stages on the 10-level attack category::

    corpus  --(80/20)-->  train+val , test
    train+val --(75/25)-->  train , val

which yields 60% / 20% / 20% of the corpus. ``random_state=42`` throughout.
Stratifying on ``attack_cat`` rather than ``label`` guarantees that rare
families (Worms, n=164) are present in all three partitions; because ``label``
is a deterministic function of ``attack_cat``, the binary target is stratified
as a side effect.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from src import config, data_loader
from src.features import ENGINEERED_FEATURE_DOCS, NetworkFeatureEngineer

# --------------------------------------------------------------------------- #
# Column groups (defined on the frame AFTER feature engineering)
# --------------------------------------------------------------------------- #

#: Strictly non-negative, heavy-tailed counters. UNSW-NB15 spans nine orders of
#: magnitude on several of these (``sload`` ranges 0 - 5.99e9), which would make
#: a standard-scaled linear model numerically hopeless and would let a handful
#: of extreme flows dominate the scaler's variance estimate. ``log1p`` is applied
#: first: it is monotonic (so tree models are unaffected), defined at zero (so
#: the 120,288 zero-byte reverse directions survive), and reversible.
LOG1P_FEATURES: tuple[str, ...] = (
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sload", "dload", "sloss", "dloss", "sinpkt", "dinpkt",
    "sjit", "djit", "stcpb", "dtcpb", "tcprtt", "synack", "ackdat",
    "smean", "dmean", "trans_depth", "response_body_len",
    "flow_bytes_total", "flow_pkts_total", "bytes_per_packet",
)

#: Bounded or already-signed numerics: scaled but not log-transformed.
#: (TTL and TCP window fields are bounded to 0-255; the ``ct_*`` connection
#: counters are bounded to ~65; the ratio features are bounded to [0, 1]; the
#: ``*_log_ratio`` features are already on a log scale.)
LINEAR_FEATURES: tuple[str, ...] = (
    "sttl", "dttl", "swin", "dwin",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst",
    "src_byte_ratio", "src_pkt_ratio", "load_log_ratio", "jit_log_ratio",
    "src_loss_rate", "dst_loss_rate",
)

#: 0/1 indicators: passed through untouched. Scaling a Boolean buys nothing and
#: makes SHAP values harder to read ("+0.4 when the flag is on" is clearer than
#: "+0.4 when the standardised flag is 2.7").
FLAG_FEATURES: tuple[str, ...] = (
    "is_sm_ips_ports",
    "is_one_way", "tcp_handshake_complete", "tcp_seq_exchanged",
    "both_win_advertised", "is_zero_duration", "service_unknown",
)


def feature_columns(
    include_engineered: bool = True,
    exclude: tuple[str, ...] = (),
) -> dict[str, list[str]]:
    """
    Return the column groups the ``ColumnTransformer`` will consume.

    Parameters
    ----------
    include_engineered:
        When ``False``, the columns produced by :class:`NetworkFeatureEngineer`
        are omitted - used by the "no feature engineering" ablation.
    exclude:
        Columns to drop entirely - used by the "no TTL features" ablation.
    """
    engineered = set(ENGINEERED_FEATURE_DOCS)
    drop = set(exclude)

    def keep(columns: tuple[str, ...]) -> list[str]:
        return [
            c for c in columns
            if c not in drop and (include_engineered or c not in engineered)
        ]

    return {
        "log1p": keep(LOG1P_FEATURES),
        "linear": keep(LINEAR_FEATURES),
        "flags": keep(FLAG_FEATURES),
        "categorical": [c for c in config.CATEGORICAL_FEATURES if c not in drop],
    }


# --------------------------------------------------------------------------- #
# Corpus preparation and splitting
# --------------------------------------------------------------------------- #
def prepare_corpus(
    deduplicate: bool | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """
    Load UNSW-NB15 and apply row-level cleaning, returning ``(corpus, report)``.

    The only row-level operation is deduplication, and it is applied to the
    *whole corpus before splitting* - which is the entire point. Deduplicating
    after a split would leave identical feature vectors straddling train and
    test, which is precisely the optimism this guards against.

    Outliers are deliberately NOT removed. In network telemetry an extreme value
    is usually the attack: a 14 MB transfer, a 5.99 Gbit/s load and a 10,646
    packet burst are all genuine records of genuine malicious behaviour. The
    only values treated as erroneous are non-finite ones, of which the dataset
    contains none. See reports/EDA_Feature_Engineering_Report.md.
    """
    if deduplicate is None:
        deduplicate = config.DEDUPLICATE

    corpus = data_loader.load_corpus()
    report: dict[str, object] = {"rows_loaded": int(len(corpus))}

    predictors = [
        c for c in corpus.columns
        if c not in (*config.LEAKAGE_COLUMNS, "partition")
    ]
    report["n_predictor_columns"] = len(predictors)

    # How many distinct feature vectors carry contradictory labels? This is an
    # irreducible noise floor: no model can be right about both copies.
    conflicting = corpus.groupby(predictors, dropna=False, observed=True)[config.TARGET].nunique()
    report["conflicting_feature_vectors"] = int((conflicting > 1).sum())

    duplicate_mask = corpus.duplicated(subset=predictors, keep="first")
    report["duplicate_rows"] = int(duplicate_mask.sum())
    report["duplicate_fraction"] = float(duplicate_mask.mean())
    report["duplicate_rate_by_attack_cat"] = (
        corpus.assign(_dup=duplicate_mask)
        .groupby(config.ATTACK_CAT, observed=True)["_dup"]
        .mean()
        .round(4)
        .to_dict()
    )

    if deduplicate:
        corpus = corpus.loc[~duplicate_mask].reset_index(drop=True)

    report["deduplicated"] = bool(deduplicate)
    report["rows_final"] = int(len(corpus))
    report["class_balance"] = (
        corpus[config.TARGET].value_counts(normalize=True).sort_index().round(4).to_dict()
    )
    report["attack_cat_counts"] = corpus[config.ATTACK_CAT].value_counts().to_dict()
    report["missing_values"] = int(corpus.isna().sum().sum())

    if verbose:
        print(f"[preprocessing] loaded {report['rows_loaded']:,} rows")
        print(f"[preprocessing] duplicate feature vectors: {report['duplicate_rows']:,} "
              f"({report['duplicate_fraction']:.1%})")
        print(f"[preprocessing] contradictory feature vectors: "
              f"{report['conflicting_feature_vectors']:,}")
        print(f"[preprocessing] corpus after dedup={deduplicate}: {report['rows_final']:,} rows")
        print(f"[preprocessing] class balance: {report['class_balance']}")

    return corpus, report


def split_corpus(
    corpus: pd.DataFrame,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 60/20/20 train/validation/test split (``random_state=42``)."""
    strata = corpus[config.STRATIFY_COLUMN]

    train_val, test = train_test_split(
        corpus,
        test_size=config.TEST_SIZE,
        stratify=strata,
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    # VAL_SIZE is expressed as a fraction of the FULL corpus, so convert it to a
    # fraction of the 80% remainder: 0.20 / 0.80 = 0.25.
    val_fraction = config.VAL_SIZE / (1.0 - config.TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[config.STRATIFY_COLUMN],
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )

    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    if verbose:
        total = len(corpus)
        for name, part in (("train", train), ("val", val), ("test", test)):
            rate = part[config.TARGET].mean()
            print(f"[preprocessing] {name:<5} {len(part):>7,} rows "
                  f"({len(part) / total:5.1%})  attack rate={rate:.4f}")
    return train, val, test


def split_official(
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Reproduce the dataset authors' published train/test partition instead.

    Used only by the ``official_split_raw`` robustness check. The published test file
    is held out untouched; a validation set is carved out of the published
    training file so threshold selection still never sees the test data.
    """
    corpus = data_loader.load_corpus()
    train_val = corpus[corpus["partition"] == "official_train"].reset_index(drop=True)
    test = corpus[corpus["partition"] == "official_test"].reset_index(drop=True)
    train, val = train_test_split(
        train_val,
        test_size=0.20,
        stratify=train_val[config.STRATIFY_COLUMN],
        random_state=config.RANDOM_STATE,
    )
    if verbose:
        print(f"[preprocessing] official protocol -> train={len(train):,} "
              f"val={len(val):,} test={len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test


def split_published(
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """
    The primary protocol: the authors' published partition, leak-proofed.

    This is deliberately the harder and more externally valid evaluation, and it
    is what the headline numbers are measured on. Three things happen, in order:

    1. **Deduplicate each partition independently.** The published train and test
       files each repeat feature vectors internally. Collapsing them separately
       removes that inflation without dissolving the partition boundary.
    2. **Carve a validation set out of the published training file** (80/20,
       stratified on ``attack_cat``). Model selection and threshold tuning see
       only this; the published test file is never touched by either.
    3. **Remove from the test set any feature vector that also occurs in train
       or validation.** Deduplicating within a partition does not catch a record
       that appears in both, and such a row is memorised, not predicted.

    Contrast with :func:`split_corpus`, which pools both files and draws a random
    split. A random split guarantees train and test share a distribution; the
    published partition does not, which is why performance drops on it. That drop
    is information, not a defect, so this protocol is the primary one and the
    pooled random split is carried as the ``pooled_random`` ablation.

    Returns ``(train, val, test, manifest)``.
    """
    corpus = data_loader.load_corpus()
    predictors = [c for c in corpus.columns
                  if c not in (*config.LEAKAGE_COLUMNS, "partition")]

    published_train = corpus[corpus["partition"] == "official_train"]
    published_test = corpus[corpus["partition"] == "official_test"]
    manifest: dict[str, object] = {
        "protocol": "published_partition",
        "published_train_rows": int(len(published_train)),
        "published_test_rows": int(len(published_test)),
    }

    def dedupe(frame: pd.DataFrame, tag: str) -> pd.DataFrame:
        conflicting = frame.groupby(predictors, dropna=False, observed=True)[
            config.TARGET].nunique()
        manifest[f"{tag}_conflicting_signatures"] = int((conflicting > 1).sum())
        mask = frame.duplicated(subset=predictors, keep="first")
        manifest[f"{tag}_duplicates_removed"] = int(mask.sum())
        return frame.loc[~mask].reset_index(drop=True)

    development = dedupe(published_train, "development")
    test = dedupe(published_test, "test")

    train, val = train_test_split(
        development,
        test_size=config.VAL_SIZE,
        stratify=development[config.STRATIFY_COLUMN],
        random_state=config.RANDOM_STATE,
        shuffle=True,
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)

    # Step 3: drop test rows whose feature vector was seen during development.
    seen = set(map(tuple, development[predictors].itertuples(index=False, name=None)))
    overlap = np.fromiter(
        (row in seen for row in test[predictors].itertuples(index=False, name=None)),
        dtype=bool, count=len(test))
    manifest["test_overlap_removed"] = int(overlap.sum())
    test = test.loc[~overlap].reset_index(drop=True)

    manifest.update({
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_attack_rate": round(float(train[config.TARGET].mean()), 4),
        "validation_attack_rate": round(float(val[config.TARGET].mean()), 4),
        "test_attack_rate": round(float(test[config.TARGET].mean()), 4),
        "random_state": config.RANDOM_STATE,
    })

    if verbose:
        print(f"[preprocessing] published protocol: "
              f"train {len(train):,} / val {len(val):,} / test {len(test):,}")
        print(f"[preprocessing]   duplicates removed - development "
              f"{manifest['development_duplicates_removed']:,}, test "
              f"{manifest['test_duplicates_removed']:,}")
        print(f"[preprocessing]   train/test overlap removed: "
              f"{manifest['test_overlap_removed']:,}")
        print(f"[preprocessing]   attack rate - train {manifest['train_attack_rate']:.4f}, "
              f"test {manifest['test_attack_rate']:.4f}")
    return train, val, test, manifest


def split_xy(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictors from the target, dropping every leakage column.

    ``id``, ``attack_cat``, ``label`` and the bookkeeping ``partition`` column
    are removed here and nowhere else, so there is a single enforcement point.
    """
    drop = [c for c in (*config.LEAKAGE_COLUMNS, "partition") if c in frame.columns]
    X = frame.drop(columns=drop)
    y = frame[config.TARGET].astype(int)
    return X, y


# --------------------------------------------------------------------------- #
# Preprocessing pipeline
# --------------------------------------------------------------------------- #
def safe_log1p(values):
    """``log1p`` with a non-negativity clip.

    Defined at module level, NOT as a lambda: the fitted pipeline is persisted
    with ``joblib``, and a closure cannot be pickled. Keeping these as named
    module functions is what makes the saved model loadable in a fresh process
    (and therefore servable at all). ``tests/test_prediction.py`` asserts the
    pipeline round-trips through pickle for exactly this reason.
    """
    return np.log1p(np.clip(np.asarray(values, dtype=float), 0.0, None))


def safe_expm1(values):
    """Inverse of :func:`safe_log1p`, for inspecting a transformed value."""
    return np.expm1(np.asarray(values, dtype=float))


def _log1p_transformer() -> FunctionTransformer:
    """``log1p`` with a non-negativity clip, safe for inverse inspection."""
    return FunctionTransformer(
        func=safe_log1p,
        inverse_func=safe_expm1,
        feature_names_out="one-to-one",
        validate=False,
    )


def build_preprocessor(
    include_engineered: bool = True,
    exclude: tuple[str, ...] = (),
) -> ColumnTransformer:
    """
    Build the ``ColumnTransformer`` applied after feature engineering.

    * heavy-tailed counters -> ``log1p`` -> ``StandardScaler``
    * bounded numerics      -> ``StandardScaler``
    * 0/1 flags             -> passthrough
    * nominal columns       -> ``OneHotEncoder`` with an infrequent-category
      bucket, ``handle_unknown="infrequent_if_exist"``

    The one-hot encoder's ``min_frequency`` is what makes the pipeline safe
    against protocols that appear in production but never appeared in training:
    they are routed to the ``infrequent_sklearn`` column rather than raising.
    """
    groups = feature_columns(include_engineered=include_engineered, exclude=exclude)

    log_branch = Pipeline([
        ("log1p", _log1p_transformer()),
        ("scale", StandardScaler()),
    ])

    categorical = OneHotEncoder(
        handle_unknown="infrequent_if_exist",
        min_frequency=config.MIN_CATEGORY_FREQUENCY,
        sparse_output=False,
        dtype=np.float64,
    )

    transformers = [
        ("log", log_branch, groups["log1p"]),
        ("num", StandardScaler(), groups["linear"]),
        ("flag", "passthrough", groups["flags"]),
        ("cat", categorical, groups["categorical"]),
    ]
    # Drop empty branches so get_feature_names_out stays clean under ablations.
    transformers = [t for t in transformers if len(t[2]) > 0]

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",          # anything not listed is intentionally unused
        verbose_feature_names_out=False,
        n_jobs=None,
    )


def build_feature_pipeline(
    include_engineered: bool = True,
    exclude: tuple[str, ...] = (),
) -> Pipeline:
    """Feature engineering followed by preprocessing, as one fittable object."""
    return Pipeline([
        ("engineer", NetworkFeatureEngineer(enabled=include_engineered)),
        ("preprocess", build_preprocessor(
            include_engineered=include_engineered, exclude=exclude)),
    ])


def build_model_pipeline(
    estimator: BaseEstimator,
    include_engineered: bool = True,
    exclude: tuple[str, ...] = (),
) -> Pipeline:
    """Full end-to-end pipeline: engineer -> preprocess -> estimator."""
    return Pipeline([
        ("engineer", NetworkFeatureEngineer(enabled=include_engineered)),
        ("preprocess", build_preprocessor(
            include_engineered=include_engineered, exclude=exclude)),
        ("model", estimator),
    ])


def transformed_feature_names(fitted_pipeline: Pipeline) -> list[str]:
    """Return the post-transform feature names of a fitted pipeline."""
    return list(fitted_pipeline.named_steps["preprocess"].get_feature_names_out())


# --------------------------------------------------------------------------- #
# CLI: materialise the splits so notebooks and tests share identical data
# --------------------------------------------------------------------------- #
def main() -> int:
    config.ensure_dirs()
    # The PRIMARY protocol, so the materialised parquet files match the models
    # in models/ rather than a different partition of the same data.
    train, val, test, manifest = split_published(verbose=True)

    for name, part in (("train", train), ("val", val), ("test", test)):
        path = config.PROCESSED_DIR / f"{name}.parquet"
        part.to_parquet(path, index=False)
        print(f"[preprocessing] wrote {path.relative_to(config.PROJECT_ROOT)} "
              f"({len(part):,} rows)")

    # Committed provenance: what the split did, in numbers, without needing the
    # 62 MB of raw data to inspect it.
    manifest_path = config.METRICS_DIR / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[preprocessing] wrote reports/metrics/{manifest_path.name}")

    pipeline = build_feature_pipeline()
    X_train, _ = split_xy(train)
    pipeline.fit(X_train)
    names = transformed_feature_names(pipeline)
    print(f"[preprocessing] design matrix width after encoding: {len(names)}")
    print(f"[preprocessing] example columns: {names[:6]} ... {names[-6:]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
