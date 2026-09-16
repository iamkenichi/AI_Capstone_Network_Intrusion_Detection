"""
Inference API: the single entry point used by the tests, the CLI and the
Streamlit application.

Everything a caller needs is here - loading the frozen model, validating that an
incoming record has the columns the pipeline expects, scoring it, and turning a
probability into a risk band a human can act on. Keeping this in one module
means the demo application and the test suite exercise exactly the same code
path that a batch scoring job would.

The persisted artefact is a complete ``Pipeline``: feature engineering,
preprocessing and the estimator travel together. A caller passes raw
UNSW-NB15-shaped flow records and never has to reproduce any transformation,
which removes the most common cause of training/serving skew.

Usage
-----
    python -m src.predict --input flows.csv --output scored.csv
    python -m src.predict --demo
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src import config

#: Model loaded by default. Written by ``src.pipeline`` once the winning model
#: and its operating threshold have been chosen on the validation split.
DEPLOYMENT_MANIFEST = config.MODELS_DIR / "deployment.json"

#: Raw flow columns the pipeline consumes (everything except id/target/bookkeeping).
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = (
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes",
    "rate", "sttl", "dttl", "sload", "dload", "sloss", "dloss", "sinpkt",
    "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len",
    "ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login", "ct_ftp_cmd",
    "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
)

#: Probability bands used to turn a score into an operational disposition.
#: The boundaries are anchored on the deployed threshold, not on round numbers:
#: anything below it is not alerted at all, and the bands above it split the
#: alerting region into "queue it" and "page someone".
RISK_BANDS: tuple[tuple[float, str, str], ...] = (
    (0.90, "Critical", "Escalate immediately; isolate the host and begin incident response."),
    (0.70, "High", "Raise a priority alert for analyst triage within the current shift."),
    (0.40, "Medium", "Queue for batch review; correlate with other telemetry before acting."),
    (0.00, "Low", "No action. Retain for retrospective hunting and drift monitoring."),
)


class ModelNotAvailable(FileNotFoundError):
    """Raised when no trained model is present on disk."""


@dataclass
class LoadedModel:
    """A frozen pipeline plus the metadata needed to interpret its output."""

    pipeline: Any
    name: str
    threshold: float
    metadata: dict[str, Any]

    @property
    def display_name(self) -> str:
        return self.metadata.get("display_name", self.name)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def available_models() -> list[str]:
    """Names of every persisted model in ``models/``."""
    return sorted(p.stem for p in config.MODELS_DIR.glob("*.joblib"))


def load_model(name: str | None = None) -> LoadedModel:
    """
    Load the deployment model, or a specific one by name.

    With no argument this reads ``models/deployment.json``, which records the
    model chosen on the validation split and the threshold chosen with it -
    so the served configuration is the evaluated configuration, by construction
    rather than by anyone remembering to keep two numbers in sync.
    """
    metadata: dict[str, Any] = {}
    threshold = config.DEFAULT_THRESHOLD

    if name is None:
        if not DEPLOYMENT_MANIFEST.exists():
            raise ModelNotAvailable(
                f"No deployment manifest at {DEPLOYMENT_MANIFEST}.\n"
                "Train and evaluate the project first:\n"
                "    python -m src.pipeline --all\n"
                f"Models currently on disk: {available_models() or 'none'}"
            )
        metadata = json.loads(DEPLOYMENT_MANIFEST.read_text(encoding="utf-8"))
        name = metadata["model_key"]
        threshold = float(metadata.get("threshold", config.DEFAULT_THRESHOLD))

    path = config.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise ModelNotAvailable(
            f"Model file not found: {path}\n"
            f"Available models: {available_models() or 'none'}\n"
            "Run `python -m src.train` to produce them."
        )
    return LoadedModel(joblib.load(path), name, threshold, metadata)


def feature_reference() -> dict[str, Any]:
    """
    Typical values and category vocabularies for every input column.

    Used by the Streamlit app to pre-populate its form with a realistic flow
    rather than zeros, and by the tests to build valid synthetic records. Built
    from the TRAINING split only - the app must never be seeded from test data.
    """
    path = config.MODELS_DIR / "feature_reference.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    train_path = config.PROCESSED_DIR / "train.parquet"
    if not train_path.exists():
        raise ModelNotAvailable(
            f"Neither {path} nor {train_path} exists. Run `python -m src.preprocessing` first."
        )
    train = pd.read_parquet(train_path)
    reference = build_feature_reference(train)
    path.write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return reference


def build_feature_reference(train: pd.DataFrame) -> dict[str, Any]:
    """Summarise the training split into per-column defaults and ranges."""
    numeric: dict[str, Any] = {}
    categorical: dict[str, Any] = {}
    for column in REQUIRED_INPUT_COLUMNS:
        series = train[column]
        if series.dtype == object:
            counts = series.value_counts()
            categorical[column] = {
                "options": counts.index.tolist(),
                "default": counts.index[0],
                "counts": counts.to_dict(),
            }
        else:
            numeric[column] = {
                "median": float(series.median()),
                "mean": float(series.mean()),
                "min": float(series.min()),
                "max": float(series.max()),
                "p01": float(series.quantile(0.01)),
                "p99": float(series.quantile(0.99)),
                "is_integer": bool(pd.api.types.is_integer_dtype(series)),
            }

    # One representative record per class, for the app's "load an example" control.
    examples: dict[str, dict] = {}
    for label, key in ((0, "typical_benign"), (1, "typical_attack")):
        subset = train[train[config.TARGET] == label]
        if not subset.empty:
            examples[key] = _representative_row(subset)
    for family in ("Exploits", "Reconnaissance", "DoS", "Generic", "Fuzzers"):
        subset = train[train[config.ATTACK_CAT] == family]
        if len(subset) >= 30:
            examples[f"typical_{family.lower()}"] = _representative_row(subset)

    return {"numeric": numeric, "categorical": categorical, "examples": examples,
            "n_training_rows": int(len(train))}


def _representative_row(subset: pd.DataFrame) -> dict[str, Any]:
    """
    Pick the real record closest to the subset's median profile.

    A column-wise median is not a valid flow - it can pair a completed TCP
    handshake with zero destination packets. Selecting the nearest ACTUAL record
    guarantees the example the app shows is a coherent one that really occurred.
    """
    numeric_cols = [c for c in REQUIRED_INPUT_COLUMNS
                    if subset[c].dtype != object]
    values = subset[numeric_cols].to_numpy(float)
    logged = np.log1p(np.clip(values, 0, None))
    centre = np.median(logged, axis=0)
    spread = np.where(logged.std(axis=0) > 0, logged.std(axis=0), 1.0)
    distance = np.linalg.norm((logged - centre) / spread, axis=1)
    row = subset.iloc[int(np.argmin(distance))]
    return {c: (row[c].item() if hasattr(row[c], "item") else row[c])
            for c in REQUIRED_INPUT_COLUMNS}


# --------------------------------------------------------------------------- #
# Validation and scoring
# --------------------------------------------------------------------------- #
def validate_input(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Check an incoming frame and return it with columns in pipeline order.

    Raises with an explicit list of what is missing rather than letting the
    failure surface deep inside a ColumnTransformer, where the message would be
    far harder to act on.
    """
    if frame.empty:
        raise ValueError("No flow records supplied: the input frame has zero rows.")

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            f"Input is missing {len(missing)} required flow column(s): {missing}\n"
            "Expected UNSW-NB15-shaped records. Columns received: "
            f"{sorted(frame.columns.tolist())}"
        )

    out = frame.loc[:, list(REQUIRED_INPUT_COLUMNS)].copy()
    for column in config.CATEGORICAL_FEATURES:
        out[column] = out[column].astype(str).str.strip().str.lower()
    for column in out.columns:
        if column not in config.CATEGORICAL_FEATURES:
            coerced = pd.to_numeric(out[column], errors="coerce")
            if coerced.isna().any():
                bad = out.loc[coerced.isna(), column].unique()[:5]
                raise ValueError(
                    f"Column '{column}' must be numeric but contains non-numeric "
                    f"values, e.g. {list(bad)}"
                )
            out[column] = coerced
    return out


def risk_band(probability: float) -> tuple[str, str]:
    """Map a probability to ``(band, recommended action)``."""
    for floor, band, action in RISK_BANDS:
        if probability >= floor:
            return band, action
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def predict(
    frame: pd.DataFrame,
    model: LoadedModel | None = None,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    Score flow records.

    Returns one row per input record with the attack probability, the binary
    verdict at the operating threshold, a risk band and a recommended action.
    """
    model = model or load_model()
    threshold = float(threshold if threshold is not None else model.threshold)
    X = validate_input(frame)

    if hasattr(model.pipeline, "predict_proba"):
        probability = model.pipeline.predict_proba(X)[:, 1]
        score_kind = "probability"
    else:
        raw = -model.pipeline.score_samples(X)
        span = raw.max() - raw.min()
        probability = (raw - raw.min()) / span if span > 0 else np.zeros_like(raw)
        score_kind = "normalised anomaly score (NOT a probability)"

    prediction = (probability >= threshold).astype(int)
    bands = [risk_band(p) for p in probability]

    return pd.DataFrame({
        "attack_probability": np.round(probability, 6),
        "prediction": prediction,
        "verdict": np.where(prediction == 1, "ATTACK", "BENIGN"),
        "risk_band": [b for b, _ in bands],
        "recommended_action": [a for _, a in bands],
        "threshold": threshold,
        "model": model.display_name,
        "score_kind": score_kind,
    }, index=frame.index)


def predict_one(record: dict[str, Any], model: LoadedModel | None = None,
                threshold: float | None = None) -> dict[str, Any]:
    """Score a single flow supplied as a dict. Convenience wrapper for the app."""
    result = predict(pd.DataFrame([record]), model=model, threshold=threshold)
    return result.iloc[0].to_dict()


def explain_one(record: dict[str, Any], model: LoadedModel | None = None,
                top_n: int = 8) -> pd.DataFrame | None:
    """
    Per-feature SHAP contributions for a single flow, or ``None`` if the model
    is not a supported tree ensemble.

    Returned as a tidy frame of ``feature``, ``value``, ``shap_value``, sorted by
    absolute contribution, so the caller can render it without knowing anything
    about SHAP's internals.
    """
    model = model or load_model()
    try:
        import shap
    except ImportError:
        return None

    estimator = model.pipeline.named_steps.get("model")
    if estimator is None or not hasattr(estimator, "feature_importances_"):
        return None

    from src.preprocessing import transformed_feature_names

    X = validate_input(pd.DataFrame([record]))
    engineered = model.pipeline.named_steps["engineer"].transform(X)
    design = model.pipeline.named_steps["preprocess"].transform(engineered)
    names = transformed_feature_names(model.pipeline)

    try:
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(design)
    except Exception:  # noqa: BLE001 - explanation is best-effort in the app
        return None

    values = np.asarray(values)
    if values.ndim == 3:            # (n_samples, n_features, n_classes)
        values = values[..., -1]
    contributions = np.asarray(values).reshape(-1)[: len(names)]

    frame = pd.DataFrame({
        "feature": names,
        "value": np.asarray(design).reshape(-1)[: len(names)],
        "shap_value": contributions,
    })
    frame["abs_shap"] = frame["shap_value"].abs()
    frame["direction"] = np.where(frame["shap_value"] >= 0, "toward ATTACK", "toward BENIGN")
    return (frame.sort_values("abs_shap", ascending=False)
            .head(top_n).drop(columns="abs_shap").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score network flows for intrusion.")
    parser.add_argument("--input", type=Path, help="CSV of UNSW-NB15-shaped flow records")
    parser.add_argument("--output", type=Path, help="where to write the scored CSV")
    parser.add_argument("--model", default=None, help="model name (default: deployment model)")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--demo", action="store_true",
                        help="score the built-in example flows and print the result")
    args = parser.parse_args(argv)

    try:
        model = load_model(args.model)
    except ModelNotAvailable as exc:
        print(exc)
        return 2

    if args.demo:
        reference = feature_reference()
        examples = reference["examples"]
        frame = pd.DataFrame(list(examples.values()), index=list(examples))
        scored = predict(frame, model=model, threshold=args.threshold)
        print(f"\nModel: {model.display_name}   threshold: {scored['threshold'].iloc[0]:.2f}\n")
        print(scored[["attack_probability", "verdict", "risk_band"]].to_string())
        return 0

    if not args.input:
        parser.error("supply --input PATH or --demo")
    if not args.input.exists():
        print(f"Input file not found: {args.input}")
        return 2

    frame = pd.read_csv(args.input, encoding="utf-8-sig")
    scored = predict(frame, model=model, threshold=args.threshold)
    combined = pd.concat([frame.reset_index(drop=True),
                          scored.reset_index(drop=True)], axis=1)
    if args.output:
        combined.to_csv(args.output, index=False)
        print(f"Wrote {len(combined):,} scored flows to {args.output}")
    else:
        print(scored.to_string())
    alerts = int(scored["prediction"].sum())
    print(f"\n{alerts:,} of {len(scored):,} flows flagged as attacks "
          f"({alerts / len(scored):.2%}) at threshold {scored['threshold'].iloc[0]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
