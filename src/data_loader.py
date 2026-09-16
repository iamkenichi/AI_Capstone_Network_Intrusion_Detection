"""
Dataset acquisition, integrity verification and loading for UNSW-NB15.

Responsibilities
----------------
1. Download the two official partitioned CSV files into ``data/raw/`` if they
   are not already present (``download_dataset``).
2. Verify that what landed on disk is really UNSW-NB15: expected row counts,
   column count, schema and target encoding (``verify_dataset``).
3. Load the partitions, optionally concatenated into a single analysis corpus,
   with dtypes normalised (``load_raw``, ``load_corpus``).

Design notes
------------
* The loader never silently substitutes a different dataset. If verification
  fails it raises with an actionable message telling the user exactly which
  files to obtain and where to place them.
* ``ct_ftp_cmd`` ships as a string column in the official CSV because a subset
  of rows carry a whitespace token instead of ``0``. It is coerced to a numeric
  dtype here (not in preprocessing) because this is a *file format* defect, not
  a modelling decision.

Run as a script::

    python -m src.data_loader            # download + verify + summarise
    python -m src.data_loader --force    # re-download even if files exist
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# Manual-download instructions, surfaced whenever automatic retrieval fails.
# --------------------------------------------------------------------------- #
MANUAL_INSTRUCTIONS = f"""
------------------------------------------------------------------------------
MANUAL DATASET DOWNLOAD REQUIRED
------------------------------------------------------------------------------
Automatic retrieval of UNSW-NB15 failed (no network access, or every mirror is
unreachable). Please obtain the two OFFICIAL partitioned CSV files yourself.

FILES NEEDED
  1. {config.TRAIN_FILE}   ({config.EXPECTED_ROWS[config.TRAIN_FILE]:,} data rows, 45 columns)
  2. {config.TEST_FILE}    ({config.EXPECTED_ROWS[config.TEST_FILE]:,} data rows, 45 columns)

WHERE TO GET THEM (any one of these)
  a) Official project page - UNSW Canberra, Australian Centre for Cyber Security
     https://research.unsw.edu.au/projects/unsw-nb15-dataset
     Follow the "CSV Files" link, then the folder
        "a part of training and testing set"
  b) IEEE DataPort  -  DOI 10.21227/8vf7-s525
     https://ieee-dataport.org/documents/unswnb15-dataset
  c) Public mirror  -  https://huggingface.co/datasets/Mouwiya/UNSW-NB15
  d) Kaggle mirror  -  https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

WHERE TO PUT THEM
  Copy both files, keeping their exact filenames, into:

      {config.RAW_DIR}

  The directory should then contain:
      data/raw/{config.TRAIN_FILE}
      data/raw/{config.TEST_FILE}

THEN RE-RUN
      python -m src.data_loader
------------------------------------------------------------------------------
""".strip()


class DatasetNotAvailable(RuntimeError):
    """Raised when UNSW-NB15 cannot be obtained or fails verification."""


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def _download_file(url: str, destination: Path, timeout: int = 180) -> None:
    """Stream ``url`` to ``destination``, writing to a temp file first."""
    import requests  # imported lazily: only needed when a download is required

    tmp = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout,
                      headers={"User-Agent": "unsw-nb15-capstone/1.0"}) as response:
        response.raise_for_status()
        with tmp.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)
    tmp.replace(destination)


def download_dataset(force: bool = False, verbose: bool = True) -> dict[str, Path]:
    """
    Ensure both partitioned CSVs exist in ``data/raw/``.

    Tries each configured mirror in order. Returns a mapping of filename -> path.
    Raises :class:`DatasetNotAvailable` (with manual instructions) on failure.
    """
    config.ensure_dirs()
    wanted = [config.TRAIN_FILE, config.TEST_FILE]
    resolved: dict[str, Path] = {}
    still_missing: list[str] = []

    for filename in wanted:
        destination = config.RAW_DIR / filename
        if destination.exists() and not force:
            if verbose:
                size_mb = destination.stat().st_size / 1e6
                print(f"[data_loader] present    {filename}  ({size_mb:,.1f} MB)")
            resolved[filename] = destination
        else:
            still_missing.append(filename)

    for filename in still_missing:
        destination = config.RAW_DIR / filename
        for mirror in config.DOWNLOAD_MIRRORS:
            url = mirror.get(filename)
            if not url:
                continue
            try:
                if verbose:
                    print(f"[data_loader] downloading {filename} from {mirror['name']} ...")
                _download_file(url, destination)
                if verbose:
                    size_mb = destination.stat().st_size / 1e6
                    print(f"[data_loader] saved      {filename}  ({size_mb:,.1f} MB)")
                resolved[filename] = destination
                break
            except Exception as exc:  # noqa: BLE001 - we want to try the next mirror
                if verbose:
                    print(f"[data_loader]   mirror failed: {type(exc).__name__}: {exc}")

    missing = [f for f in wanted if f not in resolved]
    if missing:
        raise DatasetNotAvailable(
            f"Could not obtain: {', '.join(missing)}\n\n{MANUAL_INSTRUCTIONS}"
        )

    # Best-effort: the authors' own feature description file enriches the data
    # dictionary. Its absence is not fatal.
    feature_file = config.RAW_DIR / config.FEATURE_DESCRIPTION_FILE
    if not feature_file.exists():
        try:
            _download_file(config.FEATURE_DESCRIPTION_URL, feature_file)
            if verbose:
                print(f"[data_loader] saved      {config.FEATURE_DESCRIPTION_FILE} (optional)")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[data_loader] optional feature-description file unavailable ({exc})")

    return resolved


def file_checksums() -> dict[str, str]:
    """SHA-256 of each raw file, recorded in the dataset documentation."""
    out: dict[str, str] = {}
    for filename in (config.TRAIN_FILE, config.TEST_FILE):
        path = config.RAW_DIR / filename
        if not path.exists():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        out[filename] = digest.hexdigest()
    return out


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def _read_partition(path: Path) -> pd.DataFrame:
    """Read one partitioned CSV and repair known file-format defects."""
    if not path.exists():
        raise DatasetNotAvailable(f"Missing raw file: {path}\n\n{MANUAL_INSTRUCTIONS}")

    # utf-8-sig strips the BOM the published files carry on the `id` header.
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    # `ct_ftp_cmd` is published as text: rows with no FTP command carry a
    # whitespace token rather than 0. Coerce to a nullable-free integer view.
    if "ct_ftp_cmd" in frame.columns and frame["ct_ftp_cmd"].dtype == object:
        frame["ct_ftp_cmd"] = (
            pd.to_numeric(frame["ct_ftp_cmd"].astype(str).str.strip(), errors="coerce")
            .fillna(0)
            .astype("int64")
        )

    # Normalise the nominal columns: the published files mix case and padding.
    for column in config.CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).str.strip().str.lower()

    if config.ATTACK_CAT in frame.columns:
        # The testing partition writes "Backdoors" where the training partition
        # writes "Backdoor"; several rows also carry stray padding.
        cat = frame[config.ATTACK_CAT].astype(str).str.strip()
        cat = cat.replace({"Backdoors": "Backdoor", "backdoors": "Backdoor"})
        frame[config.ATTACK_CAT] = cat

    return frame


def load_raw(verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(official_train, official_test)`` exactly as published."""
    train = _read_partition(config.RAW_DIR / config.TRAIN_FILE)
    test = _read_partition(config.RAW_DIR / config.TEST_FILE)
    if verbose:
        print(f"[data_loader] official train partition: {train.shape}")
        print(f"[data_loader] official test  partition: {test.shape}")
    return train, test


def load_corpus(verbose: bool = False) -> pd.DataFrame:
    """
    Return the two official partitions concatenated into a single corpus.

    Why recombine?
    --------------
    The published "training"/"testing" files are themselves a stratified random
    partition of one capture, not a temporal holdout; they share the same
    generating distribution. This project therefore treats their union as the
    analysis corpus and performs its own reproducible, stratified
    train/validation/test split (see :mod:`src.preprocessing`). That gives a
    genuine validation set for threshold tuning without ever touching the test
    set, which the published two-way split cannot provide.

    A ``partition`` column records each record's original file so the published
    protocol can still be reproduced as a robustness check.
    """
    train, test = load_raw(verbose=verbose)
    train = train.assign(partition="official_train")
    test = test.assign(partition="official_test")
    corpus = pd.concat([train, test], ignore_index=True)
    if verbose:
        print(f"[data_loader] combined corpus: {corpus.shape}")
    return corpus


# --------------------------------------------------------------------------- #
# Verify
# --------------------------------------------------------------------------- #
def verify_dataset(strict: bool = True, verbose: bool = True) -> dict[str, object]:
    """
    Assert that the files on disk really are the official UNSW-NB15 partitions.

    Checks row counts, column count, presence of target columns, the binary
    encoding of ``label``, and the ``label``/``attack_cat`` consistency rule.
    Returns a dict of findings; raises on failure when ``strict``.
    """
    findings: dict[str, object] = {}
    problems: list[str] = []

    train, test = load_raw()
    findings["train_shape"] = train.shape
    findings["test_shape"] = test.shape

    for name, frame in ((config.TRAIN_FILE, train), (config.TEST_FILE, test)):
        expected = config.EXPECTED_ROWS[name]
        if len(frame) != expected:
            problems.append(f"{name}: expected {expected:,} rows, found {len(frame):,}")
        if frame.shape[1] != config.EXPECTED_N_COLUMNS:
            problems.append(
                f"{name}: expected {config.EXPECTED_N_COLUMNS} columns, "
                f"found {frame.shape[1]}"
            )

    for column in (config.TARGET, config.ATTACK_CAT):
        if column not in train.columns:
            problems.append(f"required column '{column}' absent from {config.TRAIN_FILE}")

    if config.TARGET in train.columns:
        values = {int(v) for v in pd.unique(train[config.TARGET])}
        findings["label_values"] = sorted(values)
        if not values <= {0, 1}:
            problems.append(f"'{config.TARGET}' is not binary 0/1; found {sorted(values)}")

    # The defining leakage relationship: attack_cat == "Normal" <=> label == 0.
    if {config.TARGET, config.ATTACK_CAT} <= set(train.columns):
        corpus = pd.concat([train, test], ignore_index=True)
        is_normal = corpus[config.ATTACK_CAT].str.lower().eq("normal")
        agreement = (is_normal == corpus[config.TARGET].eq(0)).mean()
        findings["label_attack_cat_agreement"] = float(agreement)
        if agreement < 0.999:
            problems.append(
                "attack_cat does not map deterministically onto label "
                f"(agreement={agreement:.4f}); leakage assumption needs review"
            )

    findings["problems"] = problems
    findings["ok"] = not problems

    if verbose:
        status = "PASS" if not problems else "FAIL"
        print(f"[data_loader] integrity check: {status}")
        for problem in problems:
            print(f"[data_loader]   - {problem}")

    if problems and strict:
        raise DatasetNotAvailable(
            "Dataset verification failed:\n  - "
            + "\n  - ".join(problems)
            + f"\n\n{MANUAL_INSTRUCTIONS}"
        )
    return findings


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Acquire and verify UNSW-NB15.")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args(argv)

    try:
        download_dataset(force=args.force)
    except DatasetNotAvailable as exc:
        print(exc, file=sys.stderr)
        return 2

    try:
        verify_dataset(strict=True)
    except DatasetNotAvailable as exc:
        print(exc, file=sys.stderr)
        return 3

    corpus = load_corpus(verbose=True)
    counts = corpus[config.TARGET].value_counts().sort_index()
    print(f"[data_loader] class balance: "
          f"benign={counts.get(0, 0):,} ({counts.get(0, 0) / len(corpus):.1%}), "
          f"attack={counts.get(1, 0):,} ({counts.get(1, 0) / len(corpus):.1%})")
    print("[data_loader] attack categories:")
    for name, count in corpus[config.ATTACK_CAT].value_counts().items():
        print(f"               {name:<18} {count:>8,}")
    for filename, digest in file_checksums().items():
        print(f"[data_loader] sha256 {filename} = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
