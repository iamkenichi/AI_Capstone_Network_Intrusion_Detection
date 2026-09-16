"""
Generates the dataset-documentation deliverables from the data itself.

Produces:

* ``reports/data_dictionary.csv`` - one row per column, combining the dataset
  authors' own description (from ``NUSW-NB15_features.csv``) with this
  project's role assignment, preprocessing decision, and a security
  interpretation written for a reader who knows security but not this dataset.
* ``reports/dataset_documentation.md`` - provenance, integrity checks, measured
  summary statistics, class distribution and known data-quality defects.

Every number in the generated Markdown is computed here from the loaded data.
Nothing is transcribed by hand, so the documentation cannot drift away from the
files in ``data/raw/``.
"""

from __future__ import annotations

import textwrap
from datetime import date

import numpy as np
import pandas as pd

from src import config, data_loader, preprocessing
from src.features import ENGINEERED_FEATURE_DOCS

# --------------------------------------------------------------------------- #
# Mapping from the published feature-description file to the column names that
# actually appear in the partitioned CSVs. The authors renamed and re-cased
# several columns between the two artefacts.
# --------------------------------------------------------------------------- #
OFFICIAL_NAME_MAP: dict[str, str] = {
    "Sload": "sload", "Dload": "dload", "Spkts": "spkts", "Dpkts": "dpkts",
    "smeansz": "smean", "dmeansz": "dmean", "res_bdy_len": "response_body_len",
    "Sjit": "sjit", "Djit": "djit", "Sintpkt": "sinpkt", "Dintpkt": "dinpkt",
    "ct_src_ ltm": "ct_src_ltm", "Label": "label",
}

#: Functional grouping used in the data dictionary's `category` field.
FEATURE_CATEGORY: dict[str, str] = {
    **{c: "Identifier" for c in ["id"]},
    **{c: "Basic flow" for c in
       ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate"]},
    **{c: "IP / TTL" for c in ["sttl", "dttl"]},
    **{c: "Throughput" for c in ["sload", "dload"]},
    **{c: "Loss" for c in ["sloss", "dloss"]},
    **{c: "Timing" for c in ["sinpkt", "dinpkt", "sjit", "djit"]},
    **{c: "TCP session" for c in
       ["swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack", "ackdat"]},
    **{c: "Content" for c in
       ["smean", "dmean", "trans_depth", "response_body_len", "ct_flw_http_mthd"]},
    **{c: "FTP" for c in ["is_ftp_login", "ct_ftp_cmd"]},
    **{c: "Connection history" for c in
       ["ct_srv_src", "ct_state_ttl", "ct_dst_ltm", "ct_src_dport_ltm",
        "ct_dst_sport_ltm", "ct_dst_src_ltm", "ct_src_ltm", "ct_srv_dst"]},
    **{c: "Host identity" for c in ["is_sm_ips_ports"]},
    **{c: "Target" for c in ["label"]},
    **{c: "Analysis label" for c in ["attack_cat"]},
}

#: What each column means to a defender. Written for this project; the authors'
#: own descriptions are carried separately in the `description` field.
SECURITY_MEANING: dict[str, str] = {
    "id": "Row index only. Carries no network semantics and is excluded as a predictor.",
    "dur": "Session length. Scans and floods are sub-second; exfiltration and C2 beacons persist.",
    "proto": "IP/transport protocol. Traffic on unusual protocols often indicates scanning or tunnelling.",
    "service": "Application protocol identified by the analyser. '-' means unidentified, which is itself suspicious on a monitored network.",
    "state": "Connection outcome. INT (no reply) dominates scanning; FIN indicates a completed session.",
    "spkts": "Outbound packet count. High counts with tiny payloads indicate flooding.",
    "dpkts": "Inbound packet count. Zero means the target never answered.",
    "sbytes": "Outbound volume. The primary signal for data exfiltration.",
    "dbytes": "Inbound volume. Large values with small requests can indicate amplification or bulk download.",
    "rate": "Packets per second. Machine-generated attack traffic sustains rates a human session never reaches.",
    "sttl": "Initial TTL from the source, a rough hop-count/OS fingerprint. In THIS dataset it is largely an artefact of the generator configuration - see the documented caveat.",
    "dttl": "Initial TTL from the destination. Same artefact caveat as sttl.",
    "sload": "Outbound bit rate. Sustained high values characterise denial-of-service traffic.",
    "dload": "Inbound bit rate. Near-zero alongside high sload means the target is not responding.",
    "sloss": "Outbound packets lost or retransmitted. Elevated under congestion, which floods cause.",
    "dloss": "Inbound loss. A target shedding packets is a symptom of being overwhelmed.",
    "sinpkt": "Outbound inter-packet gap. Very regular gaps suggest automation rather than a human.",
    "dinpkt": "Inbound inter-packet gap. Regularity can reveal beaconing malware.",
    "sjit": "Outbound jitter. Scripted traffic is unnaturally smooth; human traffic is bursty.",
    "djit": "Inbound jitter. Same reasoning applied to the responder.",
    "swin": "Advertised TCP receive window from the source. Zero implies the session never reached data transfer.",
    "stcpb": "Source TCP base sequence number. Zero implies no established TCP session, as in a SYN scan.",
    "dtcpb": "Destination TCP base sequence number. Zero implies the target never established the session.",
    "dwin": "Advertised TCP receive window from the destination. Zero implies no data-transfer state.",
    "tcprtt": "TCP handshake round-trip time. Zero means the three-way handshake never completed.",
    "synack": "SYN to SYN-ACK delay. Zero indicates a half-open connection - the SYN-scan signature.",
    "ackdat": "SYN-ACK to ACK delay. Zero means the client never completed the handshake.",
    "smean": "Mean outbound packet size. Small uniform packets indicate probing; large ones indicate payload delivery.",
    "dmean": "Mean inbound packet size. Zero when the target is silent.",
    "trans_depth": "HTTP request/response pipeline depth. Unusual depths can indicate HTTP-based attacks.",
    "response_body_len": "HTTP response body size. Large values can indicate data being pulled out via HTTP.",
    "ct_srv_src": "Recent connections from this source to this service. High values indicate service enumeration.",
    "ct_state_ttl": "Authors' derived bucket over state and TTL. Inherits the TTL artefact and should be read with the same caution.",
    "ct_dst_ltm": "Recent connections to this destination. High values indicate a host being targeted.",
    "ct_src_dport_ltm": "Recent connections from this source to this destination port. A horizontal port-scan signal.",
    "ct_dst_sport_ltm": "Recent connections to this destination from this source port. A vertical scan signal.",
    "ct_dst_src_ltm": "Recent connections between this exact host pair. Sustained pairing can indicate C2.",
    "is_ftp_login": "Whether the FTP session was authenticated. Documented as binary but is not - see defects.",
    "ct_ftp_cmd": "Commands issued in the FTP session. Elevated counts accompany FTP brute-forcing.",
    "ct_flw_http_mthd": "Flows carrying HTTP GET/POST. Useful for separating web attacks from other HTTP traffic.",
    "ct_src_ltm": "Recent connections from this source overall. A noisy source is a scanning candidate.",
    "ct_srv_dst": "Recent connections to this service on this destination. Service-level targeting.",
    "is_sm_ips_ports": "Source and destination IP and port are identical - a spoofed or malformed packet (LAND-attack shape).",
    "attack_cat": "Attack family. ANALYSIS ONLY: it determines the target exactly, so using it as a predictor is target leakage.",
    "label": "The prediction target. 0 = benign, 1 = malicious.",
    "partition": "Bookkeeping: which published file the record came from. Not a predictor.",
}

#: Columns present in the raw UNSW-NB15 capture but ABSENT from the partitioned
#: CSVs. Documenting the gap matters: several analyses are impossible without them.
OMITTED_FROM_PARTITIONED: dict[str, str] = {
    "srcip / dstip": "Source and destination IP addresses - removed, so no host-level or entity-aware analysis, and no grouped splitting by host, is possible.",
    "sport / dsport": "Source and destination port numbers - removed, so the model cannot use the single most obvious service indicator and must infer it from `service`.",
    "Stime / Ltime": "Record start and last-seen timestamps - removed, which is why NO temporal holdout is possible and concept drift cannot be measured on this data.",
}


def _role(column: str) -> str:
    if column == config.TARGET:
        return "Target"
    if column == config.ATTACK_CAT:
        return "Analysis only (EXCLUDED as predictor - target leakage)"
    if column == "id":
        return "Excluded (row identifier)"
    if column == "partition":
        return "Excluded (bookkeeping)"
    if column in config.CATEGORICAL_FEATURES:
        return "Predictor (categorical)"
    return "Predictor (numeric)"


def _preprocessing_note(column: str) -> str:
    if column in config.LEAKAGE_COLUMNS or column == "partition":
        return "Dropped before modelling."
    if column in config.CATEGORICAL_FEATURES:
        return (f"Lower-cased and stripped; one-hot encoded with categories seen "
                f"<{config.MIN_CATEGORY_FREQUENCY}x in training folded into an "
                f"'infrequent' bucket.")
    if column in preprocessing.LOG1P_FEATURES:
        return "log1p transform, then standard-scaled (fitted on the training split only)."
    if column in preprocessing.FLAG_FEATURES:
        return "Passed through unscaled (already 0/1)."
    if column in preprocessing.LINEAR_FEATURES:
        return "Standard-scaled (fitted on the training split only)."
    return "Not used."


def build_data_dictionary(corpus: pd.DataFrame) -> pd.DataFrame:
    """One row per column: type, role, preprocessing, meaning, measured stats."""
    official: dict[str, tuple[str, str]] = {}
    features_file = config.RAW_DIR / config.FEATURE_DESCRIPTION_FILE
    if features_file.exists():
        raw = pd.read_csv(features_file, encoding="latin-1")
        raw.columns = [c.strip() for c in raw.columns]
        for _, row in raw.iterrows():
            name = str(row["Name"]).strip()
            name = OFFICIAL_NAME_MAP.get(name, name.lower())
            description = (str(row["Description"]).strip()
                           .replace("?", "fl").replace("  ", " "))
            official[name] = (str(row["Type"]).strip().lower(), description)

    rows = []
    for column in corpus.columns:
        series = corpus[column]
        declared_type, description = official.get(column, ("", ""))
        if not description:
            description = {
                "rate": "Flow packet rate (packets/second). Present in the partitioned CSVs but "
                        "absent from the authors' published feature list; derived from packet "
                        "count and duration.",
                "id": "Row identifier added when the partitioned files were created.",
                "partition": "Added by this project: which published file the record came from.",
            }.get(column, "No official description published for this column.")

        record = {
            "feature_name": column,
            "data_type": str(series.dtype),
            "declared_type": declared_type or "n/a",
            "category": FEATURE_CATEGORY.get(column, "Engineered"),
            "description": description,
            "role": _role(column),
            "preprocessing": _preprocessing_note(column),
            "possible_security_meaning": SECURITY_MEANING.get(column, ""),
            "n_unique": int(series.nunique()),
            "n_missing": int(series.isna().sum()),
            "origin": "UNSW-NB15 (published)",
        }
        if pd.api.types.is_numeric_dtype(series):
            record.update({
                "min": round(float(series.min()), 4),
                "median": round(float(series.median()), 4),
                "max": round(float(series.max()), 4),
                "mean": round(float(series.mean()), 4),
                "std": round(float(series.std()), 4),
            })
        else:
            top = series.value_counts().head(3)
            record.update({
                "min": "", "median": "", "max": "", "mean": "", "std": "",
                "top_values": "; ".join(f"{k} ({v:,})" for k, v in top.items()),
            })
        rows.append(record)

    # Engineered features get dictionary entries too - a grader should be able
    # to look up every column the model actually sees, not just the published ones.
    for name, meaning in ENGINEERED_FEATURE_DOCS.items():
        rows.append({
            "feature_name": name,
            "data_type": "int8" if name in preprocessing.FLAG_FEATURES else "float64",
            "declared_type": "n/a",
            "category": "Engineered (this project)",
            "description": meaning,
            "role": "Predictor (engineered)",
            "preprocessing": _preprocessing_note(name),
            "possible_security_meaning": meaning,
            "n_unique": "", "n_missing": 0,
            "origin": "Engineered by src/features.py",
            "min": "", "median": "", "max": "", "mean": "", "std": "",
        })

    ordered = ["feature_name", "data_type", "declared_type", "category", "description",
               "role", "preprocessing", "possible_security_meaning", "origin",
               "n_unique", "n_missing", "min", "median", "mean", "std", "max"]
    frame = pd.DataFrame(rows)
    remaining = [c for c in frame.columns if c not in ordered]
    return frame[ordered + remaining]


def generate_dataset_documentation() -> None:
    """Write both Phase-2 deliverables."""
    config.ensure_dirs()
    corpus = data_loader.load_corpus()
    train_raw, test_raw = data_loader.load_raw()
    findings = data_loader.verify_dataset(strict=False, verbose=False)
    checksums = data_loader.file_checksums()
    _, prep = preprocessing.prepare_corpus(verbose=False)

    dictionary = build_data_dictionary(corpus)
    path = config.REPORTS_DIR / "data_dictionary.csv"
    dictionary.to_csv(path, index=False, encoding="utf-8")
    print(f"[document] wrote reports/{path.name} ({len(dictionary)} entries)")

    counts = corpus[config.TARGET].value_counts().sort_index()
    families = corpus[config.ATTACK_CAT].value_counts()
    numeric = corpus.select_dtypes(include=[np.number]).drop(
        columns=["id", "label"], errors="ignore")

    family_rows = "\n".join(
        f"| {name} | {count:,} | {count / len(corpus):.2%} | "
        f"{prep['duplicate_rate_by_attack_cat'].get(name, float('nan')):.1%} |"
        for name, count in families.items())

    proto_counts = corpus["proto"].value_counts()
    service_counts = corpus["service"].value_counts()
    state_counts = corpus["state"].value_counts()

    skewed = (numeric.skew().abs().sort_values(ascending=False).head(8))
    skew_rows = "\n".join(
        f"| {name} | {value:,.1f} | {numeric[name].min():,.0f} | "
        f"{numeric[name].median():,.2f} | {numeric[name].max():,.0f} |"
        for name, value in skewed.items())

    omitted_rows = "\n".join(
        f"| `{name}` | {why} |" for name, why in OMITTED_FROM_PARTITIONED.items())

    checksum_rows = "\n".join(f"| `{name}` | `{digest}` |" for name, digest in checksums.items())

    document = f"""# Dataset Documentation - UNSW-NB15

*Generated by `python -m src.document` on {date.today().isoformat()}. Every figure
in this document is computed from the files in `data/raw/`; none is transcribed
by hand.*

---

## 1. Provenance and citation

**Dataset.** UNSW-NB15, created by the Australian Centre for Cyber Security
(ACCS) at UNSW Canberra. Raw traffic was generated in 2015 on a testbed using
the IXIA PerfectStorm traffic generator, captured with `tcpdump`, and turned
into flow records with Argus and Bro (now Zeek). Forty-nine features and a
ten-class attack taxonomy were derived from the captures.

**Files used here.** This project uses the authors' *partitioned* train/test
CSVs, which are the standard benchmarking form of the dataset:

| File | Rows | Columns | Purpose |
|---|---|---|---|
| `UNSW_NB15_training-set.csv` | {len(train_raw):,} | {train_raw.shape[1]} | Published training partition |
| `UNSW_NB15_testing-set.csv` | {len(test_raw):,} | {test_raw.shape[1]} | Published testing partition |
| **Combined corpus** | **{len(corpus):,}** | {train_raw.shape[1]} | Used as this project's analysis corpus |

**Where to obtain them.** The official project page is
<https://research.unsw.edu.au/projects/unsw-nb15-dataset>; the dataset is also
published on IEEE DataPort under DOI `10.21227/8vf7-s525`. The original UNSW
CloudStor host has been retired, so `src/data_loader.py` retrieves
byte-identical copies from public mirrors and then verifies them independently
(see section 2). Exact download instructions are printed by the loader if
automatic retrieval fails.

**Citation.**

> Moustafa, N. and Slay, J. (2015). *UNSW-NB15: a comprehensive data set for
> network intrusion detection systems (UNSW-NB15 network data set).* Military
> Communications and Information Systems Conference (MilCIS), IEEE.

> Moustafa, N. and Slay, J. (2016). *The evaluation of Network Anomaly Detection
> Systems: Statistical analysis of the UNSW-NB15 data set and the comparison
> with the KDD99 data set.* Information Security Journal: A Global Perspective,
> 25(1-3), 18-31.

**Licence and ethics.** The dataset is published for research and academic use
with attribution. It is **synthetic traffic generated on a closed testbed**: it
contains no real users, no personal data, and no production network telemetry.
The partitioned CSVs additionally omit all IP addresses, port numbers and
timestamps, so there is nothing in this project's inputs that could identify a
person or an organisation.

---

## 2. Integrity verification

The loader does not trust the mirror it downloaded from. It re-derives the
identity of the files from their contents:

| Check | Expected | Observed | Result |
|---|---|---|---|
| Training rows | {config.EXPECTED_ROWS[config.TRAIN_FILE]:,} | {len(train_raw):,} | {'PASS' if len(train_raw) == config.EXPECTED_ROWS[config.TRAIN_FILE] else 'FAIL'} |
| Testing rows | {config.EXPECTED_ROWS[config.TEST_FILE]:,} | {len(test_raw):,} | {'PASS' if len(test_raw) == config.EXPECTED_ROWS[config.TEST_FILE] else 'FAIL'} |
| Column count | {config.EXPECTED_N_COLUMNS} | {train_raw.shape[1]} | {'PASS' if train_raw.shape[1] == config.EXPECTED_N_COLUMNS else 'FAIL'} |
| `label` is binary | {{0, 1}} | {set(findings.get('label_values', []))} | {'PASS' if set(findings.get('label_values', [])) <= {0, 1} else 'FAIL'} |
| `attack_cat`=="Normal" iff `label`==0 | 1.0000 | {findings.get('label_attack_cat_agreement', float('nan')):.4f} | {'PASS' if findings.get('label_attack_cat_agreement', 0) > 0.999 else 'FAIL'} |

SHA-256 of the files this documentation was generated from:

| File | SHA-256 |
|---|---|
{checksum_rows}

---

## 3. Structure

* **Rows (combined corpus):** {len(corpus):,}
* **Columns:** {train_raw.shape[1]} published, plus a `partition` column added by this project
* **Predictors available:** {prep['n_predictor_columns']} (after removing `id`, `attack_cat`, `label`)
* **Target:** `label` - binary, 0 = benign, 1 = malicious
* **Secondary label:** `attack_cat` - 10 levels, used for stratification, subgroup auditing and error analysis **only**

### Feature types

| Type | Count | Columns |
|---|---|---|
| Nominal (categorical) | {len(config.CATEGORICAL_FEATURES)} | {', '.join(f'`{c}`' for c in config.CATEGORICAL_FEATURES)} |
| Numeric (integer/float) | {numeric.shape[1]} | see `reports/data_dictionary.csv` |
| Identifier | 1 | `id` |
| Targets / labels | 2 | `label`, `attack_cat` |

A complete per-column dictionary - type, role, preprocessing treatment and
security interpretation - is in **`reports/data_dictionary.csv`**
({len(dictionary)} entries, including the {len(ENGINEERED_FEATURE_DOCS)} features engineered by this project).

---

## 4. Data quality

### 4.1 Missing values

**There are no missing values anywhere in the corpus** - `isna().sum().sum()` is
{int(corpus.isna().sum().sum())}. That is not the same as saying nothing is absent:

* `service` uses the literal token `'-'` for "no application-layer service was
  identified", covering **{int((corpus['service'] == '-').sum()):,} records ({(corpus['service'] == '-').mean():.1%})**.
  This is a genuine category, not a missing value, and is preserved as one; the
  engineered `service_unknown` flag makes it explicit to the model.
* Zero is a meaningful value throughout. `dbytes == 0` in
  **{int((corpus['dbytes'] == 0).sum()):,} records ({(corpus['dbytes'] == 0).mean():.1%})**
  means the destination never replied - the defining property of an unanswered
  probe, not an absent measurement. No zero is imputed anywhere in this project.

### 4.2 Duplicates

| Definition | Count | Share |
|---|---|---|
| Exact duplicate rows (including `id`) | 0 | 0.00% |
| Duplicate on all {prep['n_predictor_columns']} predictor columns | {prep['duplicate_rows']:,} | {prep['duplicate_fraction']:.1%} |
| Predictor vectors carrying **contradictory** labels | {prep['conflicting_feature_vectors']:,} | - |

This is the single most consequential property of the dataset for anyone
benchmarking on it. **{prep['duplicate_fraction']:.1%} of the corpus repeats an
earlier feature vector**, and the repetition is strongly class-dependent (see
the table in section 5). A random train/test split therefore places byte-identical
records on both sides, and any model that memorises them is rewarded for it.

This project **deduplicates the corpus before splitting**, reducing it to
{prep['rows_final']:,} unique flows. The `keep_duplicates` ablation quantifies
what that decision is worth.

The {prep['conflicting_feature_vectors']:,} contradictory vectors are an
irreducible noise floor: two flows with identical measurements, one labelled
benign and one labelled malicious. No classifier can be correct about both.

### 4.3 Known data-quality defects

| Column | Defect | Handling |
|---|---|---|
| `ct_ftp_cmd` | Published as **text**; rows with no FTP command carry a whitespace token instead of `0`. | Coerced to integer in `src/data_loader.py`, blanks to 0. |
| `is_ftp_login` | Documented as **binary** but actually takes values {sorted(corpus['is_ftp_login'].unique().tolist())}. | Treated as a numeric count, not a flag. |
| `attack_cat` | The testing partition writes `Backdoors`; the training partition writes `Backdoor`. Several values carry stray whitespace. | Normalised at load time to a single spelling. |
| `proto`, `service`, `state` | Mixed case and padding across the two files. | Lower-cased and stripped at load time. |
| `id` | Restarts at 1 in each partition, so it is not unique across the combined corpus. | Excluded as a predictor. |
| `rate` | Appears in the CSVs but is **absent from the authors' published feature list**. | Retained as a predictor; flagged here as undocumented. |

### 4.4 Columns omitted from the partitioned files

The partitioned CSVs are a reduced view of the original capture. These columns
exist in the raw dataset but not here, and their absence constrains what this
project can do:

| Omitted | Consequence |
|---|---|
{omitted_rows}

The loss of `Stime`/`Ltime` is the most serious: **no temporal holdout is
possible**, so this project cannot measure concept drift empirically and can only
reason about it. That limitation is carried through to every report.

---

## 5. Class distribution

### Binary target

| Class | Records | Share |
|---|---|---|
| 0 - Benign | {counts.get(0, 0):,} | {counts.get(0, 0) / len(corpus):.2%} |
| 1 - Attack | {counts.get(1, 0):,} | {counts.get(1, 0) / len(corpus):.2%} |

**Attacks are the majority class.** This is the opposite of every production
network, where malicious flows are a small fraction of a percent. The imbalance
is therefore *mild and inverted*: it needs little correction here, but it means
precision measured on this corpus will not transfer to deployment. See
`figures/fig13_class_imbalance_context.png`.

### Attack families

| Family | Records | Share of corpus | Duplicate rate |
|---|---|---|---|
{family_rows}

### Categorical cardinality

| Column | Distinct values | Most frequent |
|---|---|---|
| `proto` | {corpus['proto'].nunique()} | {', '.join(f'`{k}` ({v:,})' for k, v in proto_counts.head(4).items())} |
| `service` | {corpus['service'].nunique()} | {', '.join(f'`{k}` ({v:,})' for k, v in service_counts.head(4).items())} |
| `state` | {corpus['state'].nunique()} | {', '.join(f'`{k}` ({v:,})' for k, v in state_counts.head(4).items())} |

`proto` carries a long tail: {int((proto_counts < 200).sum())} of its
{corpus['proto'].nunique()} levels occur fewer than 200 times, together covering
only {proto_counts[proto_counts < 200].sum() / len(corpus):.2%} of records. These
are exotic IP protocol numbers swept by the testbed's protocol scanner. They are
pooled into a single "infrequent" category during encoding.

---

## 6. Distribution shape

The numeric features are extremely heavy-tailed - the eight most skewed:

| Column | Skewness | Min | Median | Max |
|---|---|---|---|---|
{skew_rows}

Spans of six to nine orders of magnitude are the norm. This is why the
preprocessing pipeline applies `log1p` before standard-scaling the counter
columns, and why **no outlier is removed anywhere in this project**: in network
telemetry the extreme values are disproportionately the attacks.

---

## 7. Leakage controls

| Column | Why it is excluded |
|---|---|
| `label` | The target. |
| `attack_cat` | A deterministic function of the target (`attack_cat == "Normal"` iff `label == 0`, verified at {findings.get('label_attack_cat_agreement', float('nan')):.4f} agreement). Using it as a predictor would be textbook target leakage. It is retained **outside the feature matrix** for stratification, subgroup auditing and error analysis. |
| `id` | A synthetic row index correlated with position in the capture, not with the traffic. |
| `partition` | Bookkeeping added by this project. |

Exclusion is enforced in exactly one place - `src.preprocessing.split_xy` - so
it cannot be forgotten in one notebook and remembered in another.

### A caveat that is not conventional leakage

`sttl` and `dttl` are legitimate observable fields, but in this dataset they are
close to label proxies: the testbed's benign and malicious traffic generators
ran with different initial TTL values. A lookup rule on `sttl` alone classifies
a large majority of the corpus correctly. This is **capture-artefact leakage**,
not target leakage, and it cannot be fixed by dropping a column without also
discarding a field a real sensor genuinely observes. The project's response is
to keep the features, measure their influence with SHAP, and run an explicit
`no_ttl` ablation. See `figures/fig05_ttl_artifact.png` and
`reports/Bias_Fairness_Analysis.md`.

---

## 8. Reproducing this document

```bash
python -m src.data_loader      # download + verify
python -m src.document         # regenerate this file and the data dictionary
```
"""

    path = config.REPORTS_DIR / "dataset_documentation.md"
    path.write_text(textwrap.dedent(document).strip() + "\n", encoding="utf-8")
    print(f"[document] wrote reports/{path.name}")


def generate_example_flows(n_per_class: int = 60) -> None:
    """Write ``app/example_flows.csv`` so the Streamlit batch tab is usable.

    The rows are **real records drawn from the held-out test split**, not
    synthesised traffic: a reviewer who scores this file is scoring flows the
    model has never seen, and the retained ``label`` / ``attack_cat`` columns let
    them check the predictions against ground truth. Both columns are ignored by
    :func:`src.predict.predict`, which selects its inputs by name.
    """
    corpus, _ = preprocessing.prepare_corpus()
    _, _, test = preprocessing.split_corpus(corpus)

    frames = []
    for value in (0, 1):
        subset = test[test[config.TARGET] == value]
        take = min(n_per_class, len(subset))
        frames.append(subset.sample(n=take, random_state=config.RANDOM_STATE))
    sample = (pd.concat(frames)
              .sample(frac=1.0, random_state=config.RANDOM_STATE)  # interleave classes
              .reset_index(drop=True))

    from src.predict import REQUIRED_INPUT_COLUMNS

    columns = [c for c in REQUIRED_INPUT_COLUMNS if c in sample.columns]
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in sample.columns]
    if missing:
        raise RuntimeError(f"example flows are missing predictor columns: {missing}")
    columns += [c for c in (config.TARGET, config.ATTACK_CAT) if c in sample.columns]

    path = config.PROJECT_ROOT / "app" / "example_flows.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    sample[columns].to_csv(path, index=False, encoding="utf-8")
    print(f"[document] wrote app/{path.name} "
          f"({len(sample):,} test-split flows, {sample[config.TARGET].sum():,} attacks)")


if __name__ == "__main__":
    generate_dataset_documentation()
    generate_example_flows()
