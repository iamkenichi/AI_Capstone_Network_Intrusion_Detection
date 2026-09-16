"""
Central configuration for the Network Intrusion Detection capstone.

Everything that is a *decision* (paths, seeds, split ratios, column roles,
leakage exclusions) lives here so that it is declared once, is auditable in a
single place, and can be imported identically by scripts, notebooks, tests and
the Streamlit application.

All paths are derived from the repository root, which is resolved relative to
this file. Nothing in this project uses an absolute, machine-specific path.
"""

from __future__ import annotations

from pathlib import Path

# -----------------------------------------------------------------------------
# Repository layout
# -----------------------------------------------------------------------------
# src/config.py -> src/ -> <repo root>
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
FIGURES_DIR: Path = PROJECT_ROOT / "figures"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
TESTS_DIR: Path = PROJECT_ROOT / "tests"

#: Machine-readable artefacts produced by the pipeline (metrics tables, audit
#: tables, threshold sweeps). Kept inside reports/ so that a grader can find
#: every number that appears in the written reports.
METRICS_DIR: Path = REPORTS_DIR / "metrics"


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to (idempotent)."""
    for directory in (
        RAW_DIR,
        INTERIM_DIR,
        PROCESSED_DIR,
        MODELS_DIR,
        FIGURES_DIR,
        REPORTS_DIR,
        METRICS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_STATE: int = 42

#: Fractions of the *combined* UNSW-NB15 corpus assigned to each partition.
#: The split is performed in two stratified stages (see src.preprocessing).
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20  # fraction of the full corpus, taken from the 80% remainder

#: Number of folds used for every cross-validated hyper-parameter search.
CV_FOLDS: int = 5


# -----------------------------------------------------------------------------
# Dataset definition
# -----------------------------------------------------------------------------
DATASET_NAME: str = "UNSW-NB15 (partitioned training/testing sets)"

#: Official row counts of the two partitioned CSV files, used as an integrity
#: assertion after download. Source: Moustafa & Slay (2015, 2016).
EXPECTED_ROWS: dict[str, int] = {
    "UNSW_NB15_training-set.csv": 175_341,
    "UNSW_NB15_testing-set.csv": 82_332,
}
EXPECTED_TOTAL_ROWS: int = sum(EXPECTED_ROWS.values())  # 257,673
EXPECTED_N_COLUMNS: int = 45

#: Mirrors used by `src.data_loader.download_dataset`. The UNSW CloudStor host
#: that originally served these files has been retired, so the loader falls back
#: through public mirrors that serve byte-identical copies of the official
#: partitioned CSVs. Integrity is verified by row/column count and schema check,
#: not by trusting the mirror.
DOWNLOAD_MIRRORS: list[dict[str, str]] = [
    {
        "name": "GitHub: Nir-J/ML-Projects",
        "UNSW_NB15_training-set.csv": (
            "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
            "UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv"
        ),
        "UNSW_NB15_testing-set.csv": (
            "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/"
            "UNSW-Network_Packet_Classification/UNSW_NB15_testing-set.csv"
        ),
    },
    {
        "name": "Hugging Face: Mouwiya/UNSW-NB15 (training set only)",
        "UNSW_NB15_training-set.csv": (
            "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/"
            "UNSW_NB15_training-set.csv"
        ),
    },
]

#: Official feature-description file published with the dataset. Optional: it is
#: used to enrich the data dictionary with the authors' own wording.
FEATURE_DESCRIPTION_URL: str = (
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/NUSW-NB15_features.csv"
)
FEATURE_DESCRIPTION_FILE: str = "NUSW-NB15_features.csv"

TRAIN_FILE: str = "UNSW_NB15_training-set.csv"
TEST_FILE: str = "UNSW_NB15_testing-set.csv"


# -----------------------------------------------------------------------------
# Column roles
# -----------------------------------------------------------------------------
TARGET: str = "label"           # 0 = Normal / benign, 1 = Attack / malicious
ATTACK_CAT: str = "attack_cat"  # analysis-only grouping variable, NEVER a predictor

#: Columns that must never enter the feature matrix.
#:
#: * ``id``         - a synthetic row index. It is monotonically ordered within
#:                    each partition and therefore correlates with the record's
#:                    position in the capture, which is an artefact of how the
#:                    file was written rather than a property of the traffic.
#: * ``attack_cat`` - a direct function of the target: ``label == 0`` if and only
#:                    if ``attack_cat == "Normal"``. Using it as a predictor
#:                    would be textbook target leakage.
#: * ``label``      - the target itself.
LEAKAGE_COLUMNS: tuple[str, ...] = ("id", ATTACK_CAT, TARGET)

#: Nominal features in the UNSW-NB15 partitioned CSVs.
CATEGORICAL_FEATURES: tuple[str, ...] = ("proto", "service", "state")

#: Genuinely binary 0/1 flags. Left unscaled and not one-hot encoded.
#:
#: NOTE: ``is_ftp_login`` is *documented* as a binary flag by the dataset authors
#: but actually takes the values {0, 1, 2, 4} in the published CSVs. It is
#: therefore treated as a numeric count, not a flag. See
#: reports/dataset_documentation.md, "Known data-quality defects".
BINARY_FEATURES: tuple[str, ...] = ("is_sm_ips_ports",)

#: TTL-derived features. These are the single most predictive columns in the
#: dataset, but their predictive power is largely a *synthetic capture
#: artefact*: the IXIA PerfectStorm generators used to produce UNSW-NB15 emitted
#: benign and malicious traffic from hosts configured with different initial TTL
#: values, so `sttl` acts as a near-label proxy (a lookup rule on `sttl` alone
#: reaches ~87% accuracy on this corpus). They are RETAINED in the primary model
#: - excluding real signal would be unjustified - but an explicit ablation
#: (`--ablation no_ttl`) quantifies how much of the headline performance rests on
#: the artefact. See reports/Bias_Fairness_Analysis.md.
TTL_FEATURES: tuple[str, ...] = ("sttl", "dttl", "ct_state_ttl")

#: Column used to stratify every split. Stratifying on the 10-level attack
#: category (rather than the binary label) guarantees that rare families such as
#: Worms (n=164 after deduplication) appear in train, validation AND test, which
#: the binary label alone would not ensure.
STRATIFY_COLUMN: str = ATTACK_CAT

#: Remove exact duplicate feature vectors from the corpus BEFORE splitting.
#:
#: 40.4% of the published corpus (103,989 of 257,673 rows) consists of records
#: whose 42 predictor values are byte-identical to an earlier record.
#: Duplication is strongly class-correlated (Generic: 87.6% duplicated;
#: Normal: 8.1%). If duplicates are left in place, a random split puts identical
#: feature vectors in both train and test, so the model is scored partly on rows
#: it memorised - an optimistic bias, not generalisation. Deduplicating first is
#: the honest protocol; the `--ablation keep_duplicates` run quantifies the
#: difference.
DEDUPLICATE: bool = True

#: Sentinel used by the dataset authors for "no application-layer service was
#: identified for this flow". It is a genuine category, not a missing value.
SERVICE_MISSING_TOKEN: str = "-"

#: One-hot encoding keeps only categories seen at least this many times in the
#: TRAINING split; everything rarer is folded into a single ``infrequent``
#: bucket by scikit-learn's ``min_frequency`` mechanism.
#:
#: ``proto`` carries 133 levels, but 115 of them are exotic IP protocol numbers
#: appearing ~130 times each as part of synthetic protocol-scanning traffic.
#: A cutoff of 200 retains 15 named protocols covering 94.1% of records and
#: collapses the exotic tail into one "uncommon IP protocol" indicator. That is
#: both a dimensionality control and a *generalisation* choice: the model learns
#: "this flow used an unusual protocol" rather than memorising the specific
#: protocol names that happened to be scanned in 2015. It also gives the encoder
#: a defined destination for protocols never seen during training.
MIN_CATEGORY_FREQUENCY: int = 200


# -----------------------------------------------------------------------------
# Modelling / evaluation
# -----------------------------------------------------------------------------
#: Metric optimised during hyper-parameter search. UNSW-NB15 is only mildly
#: imbalanced (~32% benign / ~68% attack), so F1 on the attack class is a
#: reasonable selection criterion; average precision is reported alongside.
TUNING_SCORER: str = "f1"

#: Default decision threshold. Phase 10 re-tunes this on the validation split.
DEFAULT_THRESHOLD: float = 0.50

#: Operational grouping variables used for the subgroup performance audit.
#: These are *operational* strata, not demographic protected attributes - the
#: dataset contains no human subjects. See reports/Bias_Fairness_Analysis.md.
AUDIT_GROUPS: tuple[str, ...] = ("proto", "service", "state", ATTACK_CAT)

#: Plot styling shared by every figure in figures/.
FIGURE_DPI: int = 150
FIGURE_FORMAT: str = "png"

#: Colour-blind-safe palette (Okabe-Ito derived) used across all figures.
COLOR_BENIGN: str = "#0072B2"   # blue
COLOR_ATTACK: str = "#D55E00"   # vermillion
COLOR_NEUTRAL: str = "#4D4D4D"
COLOR_ACCENT: str = "#009E73"   # bluish green
COLOR_WARN: str = "#E69F00"     # orange
CATEGORICAL_PALETTE: tuple[str, ...] = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7",
    "#E69F00", "#56B4E9", "#F0E442", "#4D4D4D",
    "#8C564B", "#7F7F7F",
)
