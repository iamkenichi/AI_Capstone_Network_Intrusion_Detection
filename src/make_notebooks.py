"""
Builds the six analysis notebooks in ``notebooks/``.

The notebooks are *generated* rather than hand-edited so that they cannot drift
away from the ``src`` package they orchestrate. Each one is deliberately thin:
it narrates the reasoning, calls into ``src``, and displays results. There is no
analysis logic that exists only inside a notebook, which is what keeps the
notebooks, the tests, the reports and the Streamlit app all describing the same
system.

Expensive stages (hyper-parameter search, SHAP over the full test split) are
**loaded from artefacts** rather than recomputed, and each notebook states the
command that produced them. A notebook that takes an hour to run is a notebook
nobody runs.

Usage
-----
    python -m src.make_notebooks            # write the notebooks
    python -m src.make_notebooks --execute  # write, then execute them in place
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

from src import config

HEADER = """import sys, warnings
from pathlib import Path

# Make the repository root importable no matter where Jupyter was launched from.
ROOT = Path.cwd()
while not (ROOT / "src" / "config.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)
warnings.filterwarnings("ignore", category=FutureWarning)

from src import config
print(f"Repository root : {ROOT}")
print(f"Random seed     : {config.RANDOM_STATE}")"""

SHOW_FIGURE = '''def show(figure_name, caption=None):
    """Display a figure produced by the pipeline, or explain how to produce it."""
    path = config.FIGURES_DIR / f"{figure_name}.png"
    if path.exists():
        display(Image(filename=str(path)))
        if caption:
            display(Markdown(f"*{caption}*"))
    else:
        display(Markdown(
            f"> ⚠️ `figures/{figure_name}.png` not found. "
            "Run the pipeline stage noted above to generate it."))'''


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip())


def _notebook(cells: list) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook(cells=cells)
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    }
    return notebook


# --------------------------------------------------------------------------- #
def notebook_01() -> nbf.NotebookNode:
    return _notebook([
        md("""
# 01 — Problem Framing and Data Understanding

**Machine Learning-Based Network Intrusion Detection and Anomaly Classification**

This notebook establishes what we are predicting, from what, and what is wrong
with the data before any modelling decision is made.

| | |
|---|---|
| **Task** | Supervised binary classification |
| **Target** | `label` — 0 = benign, 1 = malicious |
| **Unit** | One bidirectional network flow record |
| **Dataset** | UNSW-NB15 (Moustafa & Slay, 2015) |
| **Constraint** | Flow statistics only — no payloads, IPs, ports or timestamps |

Full framing, pre-registered success criteria and risk analysis:
[`reports/problem_statement.md`](../reports/problem_statement.md).
"""),
        code(HEADER),
        md("""
## 1. Acquire and verify the dataset

The loader does not trust the mirror it downloads from. It re-derives the
dataset's identity from the file contents: expected row counts, column count,
the binary encoding of `label`, and the rule that `attack_cat == "Normal"` if
and only if `label == 0`.
"""),
        code("""
from src import data_loader

data_loader.download_dataset(verbose=True)
findings = data_loader.verify_dataset(strict=True, verbose=True)

display(Markdown("**Integrity findings**"))
for key, value in findings.items():
    if key != "problems":
        print(f"  {key:<32} {value}")
"""),
        code("""
for filename, digest in data_loader.file_checksums().items():
    print(f"{filename:<32} sha256 {digest}")
"""),
        md("""
### Interpretation

Both partitions match the row counts published by the dataset authors
(175,341 / 82,332) and carry the expected 45 columns. The
`label` / `attack_cat` agreement is exactly 1.0000 — which is precisely why
`attack_cat` must never be used as a predictor. It *is* the target, relabelled.
"""),
        md("## 2. Structure"),
        code("""
corpus = data_loader.load_corpus(verbose=True)

print(f"\\nShape: {corpus.shape}")
print(f"Memory: {corpus.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
display(corpus.head(5))
"""),
        code("""
summary = pd.DataFrame({
    "dtype": corpus.dtypes.astype(str),
    "n_unique": corpus.nunique(),
    "n_missing": corpus.isna().sum(),
    "example": corpus.iloc[0],
})
display(summary)
"""),
        md("""
## 3. Data quality

Three questions, in order of how much they affect the result: is anything
missing, is anything duplicated, and does anything leak the target?
"""),
        code("""
print(f"Total missing values across the entire corpus: {corpus.isna().sum().sum()}")

# "No missing values" is not the same as "nothing is absent".
sentinel = (corpus["service"] == "-").sum()
silent = (corpus["dbytes"] == 0).sum()
print(f"service == '-'  (no L7 service identified) : {sentinel:,} ({sentinel/len(corpus):.1%})")
print(f"dbytes == 0     (destination never replied) : {silent:,} ({silent/len(corpus):.1%})")
print("\\nBoth are REAL values, not absences. Neither is imputed anywhere in this project.")
"""),
        code("""
predictors = [c for c in corpus.columns if c not in (*config.LEAKAGE_COLUMNS, "partition")]

exact = corpus.duplicated().sum()
on_predictors = corpus.duplicated(subset=predictors).sum()
conflicting = (corpus.groupby(predictors, dropna=False, observed=True)["label"]
               .nunique() > 1).sum()

print(f"Exact duplicate rows (including id) : {exact:,}")
print(f"Duplicate on the {len(predictors)} predictors  : {on_predictors:,} "
      f"({on_predictors/len(corpus):.1%})")
print(f"Vectors with CONTRADICTORY labels   : {conflicting:,}")
"""),
        md("""
### This is the most consequential finding in the dataset

**40.4% of the corpus repeats an earlier feature vector.** A random train/test
split therefore places byte-identical records on both sides, and a model that
memorises them is rewarded for it. Published UNSW-NB15 benchmarks that do not
deduplicate are reporting partly-memorised performance.

This project deduplicates **before** splitting. Notebook 03 shows the effect,
and the `keep_duplicates` ablation measures what it is worth.

The contradictory vectors are an irreducible noise floor — identical
measurements labelled both ways. No classifier can be right about both copies.
"""),
        code("""
duplicate_mask = corpus.duplicated(subset=predictors, keep="first")
by_family = (corpus.assign(dup=duplicate_mask)
             .groupby("attack_cat", observed=True)["dup"]
             .agg(records="size", duplicated="sum", rate="mean")
             .sort_values("rate", ascending=False))
display(by_family.style.format({"records": "{:,}", "duplicated": "{:,}", "rate": "{:.1%}"}))
"""),
        md("""
Duplication is **strongly class-dependent** — Generic is 87.6% duplicated,
Normal only 8.1%. That asymmetry is why deduplication changes the class balance
so much, and why leaving duplicates in place biases the result rather than
merely inflating the sample size.
"""),
        md("## 4. Target and class distribution"),
        code("""
counts = corpus["label"].value_counts().sort_index()
print("Binary target:")
for value, count in counts.items():
    name = "benign" if value == 0 else "attack"
    print(f"  {value} ({name:<6}) {count:>8,}  {count/len(corpus):.2%}")

print("\\nAttack families:")
for name, count in corpus["attack_cat"].value_counts().items():
    print(f"  {name:<18} {count:>8,}  {count/len(corpus):.2%}")
"""),
        md("""
**Attacks are the MAJORITY class (63.9%).** This is the inverse of any real
network, where malicious flows are a fraction of a percent.

The consequence carries through every later result: **precision depends on the
base rate; recall and false-positive rate do not.** Any precision figure measured
on this corpus is optimistic for deployment.
"""),
        md("## 5. Leakage audit"),
        code("""
cross = pd.crosstab(corpus["attack_cat"], corpus["label"])
display(cross)

is_normal = corpus["attack_cat"].str.lower().eq("normal")
agreement = (is_normal == corpus["label"].eq(0)).mean()
print(f"\\nattack_cat == 'Normal'  <=>  label == 0 : agreement = {agreement:.6f}")
print("\\nPerfect agreement => attack_cat IS the target. Excluded as a predictor.")
"""),
        code("""
print("Columns excluded from the feature matrix, and why:\\n")
reasons = {
    "id": "synthetic row index; correlates with position in the capture, not with traffic",
    "attack_cat": "deterministic function of the target -> textbook target leakage",
    "label": "the target itself",
    "partition": "bookkeeping added by this project",
}
for column, reason in reasons.items():
    print(f"  {column:<12} {reason}")

from src import preprocessing
X, y = preprocessing.split_xy(corpus.head(100))
print(f"\\nEnforced in one place (src.preprocessing.split_xy): "
      f"{X.shape[1]} predictors survive, target '{y.name}' separated.")
"""),
        md("""
### A caveat that is *not* conventional leakage

`sttl` and `dttl` are legitimate fields a real sensor observes — but in this
dataset the benign and attack generators ran on hosts with different initial TTL
values, making `sttl` close to a label proxy. That is **capture-artefact
leakage**, and it cannot be fixed by dropping a column without also discarding a
field a real sensor genuinely sees.

Notebook 02 quantifies it; notebook 06 measures its influence on the fitted
model with SHAP; and the `no_ttl` ablation retrains without it.
"""),
        md("## 6. Data dictionary"),
        code("""
dictionary_path = config.REPORTS_DIR / "data_dictionary.csv"
if not dictionary_path.exists():
    from src import document
    document.generate_dataset_documentation()

dictionary = pd.read_csv(dictionary_path)
print(f"{len(dictionary)} documented columns "
      f"({(dictionary['origin'] != 'UNSW-NB15 (published)').sum()} engineered by this project)\\n")
display(dictionary[["feature_name", "category", "role", "possible_security_meaning"]].head(15))
"""),
        md("""
---

## Conclusions

| Finding | Consequence for the project |
|---|---|
| Both partitions verified against published row counts and SHA-256 | Dataset identity is confirmed, not assumed |
| Zero missing values, but `'-'` and `0` are meaningful | Nothing is imputed anywhere |
| 40.4% duplicate feature vectors, class-correlated | **Deduplicate before splitting** (notebook 03) |
| 414 contradictory vectors | Irreducible noise floor on achievable accuracy |
| Attacks are the majority class | Precision will not transfer to deployment |
| `attack_cat` determines `label` exactly | Excluded as a predictor at a single enforcement point |
| TTL fields look like label proxies | Quantified in notebook 02, measured in notebook 06, ablated in `src/train.py` |
| No timestamps, IPs or ports in the partitioned files | **No temporal holdout is possible** — drift cannot be measured |

**Next:** [`02_eda.ipynb`](02_eda.ipynb) — exploratory analysis on the training
split only.
"""),
    ])


# --------------------------------------------------------------------------- #
def notebook_02() -> nbf.NotebookNode:
    return _notebook([
        md("""
# 02 — Exploratory Data Analysis

## Scope discipline

Two kinds of analysis appear here and they are deliberately kept apart:

* **Corpus-level bookkeeping** (row counts, class balance, duplication)
  describes the dataset *as published*. Counting rows in a public dataset is not
  test-set peeking.
* **Distributional and relational analysis** — anything that could inform a
  modelling decision — is computed on the **training split only**. The validation
  and test splits are never inspected here.

Figures are produced by `src/eda.py` so that the notebook, the report and the
presentation all show the same images from the same code.
"""),
        code(HEADER),
        code(SHOW_FIGURE),
        code("""
from src import eda, preprocessing, data_loader

corpus = data_loader.load_corpus()
dedup, prep = preprocessing.prepare_corpus(verbose=True)
train, _, _ = preprocessing.split_corpus(dedup, verbose=True)

print(f"\\nEDA below uses the TRAINING SPLIT ONLY: {len(train):,} records")
"""),
        md("""
## 1. Class balance — and what deduplication does to it
"""),
        code('show("fig01_class_distribution")'),
        md("""
Removing duplicate feature vectors turns a 63.9% attack-majority corpus into a
55.6% / 44.4% benign-majority one, because duplication is concentrated in the
attack classes.

**Neither balance resembles a production network.** In a real enterprise,
malicious flows are a fraction of a percent. This is the single most important
caveat attached to every precision figure later in the project.
"""),
        code('show("fig13_class_imbalance_context")'),
        md("""
The gap is three orders of magnitude. Precision measured at a 44% base rate will
not survive a move to 0.1%. Recall and false-positive *rate* will.
"""),
        md("## 2. Attack families"),
        code('show("fig02_attack_category_distribution")'),
        code('show("fig03_duplication_by_attack_category")'),
        md("""
Families span three orders of magnitude, and duplication varies from 87.6%
(Generic) to 8.1% (Normal). Two consequences:

1. Rare families get wide confidence intervals — so every per-family metric in
   this project is reported **with its sample count**.
2. Families whose data was heavily duplicated contributed far fewer *distinct*
   training examples than their raw counts suggest. Their true training support
   is much smaller than it appears.
"""),
        md("## 3. Categorical structure"),
        code('show("fig04_categorical_attack_rates")'),
        code("""
for column in ("proto", "service", "state"):
    stats = (train.groupby(column, observed=True)["label"]
             .agg(records="size", attack_rate="mean")
             .query("records >= 50")
             .sort_values("attack_rate", ascending=False))
    print(f"\\n=== {column} (levels with >= 50 training records) ===")
    display(stats.head(8).style.format({"records": "{:,}", "attack_rate": "{:.1%}"}))
"""),
        md("""
### Interpretation

Several protocols are **100% malicious** and ARP is **100% benign**. That is not
a property of those protocols in the real world — it is an artefact of the
testbed, where the attack generator emitted exotic IP protocols that the
background-traffic generator never used.

**Design response:** rare protocol levels are pooled into a single "uncommon
protocol" indicator during encoding, so the model learns a generalisable
property rather than memorising which protocol numbers a 2015 scanner swept.

`state` is the one categorical whose signal is genuinely defensible: `int` (no
reply received) really is what scanning looks like on any network.
"""),
        md("""
## 4. The TTL artefact — the most important EDA finding
"""),
        code('show("fig05_ttl_artifact")'),
        code("""
lookup = train.groupby("sttl", observed=True)["label"].mean()
rule = (train["sttl"].map(lookup) > 0.5).astype(int)
accuracy = (rule == train["label"]).mean()
baseline = max(train["label"].mean(), 1 - train["label"].mean())

print(f"Accuracy of a lookup rule using sttl AND NOTHING ELSE : {accuracy:.4f}")
print(f"Majority-class baseline                               : {baseline:.4f}")
print(f"Lift from one column                                  : {accuracy - baseline:+.4f}")
"""),
        md("""
### Why this matters more than anything else in this notebook

One column recovers most of the achievable separation. The UNSW-NB15 testbed ran
its benign and attack generators on hosts configured with different initial TTL
values, so `sttl` largely encodes **which generator produced this flow**, not
**is this flow hostile**.

**The features are kept.** TTL is a real field a sensor observes and carries
genuine information (hop count, OS fingerprint). Deleting real signal on
suspicion would be its own methodological error.

**But its influence is measured, not assumed:**
- Notebook 06 quantifies it on the fitted model with SHAP.
- `python -m src.train --ablation no_ttl` retrains with all three TTL features
  removed, so we can see exactly what performance survives without them.
"""),
        md("## 5. Numeric distributions"),
        code('show("fig06_numeric_distributions_by_class")'),
        code("""
heavy = ["dur", "sbytes", "dbytes", "rate", "sload", "sjit"]
display(train[heavy].describe(percentiles=[.01, .5, .99]).T
        .style.format("{:,.3g}"))
print("\\nSkewness:")
print(train[heavy].skew().round(1).to_string())
"""),
        md("""
Six to nine orders of magnitude on every counter, with skewness in the tens to
hundreds. This is why the pipeline applies `log1p` before standard-scaling:
monotonic (trees unaffected), defined at zero (so the 46.7% zero-byte reverse
directions survive), and reversible.

The separation between classes is real but partial — attacks concentrate at
short durations, small payloads and high packet rates, which is the shape of
scanning and flooding.
"""),
        md("## 6. Outliers — analysed, not deleted"),
        code('show("fig07_outlier_boxplots")'),
        code("""
extreme = train.nlargest(8, "sbytes")[
    ["attack_cat", "proto", "service", "state", "dur", "sbytes", "dbytes", "spkts"]]
display(extreme)
print("\\nThese are genuine records of genuine behaviour, not sensor errors.")
"""),
        md("""
### Decision: no outlier is removed anywhere in this project

In network telemetry the extreme values are disproportionately the attacks. A
14 MB transfer, a 5.99 Gbit/s load and a 10,646-packet burst are exactly the
events an IDS exists to catch. Clipping the tail would delete the signal.

The only values that would have been treated as erroneous are non-finite ones,
and the dataset contains none. Heavy tails are handled by **transformation**,
not deletion.
"""),
        md("## 7. Redundancy among features"),
        code('show("fig08_correlation_heatmap")'),
        code("""
numeric = train.select_dtypes(include=[np.number]).drop(
    columns=["id", "label"], errors="ignore")
corr = numeric.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
pairs = upper[upper > 0.90].sort_values(ascending=False)

print(f"{len(pairs)} feature pairs exceed |r| = 0.90\\n")
display(pairs.head(10).to_frame("abs_correlation").style.format("{:.4f}"))
"""),
        md("""
The tightest pairs are **structural**, not coincidental: `sbytes`/`sloss` and
`dbytes`/`dloss` because loss counters are derived from byte counters;
`is_ftp_login`/`ct_ftp_cmd` because they encode the same FTP event twice.

Two consequences:
- The design matrix is severely ill-conditioned, so logistic regression needs L2
  regularisation for stable coefficients.
- **SHAP, not impurity-based importance, is used for attribution** — impurity
  gain splits credit arbitrarily between correlated twins.
"""),
        md("## 8. Feature signal and per-family fingerprints"),
        code('show("fig09_univariate_feature_ranking")'),
        code('show("fig10_engineered_feature_separation")'),
        code('show("fig11_volume_scatter")'),
        code('show("fig12_attack_family_profile")'),
        md("""
### Interpretation

- The TTL family still occupies the top univariate ranks — consistent with
  section 4.
- Several engineered features enter the top 25, confirming they add signal
  rather than restating existing columns.
- `src_byte_ratio` is strongly **bimodal**: attacks pile up near 1.0 (source
  talks, target never answers).
- The horizontal band at `dbytes = 0` in the scatter is the 46.7% of flows that
  never received a reply — captured directly by the engineered `is_one_way` flag.
- Each attack family has a distinguishable fingerprint, which is what makes the
  per-family error analysis in notebook 06 interpretable rather than descriptive.

**Caveat:** univariate correlation cannot see interactions. It is used for
orientation; the model-facing attribution is the SHAP analysis in notebook 06.
"""),
        md("""
---

## Findings that changed the project's design

| Finding | Design consequence |
|---|---|
| Duplication is class-correlated | Deduplicate **before** splitting; ablate to measure the cost |
| `sttl` alone reaches ~81% accuracy | Keep the features, quantify with SHAP, ablate with `no_ttl` |
| Exotic protocols are 100% malicious | Pool rare categories rather than memorise them |
| Outliers are genuine attacks | Transform with `log1p`, never clip |
| 12 feature pairs exceed \\|r\\| = 0.90 | Use SHAP rather than impurity importance |
| Six to nine orders of magnitude | `log1p` then standard-scale |
| Class balance is inverted vs production | Report recall and FPR as the transferable metrics |

**Next:** [`03_preprocessing_feature_engineering.ipynb`](03_preprocessing_feature_engineering.ipynb)
"""),
    ])


# --------------------------------------------------------------------------- #
def notebook_03() -> nbf.NotebookNode:
    return _notebook([
        md("""
# 03 — Preprocessing and Feature Engineering

This notebook builds the data the models actually see, and demonstrates that the
three leakage controls the project claims are real rather than asserted:

1. **No target leakage** — `id`, `attack_cat` and `label` never enter the matrix.
2. **No preprocessing leakage** — scalers and encoders are fitted inside the
   training fold only.
3. **No duplicate leakage** — identical feature vectors never span train and test.
"""),
        code(HEADER),
        md("""
## 1. Corpus preparation

Deduplication happens **before** the split. Doing it afterwards would leave
identical vectors straddling train and test — precisely the problem it exists to
prevent.
"""),
        code("""
from src import preprocessing

corpus, report = preprocessing.prepare_corpus(verbose=True)

print()
for key in ("rows_loaded", "duplicate_rows", "duplicate_fraction",
            "conflicting_feature_vectors", "rows_final", "missing_values"):
    print(f"  {key:<30} {report[key]}")
"""),
        md("## 2. Stratified split"),
        code("""
train, val, test = preprocessing.split_corpus(corpus, verbose=True)

distribution = pd.DataFrame({
    name: part["attack_cat"].value_counts(normalize=True)
    for name, part in (("train", train), ("val", val), ("test", test))
}).sort_values("train", ascending=False)
print("\\nAttack-family proportions are preserved across all three splits:")
display(distribution.style.format("{:.4f}"))
"""),
        md("""
Stratification is on **`attack_cat`**, not on `label`. Because `label` is a
deterministic function of `attack_cat`, stratifying on the family stratifies the
binary target as a side effect — while additionally guaranteeing that rare
families (Worms, n=164 after deduplication) appear in all three splits. The
binary label alone would not ensure that.
"""),
        code("""
predictors = [c for c in corpus.columns if c not in (*config.LEAKAGE_COLUMNS, "partition")]

def vectors(frame):
    return set(map(tuple, frame[predictors].itertuples(index=False)))

train_v, val_v, test_v = vectors(train), vectors(val), vectors(test)
print(f"Feature vectors shared between train and test : {len(train_v & test_v)}")
print(f"Feature vectors shared between train and val  : {len(train_v & val_v)}")
print("\\nZero overlap => the test metrics measure generalisation, not memorisation.")
"""),
        md("## 3. Leakage enforcement"),
        code("""
X_train, y_train = preprocessing.split_xy(train)
X_val, y_val = preprocessing.split_xy(val)
X_test, y_test = preprocessing.split_xy(test)

print(f"Predictors: {X_train.shape[1]}    Target: '{y_train.name}'\\n")
for forbidden in (*config.LEAKAGE_COLUMNS, "partition"):
    status = "PRESENT — LEAK!" if forbidden in X_train.columns else "absent (correct)"
    print(f"  {forbidden:<12} {status}")
"""),
        md("## 4. Feature engineering"),
        code("""
from src.features import ENGINEERED_FEATURE_DOCS, add_engineered_features

print(f"{len(ENGINEERED_FEATURE_DOCS)} engineered features:\\n")
for name, rationale in ENGINEERED_FEATURE_DOCS.items():
    print(f"  {name}")
    print(f"      {rationale[:110]}...")
"""),
        code("""
sample = X_train.head(6)
engineered = add_engineered_features(sample)

display(Markdown("**Raw flow measurements**"))
display(sample[["proto", "service", "state", "dur", "sbytes", "dbytes",
                "spkts", "dpkts", "synack", "ackdat"]])

display(Markdown("**Engineered from them**"))
display(engineered[list(ENGINEERED_FEATURE_DOCS)].round(4))
"""),
        md("""
### Why these are row-wise, and why that matters twice

Every engineered feature is a function of **one flow record**. None uses a
statistic estimated from the data — no target encoding, no dataset-level means,
no group aggregates. That property delivers two things at once:

1. **It cannot leak.** A feature that never sees the label cannot leak the
   target, and one that never sees another row cannot leak across the split.
2. **It is deployable.** Each can be computed by a sensor on a single flow at
   inference time, which is how an IDS must operate.
"""),
        code("""
# Numerical safety on the degenerate flows that make up nearly half the dataset.
degenerate = X_train[(X_train["dpkts"] == 0) & (X_train["dur"] == 0)].head(3)
print(f"Flows with zero response AND zero duration in the training split: "
      f"{((X_train['dpkts'] == 0) & (X_train['dur'] == 0)).sum():,}\\n")

if len(degenerate):
    out = add_engineered_features(degenerate)[list(ENGINEERED_FEATURE_DOCS)]
    display(out.round(4))
    print(f"\\nAll finite: {np.isfinite(out.to_numpy(float)).all()}")
"""),
        md("""
46.7% of flows have `dpkts == 0`, so an unguarded division would corrupt a large
fraction of the matrix rather than a rare edge case. Every ratio is formed as
`a / (a + b)` — bounded to [0, 1] — rather than `a / b`, every denominator is
clipped, and the module asserts on output that no non-finite value was produced.
`tests/test_features.py` verifies this on hand-built records.
"""),
        md("## 5. The preprocessing pipeline"),
        code("""
groups = preprocessing.feature_columns()
for name, columns in groups.items():
    print(f"{name:<12} ({len(columns):>2}) {', '.join(columns[:6])}"
          f"{' ...' if len(columns) > 6 else ''}")
"""),
        code("""
pipeline = preprocessing.build_feature_pipeline()
pipeline.fit(X_train, y_train)

design = pipeline.transform(X_train)
names = preprocessing.transformed_feature_names(pipeline)

print(f"Design matrix : {design.shape}")
print(f"All finite    : {np.isfinite(design).all()}")
print(f"\\nOne-hot columns produced:")
for prefix in ("proto_", "service_", "state_"):
    produced = [n for n in names if n.startswith(prefix)]
    print(f"  {prefix:<10} {len(produced):>2} — {', '.join(produced[:6])}"
          f"{' ...' if len(produced) > 6 else ''}")
"""),
        md("""
`proto` has 133 levels in the corpus but produces far fewer columns: levels seen
fewer than 200 times in training are folded into a single `infrequent` bucket.
That is both dimensionality control and a **generalisation choice** — the model
learns "this flow used an unusual protocol" rather than memorising which
protocol numbers were scanned in 2015. It also gives protocols never seen during
training a defined destination at inference time, instead of an exception.
"""),
        code("""
# Proof that fitting used the training split ONLY.
scaler = pipeline.named_steps["preprocess"].named_transformers_["num"]
engineered_train = pipeline.named_steps["engineer"].transform(X_train)
expected = engineered_train[groups["linear"]].to_numpy(float).mean(axis=0)

print(f"Scaler means match the TRAINING split exactly: "
      f"{np.allclose(scaler.mean_, expected)}")

before = scaler.mean_.copy()
pipeline.transform(X_val)
print(f"Transforming the validation split left the fitted state unchanged: "
      f"{np.array_equal(scaler.mean_, before)}")
"""),
        code("""
# An unseen protocol must be absorbed, not raise — the failure mode that takes a
# deployed detector offline the first time an unusual protocol crosses the wire.
novel = X_val.head(5).copy()
novel["proto"] = "protocol-invented-in-2030"
novel["service"] = "unheard-of-service"

result = pipeline.transform(novel)
print(f"Unseen categories handled: shape {result.shape}, all finite {np.isfinite(result).all()}")
"""),
        md("## 6. Class imbalance"),
        code("""
balance = y_train.value_counts(normalize=True).sort_index()
print(f"Training split: {balance[0]:.1%} benign / {balance[1]:.1%} attack")

from src.train import scale_pos_weight
print(f"scale_pos_weight (n_neg / n_pos) = {scale_pos_weight(y_train):.4f}")
print("\\nImbalance is MILD and INVERTED (attacks are the larger class).")
print("class_weight / scale_pos_weight are included as TUNED hyper-parameters so")
print("the search decides empirically rather than by assumption.")
"""),
        md("""
### Why SMOTE is not used by default

SMOTE interpolates between neighbouring minority points. On this data that is
questionable: many features are Boolean flags or bounded counters, and a
synthetic flow with `tcp_handshake_complete = 0.43` does not correspond to any
packet sequence that could exist on a wire.

It is nevertheless **tested** rather than dismissed — `python -m src.train
--ablation smote` applies `SMOTENC` (which leaves nominal columns
un-interpolated) to the **training split only**, and the comparison is reported
in the Model Evaluation Report. The argument is settled by evidence, not
assertion.
"""),
        md("## 7. Persist the splits"),
        code("""
config.ensure_dirs()
for name, part in (("train", train), ("val", val), ("test", test)):
    path = config.PROCESSED_DIR / f"{name}.parquet"
    part.to_parquet(path, index=False)
    print(f"wrote data/processed/{name}.parquet  ({len(part):,} rows)")
"""),
        md("""
---

## Summary

| Control | Mechanism | Verified by |
|---|---|---|
| No target leakage | Single enforcement point in `split_xy` | Section 3; `tests/test_preprocessing.py` |
| No preprocessing leakage | Estimator shares a `Pipeline` with the transforms | Section 5; scaler-mean check |
| No duplicate leakage | Deduplicate before splitting | Section 2; zero-overlap check |
| Unseen categories safe | `min_frequency` + `handle_unknown` | Section 5 |
| Numerical safety | Guarded denominators, bounded ratios | Section 4; `tests/test_features.py` |
| Reproducible | `random_state=42`, splits materialised to parquet | Section 7 |

**Next:** [`04_model_training.ipynb`](04_model_training.ipynb)
"""),
    ])


# --------------------------------------------------------------------------- #
def notebook_04() -> nbf.NotebookNode:
    return _notebook([
        md("""
# 04 — Model Training and Hyper-parameter Tuning

> **This notebook loads the results of the hyper-parameter search rather than
> re-running it.** The search takes roughly an hour on 16 cores, and a notebook
> that takes an hour to run is a notebook nobody runs. The exact command that
> produced these artefacts is shown below, and re-running it reproduces them
> because every random state is fixed.

```bash
python -m src.train              # the main experiment
python -m src.train --ablation no_ttl
```
"""),
        code(HEADER),
        md("""
## 1. The models and why each is here
"""),
        code("""
from src import train as train_module

registry = train_module.model_registry()
for key, spec in registry.items():
    print(f"=== {spec.display_name} ===")
    print(f"  tags       : {', '.join(spec.tags)}")
    print(f"  candidates : {spec.n_iter} x {config.CV_FOLDS} folds")
    print(f"  rationale  : {spec.notes}")
    print()
"""),
        code("""
print("Search spaces:\\n")
for key, spec in registry.items():
    print(f"{spec.display_name}")
    for param, distribution in spec.param_distributions.items():
        name = param.removeprefix("model__")
        kind = (f"{type(distribution).__name__}" if hasattr(distribution, "rvs")
                else str(distribution))
        print(f"    {name:<22} {kind}")
    print()
"""),
        md("""
### Isolation Forest — the unsupervised comparison

Fitted on **benign training flows only**, with the attack labels withheld
entirely. That is the honest framing of anomaly detection for intrusion
detection: show the detector what normal looks like and make it flag everything
else.

It answers a question the supervised models cannot — *how far could we get with
no attack labels at all?* — and sets the floor that supervised learning has to
beat to justify its labelling cost.
"""),
        md("""
## 2. Tuning protocol

- `RandomizedSearchCV` with 5-fold `StratifiedKFold`, `random_state=42`
- Scoring on **F1**, with average precision and ROC-AUC recorded alongside
- Each candidate is a **full `Pipeline`** — feature engineering → preprocessing →
  estimator

That last point is what makes fold-level leakage impossible. Every fold re-fits
the scaler and the one-hot vocabulary on its own training portion. A scaler
fitted once before cross-validation would leak fold information into the
validation folds.
"""),
        code("""
import json

params_path = config.MODELS_DIR / "best_params_main.json"
if not params_path.exists():
    display(Markdown(
        "> ⚠️ No tuning results found. Run `python -m src.train` first."))
    results = {}
else:
    results = json.loads(params_path.read_text(encoding="utf-8"))
    rows = []
    for key, record in results.items():
        rows.append({
            "model": record.get("display_name", key),
            "cv_f1_mean": record.get("cv_f1_mean"),
            "cv_f1_std": record.get("cv_f1_std"),
            "cv_pr_auc": record.get("cv_average_precision_mean"),
            "cv_roc_auc": record.get("cv_roc_auc_mean"),
            "train_minus_test_f1": record.get("cv_train_minus_test_f1"),
            "search_s": record.get("search_seconds"),
            "fit_s": record.get("fit_seconds"),
        })
    display(pd.DataFrame(rows).set_index("model"))
"""),
        md("""
The `train_minus_test_f1` column is the gap between a model's score on data it
was fitted on and its cross-validated score. A large positive gap indicates
memorisation. Tree ensembles are expected to show one; what matters is that the
cross-validated score and the held-out test score agree — which notebook 05
confirms.
"""),
        code("""
for key, record in results.items():
    if "best_params" in record:
        print(f"{record.get('display_name', key)}:")
        for param, value in record["best_params"].items():
            formatted = f"{value:.6g}" if isinstance(value, float) else value
            print(f"    {param:<22} {formatted}")
        print()
"""),
        md("""
## 3. Solver selection for Logistic Regression — decided by measurement

The default solver was not assumed. Four combinations were benchmarked on the
92,210 × 73 training matrix:

| Solver | Penalty | Time | Validation F1 | Converged |
|---|---|---|---|---|
| `lbfgs` | L2 | 3.6 s | 0.8566 | yes |
| `liblinear` | L2 | 10.0 s | 0.8560 | yes |
| `saga` | L2 | 101.8 s | 0.8561 | yes |
| `saga` | L1 | 199.4 s | 0.8562 | **no** |

L1 was dropped: 55× the runtime for a 0.0004 F1 difference, and it hit the
iteration cap. The cause is instructive — the TTL features make the classes very
nearly linearly separable, so the unregularised optimum runs off toward infinite
coefficients and the solver never settles.
"""),
        md("## 4. Persisted artefacts"),
        code("""
for path in sorted(config.MODELS_DIR.glob("*")):
    size = path.stat().st_size
    unit = f"{size/1e6:.1f} MB" if size > 1e6 else f"{size/1e3:.1f} KB"
    print(f"  {path.name:<42} {unit:>10}")
"""),
        code("""
manifest_path = config.MODELS_DIR / "manifest_main.json"
if manifest_path.exists():
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in manifest.items():
        print(f"  {key:<28} {value}")
"""),
        md("""
Each `.joblib` is a **complete `Pipeline`** — feature engineering, preprocessing
and estimator travel together. A caller passes raw UNSW-NB15-shaped records and
never has to reproduce any transformation, which removes the most common cause
of training/serving skew.
"""),
        md("## 5. Validation performance (test split still untouched)"),
        code("""
validation_path = config.METRICS_DIR / "validation_metrics_main.csv"
if validation_path.exists():
    validation = pd.read_csv(validation_path, index_col=0)
    columns = ["display_name", "recall", "precision", "f1", "roc_auc",
               "pr_auc", "false_positive_rate", "false_negative_rate"]
    display(validation[[c for c in columns if c in validation.columns]]
            .style.format({c: "{:.4f}" for c in columns[1:]}))
else:
    display(Markdown("> Run `python -m src.pipeline --evaluate` to populate this."))
"""),
        md("""
**These are validation numbers.** They are what model selection and threshold
tuning are allowed to use. The test split is opened once, in notebook 05, after
both choices are frozen.

**Next:** [`05_model_evaluation.ipynb`](05_model_evaluation.ipynb)
"""),
    ])


# --------------------------------------------------------------------------- #
def notebook_05() -> nbf.NotebookNode:
    return _notebook([
        md("""
# 05 — Model Evaluation

The model and its decision threshold were both frozen using the **validation**
split (notebook 04). This notebook opens the **test** split and scores it once.
Nothing here changes a parameter.

```bash
python -m src.pipeline --evaluate
python -m src.pipeline --ablations
```
"""),
        code(HEADER),
        code(SHOW_FIGURE),
        code("""
import json

deployment_path = config.MODELS_DIR / "deployment.json"
if deployment_path.exists():
    deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
    print("FROZEN DEPLOYMENT CONFIGURATION")
    print("=" * 60)
    for key, value in deployment.items():
        print(f"  {key:<22} {value}")
else:
    deployment = {}
    display(Markdown("> ⚠️ Run `python -m src.pipeline --evaluate` first."))
"""),
        md("""
## 1. Test-split results

Every metric the rubric asks for, plus MCC, balanced accuracy and Brier score.
"""),
        code("""
test_path = config.METRICS_DIR / "test_metrics_main.csv"
test = pd.read_csv(test_path, index_col=0) if test_path.exists() else None

if test is not None:
    headline = ["display_name", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    display(Markdown("### Detection quality"))
    display(test[[c for c in headline if c in test.columns]]
            .style.format({c: "{:.4f}" for c in headline[1:]}))

    operational = ["display_name", "false_positive_rate", "false_negative_rate",
                   "tp", "fn", "fp", "tn"]
    display(Markdown("### Operational error rates and confusion counts"))
    display(test[[c for c in operational if c in test.columns]]
            .style.format({"false_positive_rate": "{:.3%}",
                           "false_negative_rate": "{:.3%}",
                           "tp": "{:,.0f}", "fn": "{:,.0f}",
                           "fp": "{:,.0f}", "tn": "{:,.0f}"}))

    cost = ["display_name", "fit_seconds", "search_seconds",
            "throughput_flows_per_second", "inference_ms_per_1k_rows"]
    display(Markdown("### Computational cost"))
    display(test[[c for c in cost if c in test.columns]])
"""),
        md("""
### Why accuracy is reported but never used to decide anything

On a corpus where attacks are ~44% of records, accuracy compresses every model
into a narrow band and rewards majority-class performance. The metrics that
carry the decision are:

- **Recall** — what fraction of intrusions do we catch?
- **FNR** — its complement, and the number that maps to residual security risk.
- **FPR** — analyst workload, and the *only* workload metric that transfers to a
  different base rate.
- **PR-AUC** — threshold-free ranking quality on the positive class.
"""),
        md("## 2. Confusion matrices"),
        code('show("fig14_confusion_matrices")'),
        md("""
The bottom-left cell of each matrix is the count of intrusions that reached the
network unchallenged. That cell, not the diagonal, is what a SOC lead reads
first.
"""),
        md("## 3. ROC and Precision-Recall curves"),
        code('show("fig15_roc_pr_curves")'),
        md("""
**The PR curve is the more informative of the two here.** ROC uses the
false-positive *rate*, which stays flattering when negatives are plentiful.
Precision responds directly to the alert volume an analyst must work through.

Both are computed at this corpus's 44% attack base rate. Precision — unlike
recall and FPR — will fall sharply at a realistic production base rate.
"""),
        md("## 4. Model comparison"),
        code('show("fig16_model_comparison")'),
        code("""
if deployment:
    display(Markdown(
        f"### Selected: **{deployment['display_name']}**\\n\\n"
        f"{deployment['selection_reason']}"))
"""),
        md("""
**Selection used validation PR-AUC — deliberately not accuracy.** PR-AUC measures
how well a model *ranks* attacks above benign flows across all thresholds, on the
positive class only. That is the quantity that survives a change of base rate,
and the one that determines whether a useful operating point exists at all.
"""),
        md("""
## 5. Threshold analysis — 0.50 is a default, not an optimum

The sweep below was computed on the **validation** split. Choosing a threshold is
a fitting decision; doing it on test would make the reported test metrics
optimistic.
"""),
        code('show("fig17_threshold_analysis")'),
        code("""
recommendation_path = config.METRICS_DIR / "threshold_recommendation_main.json"
if recommendation_path.exists():
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    rows = []
    for name in ("max_f1", "min_cost", "fpr_constrained"):
        if name in recommendation:
            rows.append({"operating_point": name, **recommendation[name]})
    display(pd.DataFrame(rows).set_index("operating_point")
            .style.format({"threshold": "{:.3f}", "recall": "{:.4f}",
                           "precision": "{:.4f}", "f1": "{:.4f}",
                           "false_positive_rate": "{:.3%}",
                           "false_negative_rate": "{:.3%}"}))

    display(Markdown("**Sensitivity of the cost-optimal threshold to the cost assumption**"))
    display(pd.DataFrame(recommendation["cost_ratio_sensitivity"]).T
            .style.format({"threshold": "{:.3f}", "recall": "{:.4f}",
                           "false_positive_rate": "{:.3%}"}))
"""),
        md("""
The 20:1 false-negative-to-false-positive cost ratio is an **assumption, stated
openly, not a measurement**. Published breach-cost studies imply ratios in the
hundreds — but a detector tuned at that ratio alerts on nearly everything and
destroys itself through alert fatigue, so 20:1 is used as a deliberately
conservative working figure with its sensitivity reported.

**Which point should a SOC run?** A team with analyst capacity to spare should
take the F1-optimal point. A team already at queue capacity should take the
FPR-constrained point and pair the model with compensating controls. This is a
business decision about alert budget, not a modelling one.
"""),
        code("""
operating_path = config.METRICS_DIR / "operating_points_main.json"
if operating_path.exists():
    points = json.loads(operating_path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(points).T[
        ["threshold", "recall", "precision", "f1",
         "false_positive_rate", "false_negative_rate", "fn", "fp"]]
    display(Markdown("### The selected model at each operating point — on the TEST split"))
    display(frame.style.format({
        "threshold": "{:.3f}", "recall": "{:.4f}", "precision": "{:.4f}",
        "f1": "{:.4f}", "false_positive_rate": "{:.3%}",
        "false_negative_rate": "{:.3%}", "fn": "{:,.0f}", "fp": "{:,.0f}"}))
"""),
        md("""
## 6. Ablations — what is the headline number actually measuring?

Each row is a **separate end-to-end experiment** with its own split, reusing the
hyper-parameters tuned in the main run so that exactly one thing changes at a
time.
"""),
        code('show("fig19_ablation_comparison")'),
        code("""
ablation_path = config.METRICS_DIR / "ablation_summary.csv"
if ablation_path.exists() and deployment:
    ablations = pd.read_csv(ablation_path, index_col=[0, 1])
    best = deployment["model_key"]
    if best in ablations.index.get_level_values(1):
        sub = ablations.xs(best, level=1)[
            ["recall", "precision", "f1", "pr_auc",
             "false_positive_rate", "false_negative_rate"]]
        display(sub.style.format({
            "recall": "{:.4f}", "precision": "{:.4f}", "f1": "{:.4f}",
            "pr_auc": "{:.4f}", "false_positive_rate": "{:.3%}",
            "false_negative_rate": "{:.3%}"}))
else:
    display(Markdown("> Run `python -m src.pipeline --ablations` to populate this."))
"""),
        md("""
### How to read this table

| Experiment | The question it answers |
|---|---|
| `keep_duplicates` | How much are published UNSW-NB15 benchmarks inflated by repeated records? |
| `no_ttl` | How much of the score is *testbed recognition* rather than *attack detection*? |
| `no_engineered` | What did the 15 constructed features actually contribute? |
| `smote` | Does synthetic oversampling beat class weighting here, or not? |
| `official_split` | Does the result hold under the authors' own partition? |

This table is what separates "the model detects attacks" from "the model detects
this dataset."
"""),
        md("""
---

**Next:** [`06_explainability_bias_audit.ipynb`](06_explainability_bias_audit.ipynb)
"""),
    ])


# --------------------------------------------------------------------------- #
def notebook_06() -> nbf.NotebookNode:
    return _notebook([
        md("""
# 06 — Explainability, Error Analysis and Bias Audit

Three questions:

1. **Why** does the model decide what it decides — globally and for one flow?
2. **What** does it get wrong, and is the error concentrated anywhere?
3. **Is the audit fair** — and what can this dataset actually support?

> Explaining a model is not fitting it. These analyses run on the test split
> legitimately, because the model and threshold were frozen before it was opened
> and nothing here feeds back into either.

```bash
python -m src.pipeline --evaluate
```
"""),
        code(HEADER),
        code(SHOW_FIGURE),
        code("""
import json

deployment_path = config.MODELS_DIR / "deployment.json"
deployment = (json.loads(deployment_path.read_text(encoding="utf-8"))
              if deployment_path.exists() else {})
if deployment:
    print(f"Explaining : {deployment['display_name']}")
    print(f"Threshold  : {deployment['threshold']:.3f}")
"""),
        md("""
## 1. Global SHAP importance

SHAP is used in preference to impurity-based importance, which is biased toward
high-cardinality features and splits credit arbitrarily between the correlated
feature twins identified in notebook 02.
"""),
        code('show("fig20_shap_global_importance")'),
        code("""
shap_path = config.METRICS_DIR / "shap_analysis.json"
shap_payload = json.loads(shap_path.read_text(encoding="utf-8")) if shap_path.exists() else None

if shap_payload:
    importance = pd.DataFrame(shap_payload["global_importance"]).head(15)
    importance["direction"] = np.where(importance["mean_shap"] > 0,
                                       "toward ATTACK", "toward BENIGN")
    display(importance[["feature", "mean_abs_shap", "share_of_total", "direction"]]
            .style.format({"mean_abs_shap": "{:.4f}", "share_of_total": "{:.1%}"}))
else:
    display(Markdown("> Run `python -m src.pipeline --evaluate` first."))
"""),
        md("## 2. Beeswarm — distribution and direction of every effect"),
        code('show("fig21_shap_beeswarm")'),
        md("""
Each dot is one test flow. Horizontal position is that feature's contribution to
the attack score for that flow; colour is the feature's own standardised value.

A feature whose red points sit on the right means **high values of it argue for
"attack"**. The width of each row shows how much a feature's influence varies
between flows — a narrow row is a feature the model treats almost identically
everywhere.
"""),
        md("## 3. Dependence — how a feature's value maps to its effect"),
        code('show("fig22_shap_dependence")'),
        md("""
A clean step shape means the model learned a **threshold rule**. Vertical spread
at a fixed x-value is **interaction**: the same feature value means different
things depending on the rest of the flow.
"""),
        md("""
## 4. The artefact diagnostic

This is the analysis that decides whether the headline performance should be read
as *attack detection* or as *testbed recognition*.
"""),
        code('show("fig24_shap_artifact_check")'),
        code("""
if shap_payload and "artifact_diagnostic" in shap_payload:
    diagnostic = shap_payload["artifact_diagnostic"]
    shares = pd.Series(diagnostic["impact_share_by_group"]).sort_values(ascending=False)
    shares.index = [i.replace("\\n", " ") for i in shares.index]
    display(shares.to_frame("share_of_total_impact").style.format("{:.1%}"))
    print(f"\\nTTL family share of total attributed impact: "
          f"{diagnostic['ttl_family_share']:.1%}")
"""),
        md("""
### Interpretation

The EDA raised a suspicion; SHAP converts it into a measured quantity **on the
fitted model**. The dependence plot shows the relationship is a *step*, not a
gradient — the model has learned a near-binary switch on a field the UNSW-NB15
testbed happened to configure differently for its benign and attack generators.

**The response is measurement, not deletion.** The features stay (they are real
fields a sensor observes), and the `no_ttl` ablation in notebook 05 shows what
performance survives without them.
"""),
        md("""
## 5. Local explanations

Four cases, each chosen as the **median-scoring** member of its outcome class so
the explanation describes typical model behaviour rather than a cherry-picked
extreme.
"""),
        code('show("fig23_shap_local_cases")'),
        code("""
if shap_payload and "local_cases" in shap_payload:
    for title, case in shap_payload["local_cases"].items():
        print(f"=== {title} ===")
        print(f"  score={case['attack_score']:.4f}  true={case['true_label']}  "
              f"pred={case['predicted']}  family={case['attack_cat']}")
        print(f"  proto={case['proto']}  service={case['service']}  state={case['state']}")
        for contribution in case["top_contributions"][:5]:
            arrow = "-> ATTACK" if contribution["shap"] > 0 else "-> BENIGN"
            print(f"      {contribution['feature']:<26} "
                  f"{contribution['shap']:+.4f}  {arrow}")
        print()
"""),
        md("""
Comparing the **false negative** against the **true positive** shows exactly
which evidence was absent in the attack the model let through. That comparison is
the raw material for the detection-gap discussion below.
"""),
        md("""
## 6. Operational subgroup audit

### What kind of audit this is — and what it is not

**UNSW-NB15 contains no demographic attributes and no human subjects.** It is
synthetic traffic generated on a closed testbed; the partitioned files omit every
IP address, port number and timestamp, so there is not even an indirect
identifier from which a protected characteristic could be inferred.

**Therefore no demographic fairness claim is made, and none can be made from this
dataset.** What is performed instead is an **operational performance audit**
across the strata that genuinely exist: attack family, application service,
transport protocol and connection state.
"""),
        code('show("fig18_subgroup_audit")'),
        code("""
audit_path = config.METRICS_DIR / "subgroup_audit_main.csv"
if audit_path.exists():
    audit = pd.read_csv(audit_path)

    families = audit[audit["group_type"] == "attack_cat"]
    families = families[families["n_attack"] > 0].sort_values("recall")
    display(Markdown("### Recall by attack family"))
    display(families[["group", "n_attack", "recall", "false_negative_rate"]]
            .set_index("group")
            .style.format({"n_attack": "{:,.0f}", "recall": "{:.4f}",
                           "false_negative_rate": "{:.4f}"}))
"""),
        code("""
if audit_path.exists():
    for group_type in ("service", "proto", "state"):
        sub = audit[(audit["group_type"] == group_type) & (audit["n_benign"] >= 30)]
        sub = sub.sort_values("false_positive_rate", ascending=False).head(8)
        if sub.empty:
            continue
        display(Markdown(f"### {group_type} — where false alerts concentrate"))
        display(sub[["group", "n", "n_benign", "recall", "precision",
                     "false_positive_rate", "f1"]].set_index("group")
                .style.format({"n": "{:,.0f}", "n_benign": "{:,.0f}",
                               "recall": "{:.4f}", "precision": "{:.4f}",
                               "false_positive_rate": "{:.3%}", "f1": "{:.4f}"}))
"""),
        md("""
### Interpretation

**Recall is not uniform.** The weakest families are the stealthy, low-volume ones
a defender would most want caught. Three distinguishable mechanisms drive this,
and they are *not* equivalent problems:

1. **Scarcity** — few training examples. Fixable with more data.
2. **Behavioural overlap with benign traffic** — a backdoor that looked like a
   port scan would be a bad backdoor. **Not fixable**, and the more serious
   finding, because it bounds what *any* flow-level detector can achieve.
3. **Duplication-distorted training counts** — families that were heavily
   duplicated contributed far fewer distinct examples than their raw counts
   suggested.

**False alerts are not spread evenly either.** A handful of services generate a
disproportionate share — which is directly actionable through per-service
thresholds, and is also a warning: a stratum with an elevated FPR is a stratum
where analysts learn to dismiss alerts, converting a precision problem into a
recall problem through human behaviour.
"""),
        md("## 7. Error analysis"),
        code("""
errors_path = config.METRICS_DIR / "error_analysis_main.json"
if errors_path.exists():
    errors = json.loads(errors_path.read_text(encoding="utf-8"))

    print("Outcome counts:")
    for outcome, count in errors["counts"].items():
        print(f"  {outcome:<18} {count:>8,}")

    if "false_negative_score_stats" in errors:
        stats = errors["false_negative_score_stats"]
        print(f"\\nMissed attacks — median score {stats['median']:.4f}, "
              f"{stats['share_below_0.10']:.1%} score below 0.10")
    if "false_positive_score_stats" in errors:
        stats = errors["false_positive_score_stats"]
        print(f"False alerts   — median score {stats['median']:.4f}, "
              f"{stats['share_above_0.90']:.1%} score above 0.90")

    display(Markdown("### Median flow profile by outcome"))
    display(pd.DataFrame(errors["median_profile_by_outcome"]).T)
"""),
        md("""
### Near-misses or confident errors?

This distinction decides what you can *do* about the errors, so it is measured
rather than assumed:

* Errors clustered just the wrong side of the threshold are **near-misses**.
  Moving the threshold trades them directly against the other error type, and a
  "review the borderline alerts" workflow genuinely catches them.
* Errors scored far from the threshold are **confident mistakes**. No threshold
  change and no triage queue recovers those.

Compare the median false-negative score printed above against the deployed
threshold. Comparing the median flow profiles then shows what a missed attack
looks like — and it looks like ordinary traffic, which is precisely the point.
"""),
        md("""
## 8. Where genuine fairness risk does enter

The dataset has no protected attributes, but a **deployed** system runs on a real
network used by real people:

1. **Uneven burden across teams.** Over-flagging a service means the humans who
   use it are investigated more often. Research groups, developers and network
   engineers whose legitimate work looks statistically unusual absorb
   disproportionate scrutiny. Section 6 shows this is already true across
   services.
2. **Assistive technologies.** Screen readers and alternative input devices can
   produce atypical traffic patterns.
3. **Feedback loops.** Flagged users are monitored more closely → more of their
   activity is labelled suspicious → retraining amplifies the original skew.
4. **Automation bias.** An analyst shown a 0.94 score is measurably more likely
   to confirm it than to challenge it.

These are **deployment-time risks**, not properties of UNSW-NB15 — and a bias
audit that stopped at the dataset would miss them entirely.

---

## Summary

| Question | Answer |
|---|---|
| What makes traffic look malicious? | High source TTL, unanswered connection, no completed handshake, asymmetric direction, elevated connection counters |
| What makes it look benign? | Completed handshake with sequence exchange, balanced bidirectional volume, identified service |
| Is a proxy feature dominating? | **Yes** — the TTL family, named and quantified rather than hidden |
| Is detection uniform? | **No** — stealthy low-volume families are substantially weaker |
| Are false alerts uniform? | **No** — they concentrate in specific services |
| Can we claim demographic fairness? | **No** — and we do not. The data cannot support it |

Full analysis:
[`reports/Bias_Fairness_Analysis.md`](../reports/Bias_Fairness_Analysis.md).
"""),
    ])


# --------------------------------------------------------------------------- #
BUILDERS = {
    "01_problem_data_understanding.ipynb": notebook_01,
    "02_eda.ipynb": notebook_02,
    "03_preprocessing_feature_engineering.ipynb": notebook_03,
    "04_model_training.ipynb": notebook_04,
    "05_model_evaluation.ipynb": notebook_05,
    "06_explainability_bias_audit.ipynb": notebook_06,
}


def build(execute: bool = False) -> list[Path]:
    config.NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, builder in BUILDERS.items():
        path = config.NOTEBOOKS_DIR / name
        nbf.write(builder(), path)
        print(f"[notebooks] wrote notebooks/{name}")
        written.append(path)

    if execute:
        written = [execute_notebook(p) for p in written]
    return written


def execute_notebook(path: Path, timeout: int = 1800) -> Path:
    """Run a notebook in place, so committed notebooks carry real outputs."""
    from nbclient import NotebookClient

    notebook = nbf.read(path, as_version=4)
    client = NotebookClient(
        notebook, timeout=timeout, kernel_name="python3",
        resources={"metadata": {"path": str(config.PROJECT_ROOT)}},
        allow_errors=False,
    )
    print(f"[notebooks] executing {path.name} ...")
    client.execute()
    nbf.write(notebook, path)
    print(f"[notebooks]   ok: {path.name}")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the analysis notebooks.")
    parser.add_argument("--execute", action="store_true",
                        help="run each notebook after writing it")
    args = parser.parse_args(argv)
    build(execute=args.execute)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
