"""
Exploratory data analysis: every figure in ``figures/`` and the statistics that
back ``reports/EDA_Feature_Engineering_Report.md``.

Scope discipline
----------------
Two kinds of analysis appear here and they are deliberately kept apart:

* **Corpus-level bookkeeping** (row counts, class balance, duplication,
  missingness) describes the *dataset as published* and is computed on the whole
  corpus. Knowing how many rows a public dataset has is not test-set peeking.
* **Distributional and relational EDA** - anything that could inform a
  modelling choice - is computed on the **training split only**. The validation
  and test splits are never inspected here. This is why, for example, the
  correlation heatmap and the feature-ranking chart are labelled "training
  split, n=92,210".

Run with::

    python -m src.eda
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src import config, plotting, preprocessing
from src.features import ENGINEERED_FEATURE_DOCS, add_engineered_features

ATTACK_ORDER = [
    "Normal", "Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
    "Analysis", "Backdoor", "Shellcode", "Worms",
]


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def fig_class_distribution(corpus: pd.DataFrame, dedup: pd.DataFrame) -> None:
    """Binary class balance before and after deduplication."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, frame, title in (
        (axes[0], corpus, f"As published (n={len(corpus):,})"),
        (axes[1], dedup, f"After deduplication (n={len(dedup):,})"),
    ):
        counts = frame[config.TARGET].value_counts().sort_index()
        bars = ax.bar(["Benign\n(label=0)", "Attack\n(label=1)"],
                      counts.values, color=plotting.class_colors(), width=0.6)
        ax.set_ylim(0, counts.max() * 1.20)
        plotting.annotate_bars(
            ax, bars,
            [f"{v:,}\n{v / len(frame):.1%}" for v in counts.values])
        ax.set_title(title)
        ax.set_ylabel("Flow records")
        plotting.despine(ax)

    fig.suptitle("Class balance shifts materially once duplicate flows are removed")
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
    plotting.caption(fig, (
        "Removing exact duplicate feature vectors turns a 64% attack-majority corpus into a "
        "56%/44% benign-majority one, because duplication is concentrated in the attack classes. "
        "Neither balance resembles a production network, where attacks are a small minority."
    ))
    plotting.save(fig, "fig01_class_distribution")


def fig_attack_category_distribution(dedup: pd.DataFrame) -> None:
    """Attack-family counts on a log scale (three orders of magnitude)."""
    counts = dedup[config.ATTACK_CAT].value_counts().reindex(ATTACK_ORDER).dropna()
    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [config.COLOR_BENIGN if c == "Normal" else config.COLOR_ATTACK
              for c in counts.index]
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], height=0.7)
    ax.set_xscale("log")
    ax.set_xlim(50, counts.max() * 3)
    ax.set_xlabel("Flow records (log scale)")
    plotting.annotate_bars(
        ax, bars, [f"{v:,}" for v in counts.values[::-1]], horizontal=True, pad=0.02)
    ax.set_title("Attack families span three orders of magnitude")
    ax.legend(handles=[Patch(color=config.COLOR_BENIGN, label="Benign"),
                       Patch(color=config.COLOR_ATTACK, label="Attack family")],
              loc="lower right")
    plotting.despine(ax)
    plotting.caption(fig, (
        "Worms (164 records) is 520x rarer than Normal. Any multiclass extension - and the "
        "subgroup recall audit in Phase 9 - must expect wide confidence intervals on the tail "
        "families, and per-family recall is reported with counts for exactly this reason."
    ))
    plotting.save(fig, "fig02_attack_category_distribution")


def fig_duplication(corpus: pd.DataFrame) -> None:
    """Duplication rate by attack family - the justification for deduplicating."""
    predictors = [c for c in corpus.columns
                  if c not in (*config.LEAKAGE_COLUMNS, "partition")]
    dup = corpus.duplicated(subset=predictors, keep="first")
    stats = (corpus.assign(_dup=dup)
             .groupby(config.ATTACK_CAT, observed=True)["_dup"]
             .agg(["size", "mean"])
             .reindex(ATTACK_ORDER).dropna()
             .sort_values("mean"))

    fig, ax = plt.subplots(figsize=(9, 4.6))
    colors = [plotting.rate_to_color(r) for r in stats["mean"]]
    bars = ax.barh(stats.index, stats["mean"] * 100, color=colors,
                   edgecolor="#666666", linewidth=0.5, height=0.7)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of records that duplicate an earlier record (%)")
    plotting.annotate_bars(
        ax, bars,
        [f"{m:.1%}  (n={int(n):,})" for m, n in zip(stats["mean"], stats["size"])],
        horizontal=True, pad=0.012)
    ax.set_title("Duplication is severe and strongly class-dependent")
    plotting.despine(ax)
    plotting.caption(fig, (
        "87.6% of Generic records and 74-75% of DoS/Backdoor records are exact copies of an "
        "earlier flow, against only 8.1% of Normal records. A random split therefore places "
        "identical attack vectors in both train and test, inflating scores. This is why the "
        "corpus is deduplicated BEFORE splitting (see the keep_duplicates ablation)."
    ))
    plotting.save(fig, "fig03_duplication_by_attack_category")


def _rate_panel(ax, frame, column, min_count, title, xlabel):
    stats = (frame.groupby(column, observed=True)[config.TARGET]
             .agg(["size", "mean"]))
    stats = stats[stats["size"] >= min_count].sort_values("mean")
    colors = [plotting.rate_to_color(r) for r in stats["mean"]]
    bars = ax.barh(stats.index.astype(str), stats["mean"] * 100, color=colors,
                   edgecolor="#666666", linewidth=0.5, height=0.72)
    ax.axvline(frame[config.TARGET].mean() * 100, color=config.COLOR_NEUTRAL,
               linestyle="--", linewidth=1.2, zorder=3)
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    plotting.annotate_bars(
        ax, bars,
        [f"{m:.0%} (n={int(n):,})" for m, n in zip(stats["mean"], stats["size"])],
        horizontal=True, pad=0.010, fontsize=8)
    plotting.despine(ax)
    return stats


def fig_categorical_attack_rates(train: pd.DataFrame) -> dict:
    """Attack rate by protocol, service and connection state."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.6))
    proto = _rate_panel(axes[0], train, "proto", 150,
                        "Transport / IP protocol", "Attack rate (%)")
    service = _rate_panel(axes[1], train, "service", 30,
                          "Application-layer service", "Attack rate (%)")
    state = _rate_panel(axes[2], train, "state", 30,
                        "Connection state", "Attack rate (%)")
    axes[0].legend(handles=[Line2D([0], [0], color=config.COLOR_NEUTRAL,
                                   linestyle="--", label="Corpus base rate")],
                   loc="lower right")

    # Quantify the artefact across ALL protocols, not just the frequent ones the
    # left-hand panel has room to show.
    all_protocols = train.groupby("proto", observed=True)[config.TARGET].agg(["size", "mean"])
    eligible = all_protocols[all_protocols["size"] >= 20]
    pure_attack = int((eligible["mean"] == 1.0).sum())
    pure_benign = int((eligible["mean"] == 0.0).sum())

    fig.suptitle("Categorical fields are extremely discriminative - "
                 "partly for the wrong reason (training split, n={:,})".format(len(train)))
    plotting.caption(fig, (
        f"The left panel shows only protocols with at least 150 training records. Across all "
        f"protocols with at least 20 records, {pure_attack} are 100% malicious and {pure_benign} "
        "are 100% benign. That is not a property of those protocols in the real world; it is an "
        "artefact of the UNSW-NB15 testbed, where the attack generator emitted exotic IP "
        "protocols that the background-traffic generator never used. Protocols below the "
        f"{config.MIN_CATEGORY_FREQUENCY}-record encoding cutoff are pooled into one 'uncommon "
        "protocol' indicator so the model learns a generalisable property rather than memorising "
        "which protocol numbers a 2015 scanner happened to sweep."
    ), y=-0.03)
    plotting.save(fig, "fig04_categorical_attack_rates")
    return {"proto": proto.to_dict("index"),
            "service": service.to_dict("index"),
            "state": state.to_dict("index")}


def fig_ttl_artifact(train: pd.DataFrame) -> dict:
    """The TTL artefact: sttl is a near-perfect label proxy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    # Panel 1: sttl composition
    top = train["sttl"].value_counts().head(6).index.tolist()
    sub = train[train["sttl"].isin(top)]
    tab = pd.crosstab(sub["sttl"], sub[config.TARGET]).reindex(sorted(top))
    idx = np.arange(len(tab))
    axes[0].bar(idx, tab.get(0, 0), color=config.COLOR_BENIGN, label="Benign", width=0.7)
    axes[0].bar(idx, tab.get(1, 0), bottom=tab.get(0, 0),
                color=config.COLOR_ATTACK, label="Attack", width=0.7)
    axes[0].set_xticks(idx)
    axes[0].set_xticklabels(tab.index.astype(str))
    axes[0].set_xlabel("Source TTL (sttl)")
    axes[0].set_ylabel("Flow records")
    axes[0].set_title("Source TTL is near-deterministic for most values")
    axes[0].legend()
    plotting.despine(axes[0])

    # Panel 2: attack rate per sttl value
    rate = (train.groupby("sttl", observed=True)[config.TARGET]
            .agg(["size", "mean"]).query("size >= 20").sort_index())
    axes[1].scatter(rate.index, rate["mean"] * 100,
                    s=np.sqrt(rate["size"]) * 3.2,
                    color=[plotting.rate_to_color(r) for r in rate["mean"]],
                    edgecolor="#444444", linewidth=0.6, zorder=3)
    axes[1].axhline(train[config.TARGET].mean() * 100, color=config.COLOR_NEUTRAL,
                    linestyle="--", linewidth=1.2)
    axes[1].set_xlabel("Source TTL value")
    axes[1].set_ylabel("Attack rate (%)")
    axes[1].set_ylim(-6, 106)
    axes[1].set_title("Almost every TTL is ~0% or ~100% malicious")
    plotting.despine(axes[1])

    # Panel 3: ct_state_ttl, the authors' derived feature
    tab3 = pd.crosstab(train["ct_state_ttl"], train[config.TARGET])
    idx3 = np.arange(len(tab3))
    axes[2].bar(idx3, tab3.get(0, 0), color=config.COLOR_BENIGN, label="Benign", width=0.7)
    axes[2].bar(idx3, tab3.get(1, 0), bottom=tab3.get(0, 0),
                color=config.COLOR_ATTACK, label="Attack", width=0.7)
    axes[2].set_xticks(idx3)
    axes[2].set_xticklabels(tab3.index.astype(str))
    axes[2].set_xlabel("ct_state_ttl")
    axes[2].set_ylabel("Flow records")
    axes[2].set_title("ct_state_ttl inherits the same artefact")
    axes[2].legend()
    plotting.despine(axes[2])

    # A one-feature "model": memorise the majority class of each sttl value.
    lookup = train.groupby("sttl", observed=True)[config.TARGET].mean()
    rule_acc = float(((train["sttl"].map(lookup) > 0.5).astype(int)
                      == train[config.TARGET]).mean())
    majority = float(max(train[config.TARGET].mean(), 1 - train[config.TARGET].mean()))

    fig.suptitle("The single biggest threat to this project's external validity")
    plotting.caption(fig, (
        f"A lookup rule that uses sttl and nothing else scores {rule_acc:.1%} accuracy on the "
        f"deduplicated training split, against a {majority:.1%} majority-class baseline - one "
        "column recovers most of the achievable separation. The UNSW-NB15 testbed ran its benign "
        "and attack generators on hosts configured with different initial TTLs, so sttl largely "
        "encodes 'which generator produced this flow', not 'is this flow hostile'. The TTL "
        "columns are RETAINED - they are real fields a sensor observes, and deleting real signal "
        "on suspicion would be its own error - but the no_ttl ablation in Phase 6 measures "
        "exactly how much of the headline score rests on them."
    ))
    plotting.save(fig, "fig05_ttl_artifact")

    return {"sttl_only_rule_accuracy_train": round(rule_acc, 4),
            "majority_class_baseline_train": round(majority, 4),
            "sttl_attack_rate": rate["mean"].round(4).to_dict()}


def fig_numeric_distributions(train: pd.DataFrame) -> None:
    """Class-conditional distributions of the core flow measurements."""
    columns = [
        ("dur", "Flow duration (s)"), ("sbytes", "Source bytes"),
        ("dbytes", "Destination bytes"), ("spkts", "Source packets"),
        ("dpkts", "Destination packets"), ("rate", "Packet rate (pkt/s)"),
        ("sload", "Source load (bit/s)"), ("dload", "Destination load (bit/s)"),
        ("smean", "Mean src packet size"), ("dmean", "Mean dst packet size"),
        ("sinpkt", "Src inter-packet time (ms)"), ("sjit", "Source jitter (ms)"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(15.5, 9.5))
    benign = train[train[config.TARGET] == 0]
    attack = train[train[config.TARGET] == 1]

    for ax, (col, label) in zip(axes.ravel(), columns):
        lo = np.log1p(np.clip(train[col].to_numpy(float), 0, None))
        bins = np.linspace(0, lo.max() if lo.max() > 0 else 1, 45)
        ax.hist(np.log1p(np.clip(benign[col].to_numpy(float), 0, None)), bins=bins,
                color=config.COLOR_BENIGN, alpha=0.62, label="Benign", density=True)
        ax.hist(np.log1p(np.clip(attack[col].to_numpy(float), 0, None)), bins=bins,
                color=config.COLOR_ATTACK, alpha=0.62, label="Attack", density=True)
        ax.set_title(label, fontsize=10)
        ax.set_xlabel("log(1 + value)", fontsize=8.5)
        ax.set_yticks([])
        plotting.despine(ax, keep=("bottom",))

    axes[0, 0].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Class-conditional flow distributions (training split, n={len(train):,}; "
                 "log1p axis)")
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    plotting.caption(fig, (
        "All twelve measurements are non-negative and span 6-9 orders of magnitude, which is why "
        "the pipeline applies log1p before standard-scaling. The separation is real but partial: "
        "attacks concentrate at short durations, small payloads and high packet rates - the shape "
        "of scanning and flooding - while benign traffic shows the long-tailed, bidirectional "
        "profile of genuine sessions. Note also how much the two densities OVERLAP on every "
        "single measurement: no one of these columns separates the classes alone, which is "
        "exactly why a model is needed rather than a threshold rule."
    ), y=0.0)
    plotting.save(fig, "fig06_numeric_distributions_by_class")


def fig_boxplots(train: pd.DataFrame) -> dict:
    """Boxplots plus a measurement of who actually occupies the distribution tail."""
    columns = ["dur", "sbytes", "dbytes", "rate", "sload", "flow_bytes_total"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, col in zip(axes.ravel(), columns):
        data = [np.log1p(np.clip(train.loc[train[config.TARGET] == k, col]
                                 .to_numpy(float), 0, None)) for k in (0, 1)]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55,
                        tick_labels=["Benign", "Attack"],
                        flierprops={"marker": ".", "markersize": 2,
                                    "markerfacecolor": "#999999",
                                    "markeredgecolor": "none", "alpha": 0.35},
                        medianprops={"color": "#111111", "linewidth": 1.6})
        for patch, color in zip(bp["boxes"], plotting.class_colors()):
            patch.set_facecolor(color)
            patch.set_alpha(0.65)
        ax.set_title(col, fontsize=10)
        ax.set_ylabel("log(1 + value)", fontsize=8.5)
        plotting.despine(ax)

    # Quantify the claim rather than asserting it: who actually occupies the tail?
    base_rate = float(train[config.TARGET].mean())
    tail_shares: dict[str, float] = {}
    for col in columns:
        cutoff = train[col].quantile(0.99)
        tail = train[train[col] >= cutoff]
        tail_shares[col] = float(tail[config.TARGET].mean())
    tail_lines = [f"{c} {tail_shares[c]:.0%}"
                  for c in ("sbytes", "flow_bytes_total", "rate")]

    fig.suptitle(f"Extreme values are genuine records, not sensor errors "
                 f"(training split, n={len(train):,})")
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    plotting.caption(fig, (
        "Each panel shows thousands of points beyond the whiskers, in BOTH classes - the tail is "
        "not an attack-only phenomenon. Inspecting the extremes shows genuine records: multi-"
        "megabyte transfers, gigabit-per-second loads, ten-thousand-packet bursts. Among the "
        f"top 1% of each measurement the attack share is {', '.join(tail_lines)}, against a "
        f"{base_rate:.0%} base rate - so the tail is informative but is not simply 'the attacks'. "
        "No outlier is removed anywhere in this project: in intrusion detection, clipping the "
        "tail discards real behaviour of both kinds, and for the measurements where the tail IS "
        "attack-heavy it would delete exactly the events an IDS exists to catch. Heavy tails are "
        "handled by log1p transformation instead."
    ), y=0.015)
    plotting.save(fig, "fig07_outlier_boxplots")
    return {"top1pct_attack_share": {k: round(v, 4) for k, v in tail_shares.items()},
            "base_rate_train": round(base_rate, 4)}


def fig_correlation(train: pd.DataFrame) -> dict:
    """Correlation heatmap over the original numeric features."""
    numeric = [c for c in train.columns
               if c not in (*config.LEAKAGE_COLUMNS, "partition")
               and c not in config.CATEGORICAL_FEATURES
               and c not in ENGINEERED_FEATURE_DOCS
               and pd.api.types.is_numeric_dtype(train[c])]
    corr = train[numeric].corr()

    fig, ax = plt.subplots(figsize=(12.5, 10.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(numeric)))
    ax.set_xticklabels(numeric, rotation=90, fontsize=7.5)
    ax.set_yticks(range(len(numeric)))
    ax.set_yticklabels(numeric, fontsize=7.5)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.72, pad=0.02)
    cbar.set_label("Pearson correlation", fontsize=9)
    ax.set_title("Redundancy among the 39 original numeric features "
                 f"(training split, n={len(train):,})")

    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).abs().stack()
    pairs = upper[upper > 0.90].sort_values(ascending=False)
    plotting.caption(fig, (
        f"{len(pairs)} feature pairs exceed |r| = 0.90. The tightest are structural rather than "
        "coincidental: sbytes/sloss (r=1.00) and dbytes/dloss (r=1.00) because loss counters are "
        "derived from byte counters, and is_ftp_login/ct_ftp_cmd (r=1.00) because they encode the "
        "same FTP event twice. Tree ensembles tolerate this but split importance gets diluted "
        "across correlated twins - which is why SHAP, not impurity gain, is used for attribution."
    ), y=0.0)
    plotting.save(fig, "fig08_correlation_heatmap")
    return {"pairs_above_090": {f"{a}|{b}": round(float(v), 4)
                                for (a, b), v in pairs.items()}}


def fig_feature_ranking(train: pd.DataFrame) -> dict:
    """Rank features by absolute point-biserial correlation with the label."""
    engineered = add_engineered_features(train)
    candidates = [c for c in engineered.columns
                  if c not in (*config.LEAKAGE_COLUMNS, "partition")
                  and pd.api.types.is_numeric_dtype(engineered[c])]
    corr = (engineered[candidates]
            .corrwith(engineered[config.TARGET])
            .abs().sort_values(ascending=False).head(25))

    fig, ax = plt.subplots(figsize=(9.5, 7))
    is_eng = [c in ENGINEERED_FEATURE_DOCS for c in corr.index]
    colors = [config.COLOR_ACCENT if e else config.COLOR_NEUTRAL for e in is_eng]
    bars = ax.barh(corr.index[::-1], corr.values[::-1], color=colors[::-1], height=0.72)
    ax.set_xlim(0, corr.max() * 1.22)
    ax.set_xlabel("|point-biserial correlation| with label")
    plotting.annotate_bars(ax, bars, [f"{v:.3f}" for v in corr.values[::-1]],
                           horizontal=True, pad=0.012)
    ax.set_title(f"Top 25 univariate signals (training split, n={len(train):,})")
    ax.legend(handles=[Patch(color=config.COLOR_ACCENT, label="Engineered in this project"),
                       Patch(color=config.COLOR_NEUTRAL, label="Original UNSW-NB15 feature")],
              loc="lower right")
    plotting.despine(ax)
    n_engineered = sum(is_eng)
    plotting.caption(fig, (
        f"{n_engineered} of the {len(ENGINEERED_FEATURE_DOCS)} engineered features enter the top "
        "25, confirming that the directionality and session-state constructions add univariate "
        "signal rather than merely restating existing columns. The three TTL-derived columns "
        "still occupy the top three places - the artefact documented in fig05. Univariate ranking "
        "is used for orientation only; the model-facing attribution is the SHAP analysis in "
        "Phase 8, which accounts for feature interactions this chart cannot see."
    ))
    plotting.save(fig, "fig09_univariate_feature_ranking")
    return {"top25_abs_point_biserial": corr.round(4).to_dict(),
            "engineered_in_top25": [c for c in corr.index if c in ENGINEERED_FEATURE_DOCS]}


def fig_engineered_separation(train: pd.DataFrame) -> dict:
    """Show how well (and how unevenly) the engineered features separate the classes."""
    engineered = add_engineered_features(train)
    continuous = ["src_byte_ratio", "src_pkt_ratio", "bytes_per_packet",
                  "load_log_ratio", "src_loss_rate", "jit_log_ratio"]
    flags = ["is_one_way", "tcp_handshake_complete", "tcp_seq_exchanged",
             "both_win_advertised", "is_zero_duration", "service_unknown"]

    fig, axes = plt.subplots(2, 6, figsize=(17, 6.4))
    for ax, col in zip(axes[0], continuous):
        for cls, color, name in ((0, config.COLOR_BENIGN, "Benign"),
                                 (1, config.COLOR_ATTACK, "Attack")):
            values = engineered.loc[engineered[config.TARGET] == cls, col].to_numpy(float)
            if col == "bytes_per_packet":
                values = np.log1p(values)
            ax.hist(values, bins=40, color=color, alpha=0.62, density=True, label=name)
        ax.set_title(col + (" (log1p)" if col == "bytes_per_packet" else ""), fontsize=9)
        ax.set_yticks([])
        plotting.despine(ax, keep=("bottom",))
    axes[0, 0].legend(fontsize=7.5)

    for ax, col in zip(axes[1], flags):
        rates = engineered.groupby(col, observed=True)[config.TARGET].agg(["size", "mean"])
        bars = ax.bar(rates.index.astype(str), rates["mean"] * 100,
                      color=[plotting.rate_to_color(r) for r in rates["mean"]],
                      width=0.6, edgecolor="#666666", linewidth=0.5)
        ax.set_ylim(0, 118)
        ax.axhline(train[config.TARGET].mean() * 100, color=config.COLOR_NEUTRAL,
                   linestyle="--", linewidth=1.1)
        plotting.annotate_bars(ax, bars, [f"{m:.0%}" for m in rates["mean"]], fontsize=8)
        ax.set_title(col, fontsize=9)
        ax.set_xlabel("flag value", fontsize=8)
        if ax is axes[1, 0]:
            ax.set_ylabel("Attack rate (%)", fontsize=8.5)
        plotting.despine(ax)

    # Measure each flag's univariate lift rather than asserting a direction.
    base = float(train[config.TARGET].mean())
    lifts: dict[str, float] = {}
    for col in flags:
        rates = engineered.groupby(col, observed=True)[config.TARGET].mean()
        lifts[col] = float(rates.max() - rates.min()) if len(rates) > 1 else 0.0
    ranked = sorted(lifts.items(), key=lambda kv: -kv[1])
    strong = ", ".join(f"{name} ({gap:+.0%} spread)" for name, gap in ranked[:2])
    weak = ", ".join(name for name, gap in ranked if gap < 0.05)

    fig.suptitle("Engineered features: how much each one separates the classes on its own "
                 f"(training split, n={len(train):,})")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    plotting.caption(fig, (
        "Top row: continuous constructions. src_byte_ratio is strongly bimodal - attacks pile up "
        f"near 1.0 (source talks, target never answers). Bottom row: the 0/1 flags, with the "
        f"attack rate at each level and the {base:.0%} base rate dashed. The flags differ sharply "
        f"in univariate usefulness: {strong} separate the classes clearly, whereas "
        f"{weak or 'none'} show almost no univariate gap at all. That is NOT evidence those "
        "features are useless - the TCP-state flags split in opposite directions within the "
        "attack class (Generic almost never completes a handshake; Exploits almost always does), "
        "so the two effects cancel when the classes are pooled. Their value, if any, is in "
        "interaction, which only the SHAP analysis in Phase 8 can see. Note also that "
        "service_unknown runs OPPOSITE to the usual security intuition here - identified services "
        "are more often malicious in this testbed - another reminder that these are generator "
        "artefacts as much as attack behaviour."
    ), y=0.015)
    plotting.save(fig, "fig10_engineered_feature_separation")
    return {"flag_univariate_lift": {k: round(v, 4) for k, v in lifts.items()},
            "base_rate_train": round(base, 4)}


def fig_volume_scatter(train: pd.DataFrame) -> None:
    """
    Source vs destination bytes: benign, attack, and by family.

    Benign and attack are given their OWN panels rather than being overlaid.
    On a single panel whichever class is drawn second hides the other, which
    makes the denser class look like the whole story - an artefact of draw order
    rather than a property of the data.
    """
    sample = train.sample(n=min(25_000, len(train)), random_state=config.RANDOM_STATE)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.8), sharex=True, sharey=True)

    panels = [
        (axes[0], sample[sample[config.TARGET] == 0], config.COLOR_BENIGN, "Benign"),
        (axes[1], sample[sample[config.TARGET] == 1], config.COLOR_ATTACK, "Attack"),
    ]
    for ax, subset, color, name in panels:
        ax.scatter(np.log1p(subset["sbytes"]), np.log1p(subset["dbytes"]),
                   s=5, alpha=0.22, color=color, linewidths=0)
        share_one_way = float((subset["dbytes"] == 0).mean())
        ax.set_title(f"{name}  (n={len(subset):,};  {share_one_way:.0%} received no reply)",
                     fontsize=10.5)
        ax.set_xlabel("log(1 + source bytes)")
        plotting.despine(ax)
    axes[0].set_ylabel("log(1 + destination bytes)")

    families = [c for c in ATTACK_ORDER if c != "Normal"]
    colors = plotting.class_colors(len(families))
    for family, color in zip(families, colors):
        subset = sample[sample[config.ATTACK_CAT] == family]
        if subset.empty:
            continue
        axes[2].scatter(np.log1p(subset["sbytes"]), np.log1p(subset["dbytes"]),
                        s=6, alpha=0.40, color=color,
                        label=f"{family} ({len(subset):,})", linewidths=0)
    axes[2].set_xlabel("log(1 + source bytes)")
    axes[2].set_title("By attack family", fontsize=10.5)
    legend = axes[2].legend(markerscale=3, fontsize=7.5, ncol=2, loc="upper left")
    for handle in legend.legend_handles:
        handle.set_alpha(1.0)
    plotting.despine(axes[2])

    generic = sample[sample[config.ATTACK_CAT] == "Generic"]
    generic_silent = float((generic["dbytes"] == 0).mean()) if len(generic) else float("nan")

    fig.suptitle(f"Flow volume geometry (training split, {len(sample):,}-record sample)")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    plotting.caption(fig, (
        "The flat band along the bottom of each panel is the set of flows that received no reply "
        "at all - the engineered is_one_way flag captures it directly. Benign traffic is mostly "
        "bidirectional and spreads up the diagonal; attack traffic concentrates far more heavily "
        f"on that bottom band. Generic is the extreme case: {generic_silent:.0%} of sampled "
        "Generic flows got no response. The diagonal arcs visible in both classes are "
        "protocol-specific request/response size relationships, not artefacts."
    ), y=0.02)
    plotting.save(fig, "fig11_volume_scatter")


def fig_family_profile(train: pd.DataFrame) -> dict:
    """
    Per-family feature fingerprint, z-scored across families.

    Continuous columns are summarised by their **median** (robust to the heavy
    tails documented in fig07); the 0/1 flags are summarised by their **mean**,
    i.e. the prevalence of the flag within the family. Taking a median of a
    Boolean would collapse every family to 0 or 1 and destroy the contrast that
    makes those columns interesting in the first place.
    """
    engineered = add_engineered_features(train)
    continuous = ["dur", "sbytes", "dbytes", "spkts", "dpkts", "rate", "sload",
                  "dload", "sttl", "dttl", "smean", "dmean", "ct_srv_src",
                  "ct_dst_ltm", "src_byte_ratio", "bytes_per_packet", "src_loss_rate"]
    flags = ["is_one_way", "tcp_handshake_complete", "service_unknown"]
    columns = continuous + flags

    logged = engineered[continuous].copy()
    for col in continuous:
        if logged[col].min() >= 0 and logged[col].max() > 10:
            logged[col] = np.log1p(logged[col])

    family = engineered[config.ATTACK_CAT]
    profile = pd.concat([
        logged.groupby(family, observed=True).median(),
        engineered[flags].groupby(family, observed=True).mean(),
    ], axis=1)[columns]
    profile = profile.reindex([c for c in ATTACK_ORDER if c in profile.index])
    z = (profile - profile.mean()) / profile.std(ddof=0).replace(0, np.nan)
    z = z.fillna(0.0)

    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    im = ax.imshow(z.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(
        [c + (" *" if c in flags else "") for c in columns],
        rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(z)))
    ax.set_yticklabels(z.index, fontsize=9)
    ax.grid(False)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            value = z.iat[i, j]
            if abs(value) >= 1.0:
                ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                        fontsize=6.8, color="white" if abs(value) > 1.5 else "#222222")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.015)
    cbar.set_label("z-score across families", fontsize=9)
    ax.set_title(f"Each attack family has a distinct traffic fingerprint "
                 f"(training split, n={len(train):,})")
    fig.subplots_adjust(bottom=0.30, top=0.93)
    plotting.caption(fig, (
        "Columns marked * are 0/1 flags summarised by prevalence within the family; the rest are "
        "log-scaled medians. Generic is the extreme outlier - the highest one-way rate and the "
        "lowest completed-handshake rate of any family, with near-zero response volume and a high "
        "packet rate: unanswered, machine-generated traffic. Normal sits at the opposite corner - "
        "the lowest source TTL and the lowest source-to-destination byte ratio, i.e. the most "
        "response-heavy traffic. Exploits shows the largest bytes-per-packet and Worms the "
        "largest mean source packet size, both consistent with payload delivery, while "
        "Reconnaissance shows the smallest of each - tiny probe packets. These differences are "
        "what let the per-family recall audit in Phase 9 explain WHICH attacks the model misses "
        "and why."
    ), y=0.085)
    plotting.save(fig, "fig12_attack_family_profile")
    return {"family_profile_zscores": z.round(3).to_dict("index")}


def fig_imbalance_context(dedup: pd.DataFrame) -> None:
    """Contrast the dataset's balance with plausible production base rates."""
    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    scenarios = {
        "UNSW-NB15\n(deduplicated corpus)": dedup[config.TARGET].mean(),
        "UNSW-NB15\n(as published)": 0.6391,
        "Enterprise perimeter\n(illustrative)": 0.01,
        "Enterprise internal\n(illustrative)": 0.001,
    }
    names = list(scenarios)
    values = [v * 100 for v in scenarios.values()]
    colors = [config.COLOR_ATTACK, config.COLOR_ATTACK, config.COLOR_WARN, config.COLOR_WARN]
    bars = ax.bar(names, values, color=colors, width=0.6)
    ax.set_yscale("log")
    ax.set_ylim(0.05, 300)
    ax.set_ylabel("Malicious share of flows (%, log scale)")
    plotting.annotate_bars(ax, bars, [f"{v:.3g}%" for v in values], pad=0.03)
    ax.set_title("The dataset's class balance is inverted relative to any real network")
    plotting.despine(ax)
    plotting.caption(fig, (
        "In UNSW-NB15 attacks are the MAJORITY class. In production they are a small minority. "
        "The two illustrative bars are stated as assumptions, not measurements. The consequence "
        "is quantified in the Model Evaluation Report: precision measured here does not transfer, "
        "because precision depends on the base rate while recall and FPR do not. This is the "
        "single most important caveat for anyone reading the headline numbers."
    ))
    plotting.save(fig, "fig13_class_imbalance_context")


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run() -> dict:
    """Generate every EDA figure and return the statistics used in the report."""
    plotting.use_house_style()
    config.ensure_dirs()

    from src import data_loader
    corpus = data_loader.load_corpus()
    dedup, prep_report = preprocessing.prepare_corpus(verbose=True)
    train, _, _ = preprocessing.split_corpus(dedup, verbose=True)

    stats: dict[str, object] = {
        "corpus_rows": int(len(corpus)),
        "dedup_rows": int(len(dedup)),
        "train_rows": int(len(train)),
        "preparation": prep_report,
    }

    fig_class_distribution(corpus, dedup)
    fig_attack_category_distribution(dedup)
    fig_duplication(corpus)
    stats["categorical_attack_rates"] = fig_categorical_attack_rates(train)
    stats["ttl_artifact"] = fig_ttl_artifact(train)
    fig_numeric_distributions(train)
    stats["outliers"] = fig_boxplots(add_engineered_features(train))
    stats["correlation"] = fig_correlation(train)
    stats["feature_ranking"] = fig_feature_ranking(train)
    stats["engineered_separation"] = fig_engineered_separation(train)
    fig_volume_scatter(train)
    stats["family_profile"] = fig_family_profile(train)
    fig_imbalance_context(dedup)

    path = config.METRICS_DIR / "eda_statistics.json"
    path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")
    print(f"[eda] wrote {path.relative_to(config.PROJECT_ROOT)}")
    return stats


if __name__ == "__main__":
    run()
