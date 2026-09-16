"""
Network Intrusion Detection - interactive demonstration.

A prototype analyst console over the frozen model produced by the pipeline. It
exists to make three things concrete that a notebook cannot:

* what the model does with **one** flow, including why;
* how the decision changes as the operating threshold moves;
* what a batch of scored flows looks like when it reaches a triage queue.

Run from the repository root::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Allow `streamlit run app/streamlit_app.py` from the repository root without
# requiring the package to be pip-installed first.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config, predict  # noqa: E402

st.set_page_config(
    page_title="Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      .main > div { padding-top: 1.2rem; }
      .verdict-card {
          border-radius: 10px; padding: 1.1rem 1.4rem; margin-bottom: 0.9rem;
          border-left: 7px solid; background: #FAFAFA;
      }
      .verdict-title { font-size: 1.55rem; font-weight: 700; margin: 0 0 .25rem 0; }
      .verdict-sub   { font-size: .95rem; color: #444; margin: 0; }
      .metric-note   { font-size: .80rem; color: #666; }
      .disclaimer {
          border-radius: 8px; padding: .85rem 1.05rem; background: #FFF6E5;
          border-left: 5px solid #E69F00; font-size: .87rem; color: #4a3b1a;
      }
      .stTabs [data-baseweb="tab-list"] { gap: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

BAND_COLOR = {
    "Critical": "#B2182B",
    "High": "#D55E00",
    "Medium": "#E69F00",
    "Low": "#0072B2",
}

DISCLAIMER = (
    "**Educational / research prototype.** This model was trained on UNSW-NB15, a "
    "synthetic 2015 testbed dataset in which attacks are the majority class and "
    "certain fields carry capture artefacts. It must **not** be used as a "
    "standalone production intrusion-detection system, and its probabilities "
    "should not be read as calibrated risk for a live network. Use it as one "
    "signal inside a defence-in-depth stack, with human review."
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading the trained model ...")
def get_model():
    return predict.load_model()


@st.cache_data(show_spinner=False)
def get_reference():
    return predict.feature_reference()


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    """Load the frozen model's measured test metrics, if the pipeline has run."""
    import json

    path = config.METRICS_DIR / "operating_points_main.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def fatal(message: str) -> None:
    st.error(message)
    st.code(
        "python -m src.data_loader\n"
        "python -m src.preprocessing\n"
        "python -m src.train\n"
        "python -m src.pipeline --evaluate",
        language="bash",
    )
    st.stop()


try:
    MODEL = get_model()
    REFERENCE = get_reference()
except predict.ModelNotAvailable as exc:
    fatal(f"No trained model is available yet.\n\n{exc}")
except Exception as exc:  # noqa: BLE001
    fatal(f"Could not load the model: {type(exc).__name__}: {exc}")

METRICS = get_metrics()


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.title("🛡️ Detection console")
    st.caption("UNSW-NB15 flow-based intrusion detection")

    st.subheader("Active model")
    st.markdown(f"**{MODEL.display_name}**")
    if MODEL.metadata.get("selection_reason"):
        st.caption(MODEL.metadata["selection_reason"])

    st.subheader("Operating threshold")
    threshold = st.slider(
        "Alert when attack probability is at least",
        min_value=0.05, max_value=0.95,
        value=float(MODEL.threshold), step=0.01,
        help=("The deployed default was chosen by maximising F1 on the validation "
              "split. Raising it reduces false alerts but misses more attacks; "
              "lowering it does the reverse."),
    )
    if abs(threshold - MODEL.threshold) > 1e-9:
        st.caption(f"⚠️ Overriding the frozen threshold of {MODEL.threshold:.2f}.")

    if METRICS.get("adopted_f1_optimal"):
        m = METRICS["adopted_f1_optimal"]
        st.subheader("Measured test performance")
        st.caption("At the frozen threshold, on the held-out test split.")
        col_a, col_b = st.columns(2)
        col_a.metric("Attack recall", f"{m['recall']:.1%}")
        col_b.metric("Precision", f"{m['precision']:.1%}")
        col_a.metric("False negatives", f"{m['false_negative_rate']:.2%}")
        col_b.metric("False positives", f"{m['false_positive_rate']:.2%}")
        st.caption(
            "Precision is measured on a corpus where attacks are "
            f"{m['positive_rate_actual']:.0%} of traffic. On a real network, where "
            "they are a fraction of a percent, precision would be far lower at the "
            "same threshold."
        )

    st.divider()
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.title("Network Intrusion Detection")
st.markdown(
    "Score individual network flows or a batch of them, see the probability the "
    "model assigns, the risk band it falls into, and the specific features that "
    "drove the decision."
)

tab_single, tab_batch, tab_about = st.tabs(
    ["🔍 Score a single flow", "📦 Batch scoring", "📖 About this model"])


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_verdict(result: dict) -> None:
    band = result["risk_band"]
    color = BAND_COLOR.get(band, "#4D4D4D")
    verdict = result["verdict"]
    probability = float(result["attack_probability"])
    icon = "🚨" if verdict == "ATTACK" else "✅"

    st.markdown(
        f"""
        <div class="verdict-card" style="border-left-color:{color}">
          <p class="verdict-title" style="color:{color}">{icon} {verdict}
             &nbsp;·&nbsp; {band} risk</p>
          <p class="verdict-sub">{result['recommended_action']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        st.markdown("**Attack probability**")
        st.progress(min(max(probability, 0.0), 1.0))
        st.caption(
            f"{probability:.4f} against an alert threshold of {result['threshold']:.2f} "
            f"— {'above' if probability >= result['threshold'] else 'below'} the line."
        )
    col_b.metric("Probability", f"{probability:.1%}")
    col_c.metric("Margin to threshold", f"{probability - result['threshold']:+.3f}")


def render_explanation(record: dict) -> None:
    st.markdown("#### Why the model decided this")
    with st.spinner("Computing SHAP contributions ..."):
        contributions = predict.explain_one(record, model=MODEL, top_n=10)

    if contributions is None or contributions.empty:
        st.info(
            "Per-flow explanations are available for tree-based models only. The "
            "active model does not support them, or `shap` is not installed."
        )
        return

    display = contributions.copy()
    display["contribution"] = display["shap_value"].round(4)
    display["feature value (standardised)"] = display["value"].round(3)

    chart_data = (display.set_index("feature")["contribution"]
                  .sort_values(key=abs, ascending=True))
    st.bar_chart(chart_data, horizontal=True, height=330)

    st.caption(
        "Positive contributions push the flow toward ATTACK; negative ones push it "
        "toward BENIGN. Values are on the model's internal log-odds scale, and "
        "feature values are standardised (0 = the training-set average)."
    )
    with st.expander("Numeric detail"):
        st.dataframe(
            display[["feature", "feature value (standardised)", "contribution", "direction"]],
            use_container_width=True, hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Tab 1: single-flow scoring
# --------------------------------------------------------------------------- #
with tab_single:
    examples: dict = REFERENCE.get("examples", {})
    labels = {
        "typical_benign": "Typical benign flow",
        "typical_attack": "Typical attack flow",
        "typical_exploits": "Exploits — representative",
        "typical_reconnaissance": "Reconnaissance — representative",
        "typical_dos": "Denial of service — representative",
        "typical_generic": "Generic — representative",
        "typical_fuzzers": "Fuzzers — representative",
    }
    available = [k for k in labels if k in examples]

    left, right = st.columns([1, 1.5])
    with left:
        choice = st.selectbox(
            "Start from a real training-set flow",
            options=available,
            format_func=lambda k: labels.get(k, k),
            help=("Each option is an ACTUAL record from the training split — the one "
                  "closest to that group's median profile — not a synthetic average, "
                  "which could combine physically impossible field values."),
        )
    with right:
        st.caption(
            "Load an example, then adjust any field below to see how the model's "
            "verdict responds. This is the fastest way to build intuition about "
            "which measurements the detector is actually sensitive to."
        )

    if "flow" not in st.session_state or st.session_state.get("_choice") != choice:
        st.session_state.flow = dict(examples[choice])
        st.session_state._choice = choice

    flow = st.session_state.flow
    numeric_ref = REFERENCE["numeric"]
    categorical_ref = REFERENCE["categorical"]

    GROUPS = {
        "Connection basics": ["proto", "service", "state", "dur", "rate"],
        "Volume": ["sbytes", "dbytes", "spkts", "dpkts", "smean", "dmean"],
        "Throughput & loss": ["sload", "dload", "sloss", "dloss"],
        "TTL & TCP session": ["sttl", "dttl", "swin", "dwin", "stcpb", "dtcpb",
                              "tcprtt", "synack", "ackdat"],
        "Timing": ["sinpkt", "dinpkt", "sjit", "djit"],
        "Connection history": ["ct_srv_src", "ct_srv_dst", "ct_state_ttl",
                               "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                               "ct_dst_sport_ltm", "ct_dst_src_ltm"],
        "Application & flags": ["trans_depth", "response_body_len", "ct_flw_http_mthd",
                                "is_ftp_login", "ct_ftp_cmd", "is_sm_ips_ports"],
    }

    st.markdown("#### Flow features")
    edited: dict = {}
    for group_name, columns in GROUPS.items():
        with st.expander(group_name, expanded=(group_name == "Connection basics")):
            grid = st.columns(3)
            for i, column in enumerate(columns):
                target = grid[i % 3]
                if column in categorical_ref:
                    options = categorical_ref[column]["options"]
                    current = str(flow.get(column, options[0]))
                    index = options.index(current) if current in options else 0
                    edited[column] = target.selectbox(column, options, index=index,
                                                      key=f"in_{column}")
                else:
                    ref = numeric_ref[column]
                    value = float(flow.get(column, ref["median"]))
                    edited[column] = target.number_input(
                        column, value=value,
                        min_value=float(ref["min"]),
                        # Allow headroom above the observed training maximum so a
                        # user can probe how the model behaves out of range.
                        max_value=float(ref["max"]) * 10 + 1.0,
                        format="%.6g", key=f"in_{column}",
                        help=(f"training median {ref['median']:.4g}, "
                              f"range {ref['min']:.4g} – {ref['max']:.4g}"),
                    )

    st.session_state.flow = edited

    if st.button("Score this flow", type="primary", use_container_width=True):
        try:
            result = predict.predict_one(edited, model=MODEL, threshold=threshold)
        except ValueError as exc:
            st.error(f"Invalid flow record: {exc}")
        else:
            st.divider()
            render_verdict(result)
            render_explanation(edited)


# --------------------------------------------------------------------------- #
# Tab 2: batch scoring
# --------------------------------------------------------------------------- #
with tab_batch:
    st.markdown("#### Score a file of flow records")
    st.caption(
        "Upload a CSV with the 42 UNSW-NB15 predictor columns. A `label` column, "
        "if present, is ignored for scoring and used only to report accuracy."
    )
    with st.expander("Required columns"):
        st.code(", ".join(predict.REQUIRED_INPUT_COLUMNS), language="text")

    example_path = config.PROJECT_ROOT / "app" / "example_flows.csv"
    if example_path.exists():
        st.download_button(
            "Download a 120-flow example file",
            data=example_path.read_bytes(),
            file_name="example_flows.csv",
            mime="text/csv",
            help=("Real records from the held-out test split, with their true "
                  "label and attack family kept for comparison. Both are ignored "
                  "during scoring."),
        )

    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            frame = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read that file: {exc}")
        else:
            st.caption(f"Loaded {len(frame):,} rows × {frame.shape[1]} columns.")
            try:
                scored = predict.predict(frame, model=MODEL, threshold=threshold)
            except ValueError as exc:
                st.error(str(exc))
            else:
                alerts = int(scored["prediction"].sum())
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Flows scored", f"{len(scored):,}")
                col_b.metric("Alerts raised", f"{alerts:,}")
                col_c.metric("Alert rate", f"{alerts / len(scored):.2%}")
                col_d.metric("Critical / High",
                             f"{int(scored['risk_band'].isin(['Critical', 'High']).sum()):,}")

                if "label" in frame.columns:
                    truth = pd.to_numeric(frame["label"], errors="coerce")
                    valid = truth.notna()
                    if valid.any():
                        from sklearn.metrics import recall_score, precision_score
                        y = truth[valid].astype(int)
                        p = scored.loc[valid.values, "prediction"]
                        col_a, col_b = st.columns(2)
                        col_a.metric("Recall on this file",
                                     f"{recall_score(y, p, zero_division=0):.2%}")
                        col_b.metric("Precision on this file",
                                     f"{precision_score(y, p, zero_division=0):.2%}")

                st.markdown("##### Triage queue — highest risk first")
                queue = scored.sort_values("attack_probability", ascending=False)
                st.dataframe(
                    queue[["attack_probability", "verdict", "risk_band",
                           "recommended_action"]].head(200),
                    use_container_width=True, height=380,
                )
                st.download_button(
                    "Download all scored flows (CSV)",
                    data=pd.concat([frame.reset_index(drop=True),
                                    scored.reset_index(drop=True)], axis=1)
                          .to_csv(index=False).encode("utf-8"),
                    file_name="scored_flows.csv",
                    mime="text/csv",
                )
                st.markdown("##### Risk-band distribution")
                st.bar_chart(scored["risk_band"].value_counts())


# --------------------------------------------------------------------------- #
# Tab 3: about
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown(f"""
### What this is

A flow-based intrusion detector: it reads the statistical summary of one network
conversation — duration, byte and packet counts in each direction, protocol,
service, connection state and TCP session timing — and estimates the probability
that the conversation was malicious. It never sees packet payloads, IP addresses
or port numbers.

**Active model:** {MODEL.display_name}
**Frozen threshold:** {MODEL.threshold:.2f}
**Training corpus:** UNSW-NB15, deduplicated, {REFERENCE.get('n_training_rows', 0):,} training flows

### Risk bands

| Band | Probability | Recommended action |
|---|---|---|
""" + "\n".join(
        f"| {band} | ≥ {floor:.2f} | {action} |"
        for floor, band, action in predict.RISK_BANDS
    ) + """

### How it was built and validated

1. The corpus was **deduplicated before splitting** — 40.4% of UNSW-NB15 repeats
   an earlier feature vector, and leaving those in place puts identical records
   in both training and test data.
2. Splits are stratified 60/20/20 by attack family at `random_state=42`.
3. All preprocessing is fitted **inside the training fold only**.
4. `attack_cat` is never used as a predictor — it determines the target exactly.
5. Hyper-parameters were tuned with randomised search under 5-fold stratified
   cross-validation, on training data only.
6. The decision threshold was chosen on the **validation** split.
7. The test split was scored **once**, after the model and threshold were frozen.

### What you should be sceptical about

- **The class balance is inverted.** Attacks are the majority class in this
  dataset. Real networks are the opposite, so the precision shown in the sidebar
  is optimistic for deployment. Recall and false-positive *rate* transfer; raw
  precision does not.
- **`sttl` is partly a capture artefact.** The testbed's benign and attack
  generators used different initial TTLs, so the model gets real predictive
  power from a field that would not behave that way on a live network. The
  project quantifies this with SHAP and a dedicated ablation — see
  `figures/fig05_ttl_artifact.png` and `figures/fig24_shap_artifact_check.png`.
- **The data is from 2015.** The attack taxonomy predates modern ransomware,
  living-off-the-land and supply-chain techniques.
- **An adversary adapts.** Flow statistics are manipulable: padding packets and
  throttling rates can move a flow across the boundary. Treat this as one layer,
  never the only one.

### Reproducing everything

```bash
pip install -r requirements.txt
python -m src.data_loader
python -m src.pipeline --all
streamlit run app/streamlit_app.py
```
""")
    st.markdown(f'<div class="disclaimer">{DISCLAIMER}</div>', unsafe_allow_html=True)
