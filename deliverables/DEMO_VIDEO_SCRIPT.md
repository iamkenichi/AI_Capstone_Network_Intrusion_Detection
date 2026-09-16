# Demo Video — Script and Recording Guide

**Project:** Machine Learning-Based Network Intrusion Detection and Anomaly Classification
**Presenter:** Arne Ramos
**Target length:** 7–9 minutes (aim for 8)
**Format:** Screen recording with voice-over

> This document gives you a **shot-by-shot script** with exact commands, what to
> have on screen, and narration you can read aloud — plus the recording setup at
> the end. Numbers marked `[…]` should be read off your own generated reports so
> the video matches your actual run; every one of them is in
> `reports/Model_Evaluation_Report.md` or `reports/Rubric_Audit.md`.

---

## Before you press record — 10-minute checklist

```bash
cd AI_Capstone_Network_Intrusion_Detection

# 1. Make sure everything is generated and current
python -m src.pipeline --all
pytest -q

# 2. Pre-warm the Streamlit app so it does not compile on camera
streamlit run app/streamlit_app.py
#    → open http://localhost:8501, click through both tabs once, then leave it running
```

**Screen setup**

- Display resolution **1920×1080**, recording the full screen or a 1920×1080 region.
- Editor font size **16pt or larger** — code must be readable at 720p playback.
- Light theme in both the editor and the terminal (projector-safe, and the
  figures have white backgrounds).
- Close Slack, mail, notifications. Turn on Do Not Disturb.
- Have these open as **tabs or windows, pre-loaded**, so you never wait on screen:
  1. Terminal, at the repo root
  2. Editor with the repo open
  3. Browser tab: `README.md` rendered (GitHub or editor preview)
  4. Browser tab: Streamlit app at `localhost:8501`
  5. File explorer showing `figures/`

**One rehearsal.** Read the script once out loud with the screens in front of
you. You will find two or three places where the click lags the sentence — fix
those, then record.

---

## SHOT 1 — Title and the hook
**Duration:** 0:00 – 0:35 · **Screen:** Title slide (technical deck, slide 1)

> "Hi, I'm Arne Ramos. This is my capstone: a machine-learning system that
> detects malicious network traffic.
>
> I'll show you the result up front — it detects `[recall]%` of attacks on data
> it has never seen, and it meets every performance target I set before I started
> modelling.
>
> But that's not the interesting part. The interesting part is what I found when
> I checked whether that number actually means what it looks like it means. So
> that's where I'll spend most of these eight minutes."

**Direction:** Say the last line slightly slower. It sets up everything else.

---

## SHOT 2 — The problem
**Duration:** 0:35 – 1:15 · **Screen:** Technical deck slide 2, then executive deck slide 2

> "The problem is volume. A mid-sized enterprise produces tens of millions of
> network flow records a day. A tiny fraction are an attacker scanning, spreading,
> or taking data out.
>
> Reviewing them at one second each would take a century of analyst-hours per
> day. Signature-based tools help, but they only catch what someone has already
> written a rule for.
>
> And there are two ways to fail. Miss an attack, and it proceeds unchallenged.
> Alert on too much benign traffic, and the queue overflows — analysts stop
> trusting it and real detections get closed unread. That second failure mode is
> the one people underestimate."

---

## SHOT 3 — The dataset, and verifying it
**Duration:** 1:15 – 2:05 · **Screen:** Terminal

```bash
python -m src.data_loader
```

> "I used UNSW-NB15 — about 258,000 labelled flow records from the Australian
> Centre for Cyber Security.
>
> Notice what the loader does: it doesn't just download the files, it verifies
> them. Row counts against the published figures, column count, that the target
> is really binary, and SHA-256 hashes. I'm not trusting the mirror I downloaded
> from — I'm re-deriving the dataset's identity from its contents."

**Direction:** Let the output scroll and pause on the class-balance line.

> "And there's the first thing worth noticing: **attacks are the majority class
> here — 64%.** That's the opposite of a real network. I'll come back to why that
> matters."

---

## SHOT 4 — Finding #1: the duplication problem
**Duration:** 2:05 – 3:00 · **Screen:** `figures/fig03_duplication_by_attack_category.png`

> "Here's the first real finding. **40% of this dataset is duplicated** — 104,000
> records repeat an earlier feature vector exactly.
>
> And look at the shape of it. It isn't uniform. Generic attacks are 88%
> duplicates. Normal traffic is 8%.
>
> So if you split this data randomly without deduplicating — which is what most
> published benchmarks on UNSW-NB15 do — you put byte-identical attack records in
> both your training set and your test set. Your model gets rewarded for
> memorising, and your score goes up for the wrong reason.
>
> I deduplicate **before** splitting. That drops the corpus to about 154,000
> unique flows and flips the class balance to 56/44. And I ran the experiment both
> ways so I could measure exactly what that decision was worth."

---

## SHOT 5 — Finding #2: the capture artefact
**Duration:** 3:00 – 4:05 · **Screen:** `figures/fig05_ttl_artifact.png`

> "The second finding is bigger. This is the TTL field — time-to-live, the hop
> counter in every IP packet.
>
> Look at the middle panel. Almost every TTL value in this dataset is either
> about zero percent malicious or about one hundred percent malicious. There's
> almost nothing in between.
>
> A lookup rule using **that one column and nothing else** classifies `[81]%` of
> the training data correctly, against a `[56]%` baseline.
>
> That is not how TTL behaves on a real network. What happened is that the
> UNSW-NB15 testbed ran its benign traffic generator and its attack generator on
> hosts configured with different initial TTL values. So `sttl` doesn't mostly
> encode 'is this hostile' — it encodes 'which machine generated this'.
>
> Now, I did **not** delete the feature. It's a real field a real sensor
> observes, and throwing away genuine signal on suspicion is its own mistake.
> Instead I did two things: I measured its influence on the fitted model with
> SHAP, and I retrained the whole pipeline without it. I'll show you both."

---

## SHOT 6 — Pipeline and leakage controls
**Duration:** 4:05 – 4:50 · **Screen:** Editor — `src/preprocessing.py`, then terminal

> "The pipeline itself is built so that leakage is structurally impossible rather
> than something I have to remember.
>
> Feature engineering, preprocessing and the estimator all live in one
> scikit-learn `Pipeline`. So when cross-validation runs, it re-fits the scaler
> and the one-hot encoder on each fold's own training portion. I can't
> accidentally fit a scaler on the full dataset.
>
> `attack_cat` — the attack-family label — is dropped in exactly one place,
> because it determines the target perfectly. Using it would be textbook leakage.
>
> And I engineered 15 features, all of them row-wise — each is a function of a
> single flow. That matters twice: a feature that never sees another row can't
> leak across the split, and it can actually be computed by a sensor on one flow
> at inference time."

```bash
pytest tests/test_preprocessing.py -q
```

> "These aren't smoke tests. They assert that no leakage column survives, that no
> feature correlates above 0.999 with the target, and that no feature vector
> appears in both train and test."

---

## SHOT 7 — Results
**Duration:** 4:50 – 5:50 · **Screen:** `figures/fig16_model_comparison.png`, then `fig15_roc_pr_curves.png`

> "Five models: logistic regression as an interpretable baseline, random forest,
> XGBoost, LightGBM, and an isolation forest trained on benign traffic only —
> that last one tells me what the labels are actually buying me.
>
> The winner is `[MODEL]`, at F1 `[F1]`, ROC-AUC `[AUC]`, PR-AUC `[PRAUC]`.
>
> I selected on **PR-AUC, not accuracy** — deliberately. On a corpus that's 44%
> positive, accuracy squashes every model into a narrow band and rewards being
> good at the majority class. PR-AUC measures how well the model *ranks* attacks
> above benign flows, which is what survives when the base rate changes.
>
> And on the operating side: `[FNR]%` of attacks missed, `[FPR]%` of benign
> traffic flagged. Those two numbers, not accuracy, are what a SOC lead reads."

**Screen:** `figures/fig17_threshold_analysis.png`

> "I also didn't leave the threshold at 0.5. I swept it on the validation split —
> never on test — and published three operating points: F1-optimal, cost-optimal
> under a stated 20-to-1 assumption, and the tightest point that respects a 1%
> false-positive budget. Which one you run is a business decision about analyst
> capacity, not a modelling decision."

---

## SHOT 8 — Explainability and the artefact, measured
**Duration:** 5:50 – 6:40 · **Screen:** `figures/fig21_shap_beeswarm.png`, then `fig24_shap_artifact_check.png`

> "Here's SHAP. Each dot is one test flow, and its horizontal position is that
> feature's contribution to the attack score.
>
> What makes traffic look malicious to this model: high source TTL, a connection
> where the destination never answered, no completed TCP handshake, and strongly
> asymmetric direction. What makes it look benign: a completed handshake with
> sequence numbers both ways, and balanced two-way volume. Those are sensible.
>
> But now the diagnostic I actually built this for."

**Screen:** `fig24_shap_artifact_check.png`

> "The TTL family carries `[X]%` of the model's total attributed decision impact.
> And look at the right-hand panel — the relationship is a **step**, not a
> gradient. The model learned a near-binary switch on a field that the testbed
> happened to configure differently for its two generators.
>
> So SHAP turned my suspicion from the EDA into a measured number on the actual
> fitted model."

**Screen:** `figures/fig19_ablation_comparison.png`

> "And then the ablations. Each bar group is a separate end-to-end experiment.
> Duplicates retained. TTL removed. Engineered features removed. SMOTE instead of
> class weights. This table is what separates 'the model detects attacks' from
> 'the model detects this dataset'."

---

## SHOT 9 — Error analysis and the bias audit
**Duration:** 6:40 – 7:20 · **Screen:** `figures/fig18_subgroup_audit.png`

> "Aggregate metrics hide things, so I disaggregated them.
>
> On the left, recall by attack family. It is **not** uniform — and the weakest
> families are the stealthy, low-volume ones a defender most wants caught. Some of
> that is scarcity, which more data fixes. Some of it is behavioural overlap with
> normal traffic — a backdoor that looked like a port scan would be a bad backdoor
> — and that is **not** fixable at the flow level, by any model.
>
> On the right, where false alerts concentrate. A handful of services generate a
> disproportionate share, which is directly actionable with per-service thresholds.
>
> One thing I want to be explicit about: **this dataset has no demographic
> attributes and no human subjects.** It's synthetic machine-generated traffic with
> the IPs and ports stripped. So I make no demographic fairness claims — I can't,
> and inventing one would be dishonest. What I did instead is an *operational*
> performance audit, and then I separately wrote up where real fairness risk does
> appear at deployment time — because uneven per-service false-positive rates mean
> uneven scrutiny of the people behind those services."

---

## SHOT 10 — Live demo
**Duration:** 7:20 – 8:10 · **Screen:** Streamlit app

**Actions, in order:**
1. Select **"Reconnaissance — representative"** from the example dropdown.
2. Click **Score this flow**. Let the verdict card and SHAP chart render.
3. Drag the sidebar threshold slider up until the verdict flips to BENIGN.
4. Switch to the **Batch scoring** tab briefly.

> "And here's the deployment prototype. I load a real reconnaissance flow from the
> training set — not a synthetic average, an actual record — and score it.
>
> It comes back as an attack, with a probability, a risk band, and a recommended
> action. And underneath, the specific features that drove it: red pushes toward
> attack, blue toward benign. That's what an analyst needs — not a number, a
> reason.
>
> If I raise the threshold" — *drag slider* — "you can watch the verdict flip.
> That's the recall-versus-workload trade-off, live.
>
> There's also batch scoring for a CSV, which produces a ranked triage queue."

---

## SHOT 11 — Close
**Duration:** 8:10 – 8:40 · **Screen:** Technical deck slide 12

> "So — the model works. It hits every target I pre-registered.
>
> And it comes with three honest qualifications: a meaningful share of its signal
> is a capture artefact, its recall is uneven across attack families, and its
> precision is measured on a corpus where attacks are the majority class, so it
> will not transfer to a real network.
>
> Which is exactly why my recommendation is human-in-the-loop triage ranking,
> recalibrated on the target network — not autonomous blocking.
>
> Everything is reproducible: clone it, `pip install -r requirements.txt`, run
> `python -m src.pipeline --all`, and every number in every report regenerates
> from executed code. Thanks for watching."

---

## Timing summary

| Shot | Content | Start | Length |
|---|---|---|---|
| 1 | Title and hook | 0:00 | 0:35 |
| 2 | The problem | 0:35 | 0:40 |
| 3 | Dataset and verification | 1:15 | 0:50 |
| 4 | Finding #1 — duplication | 2:05 | 0:55 |
| 5 | Finding #2 — TTL artefact | 3:00 | 1:05 |
| 6 | Pipeline and leakage controls | 4:05 | 0:45 |
| 7 | Results and threshold | 4:50 | 1:00 |
| 8 | SHAP and ablations | 5:50 | 0:50 |
| 9 | Error analysis and bias audit | 6:40 | 0:40 |
| 10 | Live Streamlit demo | 7:20 | 0:50 |
| 11 | Close | 8:10 | 0:30 |
| | **Total** | | **8:40** |

If you need to cut to 5 minutes, drop shots 6 and 9 and shorten shot 2. Do not
cut shots 4, 5 or 8 — they are the contribution.

---

## Recording setup

### Option A — OBS Studio (free, best quality)

1. Install from <https://obsproject.com>.
2. **Sources** → `+` → **Display Capture** (or **Window Capture** to hide your taskbar).
3. **Sources** → `+` → **Audio Input Capture** → your microphone.
4. **Settings → Output**: Recording Quality *High*, Format **MP4**, Encoder *x264*.
5. **Settings → Video**: Base and Output resolution **1920×1080**, FPS **30**.
6. **Settings → Audio**: Sample rate 48 kHz.
7. Check the mic level meter sits in the **yellow**, never the red, when you speak.
8. `Start Recording` → present → `Stop Recording`. Output lands in `Videos/`.

### Option B — PowerPoint (simplest, no install)

1. Open the technical deck.
2. **Slide Show → Record Slide Show → Record from Beginning**.
3. Narrate through the slides; use **Alt+Tab** to switch to the terminal and the
   Streamlit app when the script calls for it *(note: PowerPoint records the
   slide area, so for shots 3, 6 and 10 you need Option A or C)*.
4. **File → Export → Create a Video** → *Full HD (1080p)* → `Create Video`.

### Option C — Windows Game Bar (built in, quick)

1. `Win + G` → **Capture** widget → click the record button (or `Win + Alt + R`).
2. Records the **active window**, so click into the window you want first.
3. Output lands in `Videos\Captures\`.

**Caveat:** Game Bar cannot record the desktop or File Explorer, only app
windows. Use Option A if you need to show the file system.

### Audio tips that matter more than the camera

- Use a headset mic, not the laptop's built-in one.
- Record in a room with soft furnishing. Bare walls sound like a bathroom.
- Stay ~15 cm from the mic, slightly off-axis so plosives don't pop.
- Record 3 seconds of silence at the start — editors use it for noise reduction.
- If you fluff a line, **pause, clap once, and redo the sentence**. The clap gives
  you a visible spike in the waveform to cut on.

### Editing (optional, 20 minutes)

Free options: **DaVinci Resolve**, **Clipchamp** (ships with Windows), **Shotcut**.

Minimum worthwhile edits:
1. Trim dead air at the start and end.
2. Cut the retakes at your clap marks.
3. Normalise audio to about **−16 LUFS**.
4. Add a title card for 3 seconds at the front: project title, your name, date.

### Export settings

| Setting | Value |
|---|---|
| Container | MP4 (H.264 + AAC) |
| Resolution | 1920×1080 |
| Frame rate | 30 fps |
| Video bitrate | 8–12 Mbps |
| Audio | 192 kbps, 48 kHz |
| Target size | under 500 MB for 8 minutes |

**Filename:** `Arne_Ramos_AI_Capstone_Demo_Video.mp4`

If the submission portal caps upload size, upload to YouTube as **Unlisted** or
to Google Drive with link sharing, and put the link in the report cover page and
on slide 1 of the technical deck.

---

## Where to read each `[…]` number off your own run

| Placeholder | Source |
|---|---|
| `[recall]`, `[F1]`, `[AUC]`, `[PRAUC]`, `[FNR]`, `[FPR]`, `[MODEL]` | `reports/Model_Evaluation_Report.md` §2–§3, or the README headline table |
| `[81]%` TTL rule accuracy, `[56]%` baseline | `reports/EDA_Feature_Engineering_Report.md` §3 |
| `[X]%` TTL share of SHAP impact | `reports/Bias_Fairness_Analysis.md` §6.1, or `figures/fig24` caption |

Read them once before recording and write them on a sticky note. Do not read
them off the screen mid-take — it shows.
