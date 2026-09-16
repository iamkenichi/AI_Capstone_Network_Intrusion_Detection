# Executive Presentation

**Automated Detection of Malicious Network Activity**

*Audience: executives and security leadership · 10 slides · ~12 minutes*
*Generated 2026-09-16. No equations, no code.*

---

## Slide 1 — Executive Title

**Content**
- Automated Detection of Malicious Network Activity
- A machine-learning layer that reviews every network connection and ranks the suspicious ones for analysts
- Proof-of-concept result: **96% of attacks detected**, with **26.74% of normal traffic flagged for review**
- Recommendation: deploy as an **analyst assistant**, not an automated gatekeeper

**Recommended visual:** Clean title slide; one large statistic — "96% of attacks detected" — as the focal point.

**Speaker notes:** Lead with the outcome and the recommendation. Signal early that this is a decision-support tool, so nobody leaves thinking we are proposing to let software block traffic on its own.

---

## Slide 2 — The Cybersecurity Problem

**Content**
- Our network records **millions of connections every day**
- A small number of them are an attacker probing, spreading, or taking data out
- No team can review them all — reviewing one second each would take a century of staff time per day
- Today we rely on rules that recognise **known** attacks. They work, and they are blind to anything new

**Recommended visual:** A haystack graphic — millions of grey connections, a handful of red ones. No numbers on the slide itself.

**Speaker notes:** Avoid jargon entirely. The question this slide should raise in the audience's mind is "so how do we find the red ones?"

---

## Slide 3 — Why Existing Monitoring Is Difficult

**Content**
- **Rules only catch what we have already seen.** A new tool or technique passes straight through
- **Volume beats people.** Analyst capacity is fixed; traffic is not
- **Too many alerts is as bad as too few.** A team that cannot work its queue stops trusting it — and real attacks get closed unread
- The cost of the two mistakes is very different, and both are real

**Recommended visual:** Simple two-column comparison: "Missed attack" (breach, dwell time, regulatory exposure) vs "False alarm" (wasted analyst hours, eroded trust).

**Speaker notes:** The alert-fatigue point usually lands hardest with this audience. Emphasise that a detector producing more alerts than we can process does not degrade gracefully — it fails completely, because people stop reading it.

---

## Slide 4 — The Proposed Solution

**Content**
- Software that learns what attacks **look like** from thousands of past examples
- It examines the *shape* of each connection — how long, how much data, which direction, whether the other side answered — and never reads message contents
- Every connection gets a **risk score** and a **plain-English reason**
- Analysts start at the top of a ranked list instead of searching a haystack

**Recommended visual:** Simple pipeline: Network traffic → Risk scoring → Ranked queue → Analyst → Action. One row, five boxes.

**Speaker notes:** The privacy point is worth stating unprompted: the system reads connection statistics, not message contents. It usually pre-empts the first question from legal or from the works council.

---

## Slide 5 — Detection Performance

**Content**
- Tested on **52,644 connections the system had never seen**
- **Caught 18,231 of 18,982 attacks** (96.0%)
- **Missed 751** (4.0%)
- Flagged **9,000 of 33,662 normal connections** for review (26.74%)
- Fast enough to keep up with our traffic on ordinary hardware — no special equipment needed

**Recommended visual:** Four large stat tiles: attacks caught, attacks missed, normal traffic flagged, speed. `figures/fig16_model_comparison.png` (middle panel only) as support.

**Speaker notes:** Give the honest caveat here rather than burying it: the test data has far more attacks than a real network does, so the "how many of our alerts are real" figure will be worse in production. What does carry over is the detection rate and the share of normal traffic flagged.

---

## Slide 6 — Business and Security Value

**Content**
- **Faster detection.** Time-to-detection is the variable most strongly linked to breach cost — every hour saved limits the damage
- **Coverage for unknown attacks.** Catches suspicious *behaviour*, not just known signatures
- **Better use of analyst time.** A ranked, explained queue replaces undirected searching
- **Quantified risk.** We can finally answer "what fraction of attacks of this kind do we catch?" with a measured number
- **Consistency.** The same judgement applied to the ten-thousandth connection of a night shift as to the first

**Recommended visual:** Four value tiles with icons. Deliberately no ROI figure.

**Speaker notes:** If asked for a financial ROI: we can build one from our own analyst cost and incident history, but the honest position today is that we have not measured it on our own network, and we are not going to present an invented number.

---

## Slide 7 — The Trade-off We Control

**Content**
- There is a dial. Turning it up catches more attacks and generates more false alarms; turning it down does the reverse
- **This is a business decision about analyst capacity, not a technical one**
- At the recommended setting: 96% of attacks caught, 26.74% of normal traffic flagged
- A tighter setting is available: it alerts on only 5.84% of normal traffic, at the cost of catching 83% of attacks instead of 96%
- At our illustrative volume of 10 million connections a day, that recommended setting implies roughly **2,673,640 items to review per day** — which is exactly the number we must size the team against

**Recommended visual:** `figures/fig17_threshold_analysis.png`, left panel only, relabelled in plain language ("attacks caught" / "false alarms").

**Speaker notes:** This is the slide that needs a decision from the room. Present the two settings and ask which the SOC can actually staff. Be clear the daily volume is an illustration, not a measurement of our network.

---

## Slide 8 — Explainability and Governance

**Content**
- Every alert comes with **the specific reasons it was raised** — not just a score
- Example: *"1.2 MB sent, nothing returned, the connection never properly opened, and this source contacted 43 different services in the last few minutes"*
- Supports audit, regulatory review and analyst training
- We can show **which attack types we detect well and which we do not** — and we do
- Known limitations are documented rather than hidden

**Recommended visual:** A mocked-up alert card showing score, risk band, and three plain-English reasons.

**Speaker notes:** Explainability is a governance requirement, not a nice-to-have. Add that the same analysis told us about a weakness in our own test data — which is how we know the process is working rather than just producing flattering numbers.

---

## Slide 9 — Deployment Recommendation

**Content**
- **Deploy as an analyst assistant.** It ranks and explains; people decide. It does **not** block traffic
- **Phase 1 — Shadow mode (4–6 weeks).** Run alongside existing tools, measure the real false-alarm rate on our own traffic. No analyst action
- **Phase 2 — Assisted triage.** Analysts work the ranked queue; every decision is captured
- **Phase 3 — Continuous improvement.** Retrain on our own data and analyst feedback; monitor for drift
- **Requirement:** recalibrate on our network before go-live. Performance measured on public research data will not transfer unchanged

**Recommended visual:** Three-phase timeline with a decision gate between each phase.

**Speaker notes:** The shadow-mode phase is non-negotiable and is also the easiest thing to approve — it carries no operational risk and produces the number we actually need.

---

## Slide 10 — Key Takeaways

**Content**
1. **It works.** 96% of attacks detected on data the system had never seen, at ordinary hardware speed
2. **It is honest about what it misses.** Weaker on stealthy, low-volume attack types — documented, not hidden
3. **It is explainable.** Every alert carries its reasons, satisfying audit and regulatory needs
4. **It is a layer, not a replacement.** It works alongside existing controls and human judgement
5. **Next step:** approve a 4–6 week shadow-mode pilot on our own traffic — no operational risk, and it gives us the real numbers

**Recommended visual:** Five numbered takeaways, generous white space, the ask highlighted.

**Speaker notes:** Close on the specific ask: approval for the shadow-mode pilot. That is the only decision needed today, and it is a low-risk one.

---

## Anticipated questions

| Question | Answer |
|---|---|
| *Can it replace our current tools?* | No. It catches different things and should run alongside them. |
| *Will it block attacks automatically?* | Not as recommended. It misses some attacks with high confidence, which makes autonomous blocking inappropriate today. |
| *How many false alarms will we actually get?* | Unknown until shadow mode. The research-data figure is optimistic because that data has far more attacks than our network does. |
| *Does it read our data?* | No. It uses connection statistics — size, timing, direction — never message contents. |
| *What is the ROI?* | We can model it from our analyst cost and incident history after shadow mode. We are not presenting an invented figure today. |
| *Can attackers evade it?* | Yes, with effort — they can reshape traffic to look ordinary. That is one reason it is a layer rather than a sole control. |
| *How often must it be retrained?* | Traffic changes continuously. Plan for scheduled retraining with drift monitoring; the pilot will tell us the right cadence. |
