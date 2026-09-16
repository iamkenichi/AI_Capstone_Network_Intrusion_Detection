# Submission Bundle

**Machine Learning-Based Network Intrusion Detection and Anomaly Classification**
**Author:** Arne Ramos   ·   **Generated:** 2026-09-16

---

## Files in this folder

| File | Format | What it is |
|---|---|---|
| `Arne_Ramos_AI_Capstone_Network_Intrusion_Detection_Report.docx` | Word | **All written responses collated into one document** — problem statement, dataset documentation, EDA, model evaluation, bias audit, final report, AI-usage disclosure and rubric audit, with every figure embedded. This is the file to upload if only one is accepted. |
| `Arne_Ramos_AI_Capstone_Technical_Presentation.pptx` | PowerPoint | 12-slide technical deck with figures on the slides and speaker notes in the notes pane. |
| `Arne_Ramos_AI_Capstone_Executive_Presentation.pptx` | PowerPoint | 10-slide executive deck — no equations, no code. |
| `DEMO_VIDEO_SCRIPT.md` | Markdown | Shot-by-shot script, timings, narration and recording setup for the demo video. |

## Still to do by hand

1. **Record the demo video** using `DEMO_VIDEO_SCRIPT.md`. Export as
   `Arne_Ramos_AI_Capstone_Demo_Video.mp4`.
2. **Open the Word document and update the Table of Contents** — right-click the
   contents field, choose *Update Field* → *Update entire table*. Word cannot
   populate it without being opened once.
3. **Check the decks in Presenter View** so you can see the speaker notes.
4. If the portal caps upload size, host the video (YouTube *Unlisted* or Drive)
   and paste the link onto slide 1 of the technical deck and the report cover page.

## Converting to PDF, if required

Open each file in Word or PowerPoint → **File → Export → Create PDF/XPS**.
Or from a terminal, if LibreOffice is installed:

```bash
soffice --headless --convert-to pdf deliverables/*.docx deliverables/*.pptx
```

## Regenerating this bundle

```bash
python -m src.pipeline --all     # recompute everything
python -m src.deliverables       # rebuild the .docx and .pptx files
```

Every number in these documents is read from `reports/metrics/`, which is
written by executed code — so the bundle cannot drift out of step with the
results.
