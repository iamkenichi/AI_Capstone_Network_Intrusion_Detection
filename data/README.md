# `data/`

The dataset is **not committed** to this repository. It is a 48 MB public
research dataset that anyone can re-fetch in under a minute, and committing it
would bloat the repository while adding nothing a checksum cannot guarantee.

```bash
python -m src.data_loader
```

That command downloads both partitions, verifies them, and writes them to
`data/raw/`. The SHA-256 of the exact files used to produce the committed
results is recorded in
[`reports/dataset_documentation.md`](../reports/dataset_documentation.md) §2, so
you can confirm you have the same bytes.

---

## Layout

| Directory | Contents | Committed? |
|---|---|---|
| `raw/` | The two official UNSW-NB15 partitioned CSVs, plus the authors' feature-description file | No |
| `interim/` | Reserved for intermediate artefacts | No |
| `processed/` | `train.parquet`, `val.parquet`, `test.parquet` — the materialised stratified splits | No |

`processed/` is written by `python -m src.preprocessing`. It exists so the
notebooks, the tests and the training scripts all read **byte-identical**
splits rather than each re-deriving them — which also makes it obvious if a
split ever changes.

---

## If automatic download fails

The loader prints exact instructions, but for reference:

### Files needed

| File | Rows | Columns |
|---|---|---|
| `UNSW_NB15_training-set.csv` | 175,341 | 45 |
| `UNSW_NB15_testing-set.csv` | 82,332 | 45 |

### Where to get them

1. **Official project page** — UNSW Canberra, Australian Centre for Cyber Security
   <https://research.unsw.edu.au/projects/unsw-nb15-dataset>
   Follow the *CSV Files* link, then the folder
   *"a part of training and testing set"*.
2. **IEEE DataPort** — DOI `10.21227/8vf7-s525`
   <https://ieee-dataport.org/documents/unswnb15-dataset>
3. **Hugging Face mirror** — <https://huggingface.co/datasets/Mouwiya/UNSW-NB15>
4. **Kaggle mirror** — <https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15>

> The original UNSW CloudStor host referenced in older papers has been retired.

### Where to put them

Copy both files, **keeping their exact filenames**, into `data/raw/`:

```
data/raw/UNSW_NB15_training-set.csv
data/raw/UNSW_NB15_testing-set.csv
```

Then re-run:

```bash
python -m src.data_loader
```

It will verify row counts, column count, target encoding and the
`label`/`attack_cat` consistency rule before proceeding. If any check fails it
stops with a specific message rather than silently continuing on the wrong data.

---

## Citation

Using this dataset requires citing its authors:

> Moustafa, N. and Slay, J. (2015). *UNSW-NB15: a comprehensive data set for
> network intrusion detection systems (UNSW-NB15 network data set).* Military
> Communications and Information Systems Conference (MilCIS), IEEE.

> Moustafa, N. and Slay, J. (2016). *The evaluation of Network Anomaly Detection
> Systems: Statistical analysis of the UNSW-NB15 data set and the comparison
> with the KDD99 data set.* Information Security Journal: A Global Perspective,
> 25(1-3), 18-31.

---

## Privacy note

UNSW-NB15 is **synthetic traffic generated on a closed testbed**. It contains no
real users, no personal data and no production network telemetry. The
partitioned CSVs used here additionally omit all IP addresses, port numbers and
timestamps — so nothing in this project's inputs could identify a person or an
organisation.

That absence is also a methodological constraint, not only a privacy benefit:
without timestamps **no temporal holdout is possible**, so concept drift cannot
be measured on this data. See `reports/Bias_Fairness_Analysis.md` §6.3.
