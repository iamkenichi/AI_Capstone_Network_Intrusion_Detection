# UNSW-NB15 acquisition

From the repository root run `python scripts/download_data.py`.
The helper uses the public [mirror](https://github.com/jamshaid120/UNSW_NB15-Complete-dataset), normalizes its reversed training/testing filenames by the official partition counts, and writes URL/hash provenance to `reports/data_provenance.json`.

For manual acquisition, open the [official UNSW page](https://research.unsw.edu.au/projects/unsw-nb15-dataset) and follow its dataset download link. Download these files into this `data/raw/` directory:

- `UNSW_NB15_training-set.csv` — the partition with 175,341 records.
- `UNSW_NB15_testing-set.csv` — the partition with 82,332 records.
- `UNSW-NB15_features.csv` — the source dictionary; save a copy as `NUSW-NB15_features.csv` for compatibility with the mirrored dictionary filename.

If using the mirror manually, its filenames are reversed relative to the published counts; verify and rename accordingly. Never mix files merely because their names match. The original full dataset shards and raw packet captures are not needed for this binary-classification experiment. Do not commit raw CSVs to Git. Academic-use terms and requested citations remain those of the dataset authors.
