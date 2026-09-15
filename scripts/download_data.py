"""Download the same UNSW-NB15 partitions from a public mirror with provenance."""
from pathlib import Path
import csv
import hashlib
import io
import json
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://media.githubusercontent.com/media/jamshaid120/UNSW_NB15-Complete-dataset/main/'
EXPECTED = {
    'UNSW_NB15_training-set.csv':'734fe6642edf758f7c94d7d9149426b49d202fe8e7bf0bef47392489c3c0a559',
    'UNSW_NB15_testing-set.csv':'bec7dd5ec88dc2a0ccc7a07879d338395ed7421750f675fd0339e07dfe0648fa',
    'NUSW-NB15_features.csv':'c55f19cceebb6360dc50f44f8a5f246ccefbcf8a6c604ac1ad46e643869cafce'}


def main():
    raw = ROOT / 'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    records = []
    for source in ['UNSW_NB15_training-set.csv', 'UNSW_NB15_testing-set.csv', 'NUSW-NB15_features.csv']:
        content = urllib.request.urlopen(BASE + source, timeout=90).read()
        if hashlib.sha256(content).hexdigest() != EXPECTED[source]:
            raise ValueError(f'{source} differs from the recorded experiment checksum; investigate before replacing data.')
        rows = list(csv.reader(io.StringIO(content.decode('utf-8-sig', errors='replace'))))
        count = len(rows)-1
        if 'features' in source:
            target = source
        elif count == 175341:
            target = 'UNSW_NB15_training-set.csv'
        elif count == 82332:
            target = 'UNSW_NB15_testing-set.csv'
        else:
            raise ValueError(f'Unexpected record count {count} for {source}; inspect source manually.')
        record = dict(file=target, source_filename=source, url=BASE+source,
            sha256=hashlib.sha256(content).hexdigest(), rows=count, bytes=len(content),
            retrieved_utc=datetime.now(timezone.utc).isoformat())
        (raw / target).write_bytes(content)
        records.append(record)
        print(record, flush=True)
    (ROOT / 'reports/data_provenance.json').write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
