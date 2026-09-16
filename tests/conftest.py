"""
Shared pytest fixtures.

Tests that need the real corpus are given it once per session (loading
257,673 rows repeatedly would dominate the runtime). Tests that only need
*a valid flow record* get a small synthetic frame instead, so the bulk of the
suite runs without any data on disk at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import config  # noqa: E402

RAW_AVAILABLE = (config.RAW_DIR / config.TRAIN_FILE).exists()
MODEL_AVAILABLE = (config.MODELS_DIR / "deployment.json").exists()

requires_data = pytest.mark.skipif(
    not RAW_AVAILABLE,
    reason="UNSW-NB15 not present in data/raw/. Run `python -m src.data_loader`.")

requires_model = pytest.mark.skipif(
    not MODEL_AVAILABLE,
    reason="No trained model. Run `python -m src.pipeline --all`.")


@pytest.fixture(scope="session")
def corpus() -> pd.DataFrame:
    """The full combined UNSW-NB15 corpus (session-scoped: loaded once)."""
    from src import data_loader
    return data_loader.load_corpus()


@pytest.fixture(scope="session")
def small_corpus(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    A stratified sample of up to 800 rows per attack family.

    Large enough to exercise every code path - including the rare families that
    stratified splitting has to cope with - and small enough that the whole
    suite runs in seconds rather than minutes.
    """
    positions: list[int] = []
    for _, index in corpus.groupby(config.ATTACK_CAT, observed=True).indices.items():
        take = min(len(index), 800)
        rng = np.random.default_rng(config.RANDOM_STATE)
        positions.extend(rng.choice(index, size=take, replace=False).tolist())
    return corpus.iloc[sorted(positions)].reset_index(drop=True)


@pytest.fixture
def synthetic_flows() -> pd.DataFrame:
    """
    Hand-built flow records exercising the awkward cases.

    Row 0: an ordinary bidirectional TCP session.
    Row 1: a one-way probe - zero destination packets and bytes, zero duration.
           This is the row that breaks any unguarded division.
    Row 2: an extreme-volume flow, to check nothing overflows.
    """
    from src.predict import REQUIRED_INPUT_COLUMNS

    base = {column: 0 for column in REQUIRED_INPUT_COLUMNS}
    base.update({"proto": "tcp", "service": "http", "state": "fin"})

    ordinary = {**base, "dur": 1.25, "spkts": 12, "dpkts": 10, "sbytes": 1400,
                "dbytes": 3200, "rate": 17.6, "sttl": 62, "dttl": 252,
                "sload": 8960.0, "dload": 20480.0, "sloss": 1, "dloss": 0,
                "sinpkt": 104.0, "dinpkt": 125.0, "sjit": 12.5, "djit": 9.1,
                "swin": 255, "dwin": 255, "stcpb": 123456, "dtcpb": 654321,
                "tcprtt": 0.08, "synack": 0.04, "ackdat": 0.04,
                "smean": 117, "dmean": 320, "ct_srv_src": 3, "ct_state_ttl": 1,
                "ct_dst_ltm": 2, "ct_src_dport_ltm": 1, "ct_dst_sport_ltm": 1,
                "ct_dst_src_ltm": 2, "ct_src_ltm": 3, "ct_srv_dst": 3}

    one_way = {**base, "proto": "udp", "service": "-", "state": "int",
               "dur": 0.0, "spkts": 2, "dpkts": 0, "sbytes": 200, "dbytes": 0,
               "rate": 0.0, "sttl": 254, "dttl": 0, "sload": 0.0, "dload": 0.0,
               "smean": 100, "dmean": 0, "ct_srv_src": 40, "ct_state_ttl": 2,
               "ct_dst_ltm": 30, "ct_src_dport_ltm": 28, "ct_dst_sport_ltm": 25,
               "ct_dst_src_ltm": 30, "ct_src_ltm": 35, "ct_srv_dst": 40}

    extreme = {**ordinary, "sbytes": 14_355_774, "dbytes": 14_657_531,
               "spkts": 10_646, "dpkts": 11_018, "sload": 5.988e9,
               "dload": 2.24e7, "rate": 1_000_000.0, "dur": 59.999999,
               "sjit": 1_483_831.0, "response_body_len": 6_558_056}

    return pd.DataFrame([ordinary, one_way, extreme])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(config.RANDOM_STATE)
