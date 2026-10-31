import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from merchant_pulse import synth  # noqa: E402
from merchant_pulse.features import build_features, split_dates  # noqa: E402


@pytest.fixture(scope="session")
def tables():
    return synth.generate(seed=7)


@pytest.fixture(scope="session")
def feats(tables):
    return build_features(tables["sales"], tables["calendar"])


@pytest.fixture(scope="session")
def dates(feats):
    return split_dates(feats)
