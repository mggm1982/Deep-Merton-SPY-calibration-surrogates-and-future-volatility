from pathlib import Path
import json
import pytest


@pytest.fixture
def root():
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def config(root):
    return json.loads((root / "config.json").read_text())
