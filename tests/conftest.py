"""Shared test fixtures."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir() -> Path:
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_fasta(test_data_dir: Path) -> Path:
    return test_data_dir / "sample_ecoli.fasta"
