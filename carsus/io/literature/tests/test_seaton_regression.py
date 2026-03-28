"""Regression tests for the Seaton opacity parser."""

from pathlib import Path
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

from carsus.io.literature import get_seaton_opacity_df


def _load_sample_gz_bytes():
    data_dir = Path(__file__).parent / "data"
    sample_path = data_dir / "s92.201.sample.gz"
    return sample_path.read_bytes()


def _load_expected_head():
    data_dir = Path(__file__).parent / "data"
    expected_path = data_dir / "expected_head.csv"
    return pd.read_csv(expected_path)


@mock.patch("carsus.io.literature.seaton.requests.get")
def test_get_seaton_opacity_df_regression_head(mock_get):
    """Ensure parser output head remains stable for known Seaton sample input."""
    mock_get.return_value.content = _load_sample_gz_bytes()

    df = get_seaton_opacity_df("s92.201.gz")
    expected_head = _load_expected_head()

    assert_frame_equal(df.head(5).reset_index(drop=True), expected_head)
