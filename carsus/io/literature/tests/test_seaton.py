"""
Tests for Seaton opacity reader.
"""

import gzip
import io
from unittest import mock

import pandas as pd
import pytest

from carsus.io.literature import get_seaton_opacity_df


def create_sample_data():
    """Create gzip-compressed sample Seaton data."""
    sample_lines = [
        "OP Version 2.0 S92 201",
        "140        14   68    2",
        "14    -4.0 1.23e-1 1.15e-1",
        "16    -3.5 1.45e-1 1.32e-1",
    ]
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode='wb') as f:
        f.write("\n".join(sample_lines).encode('utf-8'))
    return buffer.getvalue()


@mock.patch('carsus.io.literature.seaton.requests.get')
def test_get_seaton_opacity_df(mock_get):
    """Test reading Seaton opacity data."""
    mock_get.return_value.content = create_sample_data()
    
    df = get_seaton_opacity_df("s92.201.gz")
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert list(df.columns) == ['logT', 'logNe', 'kappa_planck', 'kappa_rosseland']
    assert (df['kappa_planck'] > 0).all()
    assert (df['kappa_rosseland'] > 0).all()


