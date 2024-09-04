import pytest
import pandas as pd
from carsus.io.zeta import KnoxLongZeta, ZETA_DATA_URL

@pytest.fixture
def reference_file_path():
    return "carsus/data/knox_long_recombination_zeta.dat"

def test_knoxlongzeta_init_with_default_url():
    zeta = KnoxLongZeta()
    assert zeta.fname == ZETA_DATA_URL

def test_knoxlongzeta_init_with_custom_file(reference_file_path):
    zeta = KnoxLongZeta(fname=reference_file_path)
    assert zeta.fname == reference_file_path

def test_knoxlongzeta_prepare_data(reference_file_path):
    zeta = KnoxLongZeta(reference_file_path)
    zeta._prepare_data()

    expected_columns = [float(i) for i in range(2000, 42000, 2000)]
    assert isinstance(zeta.base, pd.DataFrame)
    assert list(zeta.base.columns) == expected_columns
    assert zeta.base.index.names == ["atomic_number", "ion_charge"]
    assert zeta.version == "a1d4bed2982e8d6a4f8b0076bf637e49"