import pytest
from carsus.io.seaton.seaton import Seaton1992Reader

LOCAL_FILE = r"D:\Study\Hackhathons\GSOC\s92.201.gz"


@pytest.fixture(scope="module")
def reader():
    return Seaton1992Reader(fpath=LOCAL_FILE)


def test_abundances_shape(reader):
    assert reader.abundances.shape == (17, 2)


def test_abundances_columns(reader):
    assert list(reader.abundances.columns) == ["atomic_number", "abundance"]


def test_abundances_dtypes(reader):
    assert reader.abundances["atomic_number"].dtype == int
    assert reader.abundances["abundance"].dtype == float


def test_opacities_shape(reader):
    assert reader.opacities.shape == (1748, 3)


def test_opacities_columns(reader):
    assert list(reader.opacities.columns) == [
        "log_frequency", "log_temperature", "opacity"
    ]


def test_caching(reader):
    first = reader.abundances
    second = reader.abundances
    assert first is second


def test_invalid_path():
    reader = Seaton1992Reader(fpath="invalid/path.gz")
    with pytest.raises(Exception):
        _ = reader.abundances