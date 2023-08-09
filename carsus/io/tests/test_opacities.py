import pytest
import numpy as np

from numpy.testing import assert_almost_equal
from carsus.io.opacities import HMINUSOPACITIESReader


@pytest.fixture()
def h_minus_rdr(h_minus_fname):
    return HMINUSOPACITIESReader(fname=h_minus_fname)


@pytest.fixture()
def h_minus_rdr_http(h_minus_http):
    return HMINUSOPACITIESReader(fname=h_minus_http)


@pytest.fixture()
def h_minus(h_minus_rdr):
    return h_minus_rdr.h_minus


@pytest.mark.parametrize(
    "index, wavelength, cross_section",
    [
        (7, 404.0, 7.03e-19),
        (22, 4000.0, 2.3020000000000002e-17),
        (64, 14500.0, 1.107e-17),
    ],
)
def test_h_minus_reader_h_minus(
    h_minus,
    index,
    wavelength,
    cross_section,
):
    row = h_minus.loc[index]
    assert_almost_equal(row["wavelength"], wavelength)
    assert_almost_equal(row["cross_section"], cross_section)


@pytest.mark.remote_data
@pytest.mark.parametrize(
    "index, wavelength, cross_section",
    [
        (7, 404.0, 7.03e-19),
        (22, 4000.0, 2.3020000000000002e-17),
        (64, 14500.0, 1.107e-17),
    ],
)
def test_h_minus_reader_h_minus_http(
    h_minus,
    index,
    wavelength,
    cross_section,
):
    row = h_minus.loc[index]
    assert_almost_equal(row["wavelength"], wavelength)
    assert_almost_equal(row["cross_section"], cross_section)


@pytest.mark.remote_data
def test_h_minus_hash(h_minus_rdr):
    hm = h_minus_rdr
    h_minus = hm.h_minus

    assert hm.version == "11b033b4a697df1a91b8ea9d54b12e1e"


@pytest.mark.remote_data
def test_h_minus_hash_http(h_minus_rdr_http):
    hm = h_minus_rdr_http
    h_minus = hm.h_minus

    assert hm.version == "11b033b4a697df1a91b8ea9d54b12e1e"
