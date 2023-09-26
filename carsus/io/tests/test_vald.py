import pytest

from numpy.testing import assert_almost_equal, assert_allclose
from carsus.io.vald import VALDReader


@pytest.fixture()
def vald_rdr(vald_fname):
    return VALDReader(fname=vald_fname, strip_molecules=False)


@pytest.fixture()
def vald_raw(vald_rdr):
    return vald_rdr.vald_raw


@pytest.fixture()
def vald(vald_rdr):
    return vald_rdr.vald


@pytest.mark.parametrize(
    "index, wl_air, log_gf, e_low, e_up",
    [
        (0, 4100.00020, -11.472, 0.2011, 3.2242),
        (24, 4100.00560, -2.967, 1.6759, 4.6990),
    ],
)
def test_vald_reader_vald_raw(vald_raw, index, wl_air, log_gf, e_low, e_up):
    row = vald_raw.loc[index]
    assert_almost_equal(row["wl_air"], wl_air)
    assert_allclose([row["log_gf"], row["e_low"], row["e_up"]], [log_gf, e_low, e_up])


@pytest.mark.parametrize(
    "index, wl_air, log_gf, e_low, e_up, ion_charge",
    [
        (0, 4100.00020, -11.472, 0.2011, 3.2242, 0),
        (24, 4100.00560, -2.967, 1.6759, 4.6990, 0),
    ],
)
def test_vald_reader_vald(vald, index, wl_air, log_gf, e_low, e_up, ion_charge):
    row = vald.loc[index]
    assert_almost_equal(row["wl_air"], wl_air)
    assert_allclose(
        [row["log_gf"], row["e_low"], row["e_up"], row["ion_charge"]],
        [log_gf, e_low, e_up, ion_charge],
    )
