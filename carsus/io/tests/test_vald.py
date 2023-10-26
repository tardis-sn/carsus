import pytest

from numpy.testing import assert_almost_equal, assert_allclose
from carsus.io.vald import VALDReader


@pytest.fixture()
def vald_rdr(vald_fname):
    return VALDReader(fname=vald_fname, strip_molecules=False)


@pytest.fixture()
def vald_rdr_stripped_molecules(vald_fname):
    return VALDReader(fname=vald_fname, strip_molecules=True)


@pytest.fixture()
def vald_raw(vald_rdr):
    return vald_rdr.vald_raw


@pytest.fixture()
def vald(vald_rdr):
    return vald_rdr.vald


@pytest.fixture()
def vald_stripped_molecules(vald_rdr_stripped_molecules):
    return vald_rdr_stripped_molecules.vald


@pytest.fixture()
def vald_linelist(vald_rdr_stripped_molecules):
    return vald_rdr_stripped_molecules.linelist


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


def test_vald_reader_strip_molecules(vald, vald_stripped_molecules):
    # The stripped molecules list should always be smaller or equal to the size of the non-stripped list
    assert len(vald) >= len(vald_stripped_molecules)
    # There is only a single non-molecule in the current test vald file
    assert len(vald_stripped_molecules) == 1


def test_vald_linelist(vald_linelist):
    assert all(
        vald_linelist.columns
        == [
            "atomic_number",
            "ion_charge",
            "wavelength",
            "log_gf",
            "e_low",
            "e_up",
            "j_lo",
            "j_up",
            "rad",
            "stark",
            "waals",
        ]
    )
    # Test to see if any values have become nan in new columns
    assert ~vald_linelist.isna().values.any()
