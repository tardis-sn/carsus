import pytest
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose
from carsus.io.kurucz import GFALLReader

@pytest.fixture()
def gfall_rdr(gfall_fname):
    return GFALLReader(fname=gfall_fname)


@pytest.fixture()
def gfall_rdr_http(gfall_http):
    return GFALLReader(fname=gfall_http)


@pytest.fixture()
def gfall_raw(gfall_rdr):
    return gfall_rdr.gfall_raw


@pytest.fixture()
def gfall_raw_http(gfall_rdr_http):
    return gfall_rdr_http.gfall_raw


@pytest.fixture()
def gfall(gfall_rdr):
    return gfall_rdr.gfall


@pytest.fixture()
def levels(gfall_rdr):
    return gfall_rdr.levels


@pytest.fixture()
def lines(gfall_rdr):
    return gfall_rdr.lines


@pytest.mark.parametrize("index, wavelength, element_code, e_first, e_second",[
    (14, 72.5537, 4.02, 983355.0, 1121184.0),
    (37, 2.4898, 7.05, 0.0, 4016390.0)
])
def test_grall_reader_gfall_raw(gfall_raw, index, wavelength, element_code, e_first, e_second):
    row = gfall_raw.loc[index]
    assert_almost_equal(row["element_code"], element_code)
    assert_almost_equal(row["wavelength"], wavelength)
    assert_allclose([row["e_first"], row["e_second"]], [e_first, e_second])

@pytest.mark.remote_data
@pytest.mark.parametrize("index, wavelength, element_code, e_first, e_second",[
    (14, 72.5537, 4.02, 983355.0, 1121184.0),
    (37, 2.4898, 7.05, 0.0, 4016390.0)
])
def test_grall_reader_gfall_raw_http(gfall_raw_http, index, wavelength, element_code, e_first, e_second):
    row = gfall_raw_http.loc[index]
    assert_almost_equal(row["element_code"], element_code)
    assert_almost_equal(row["wavelength"], wavelength)
    assert_allclose([row["e_first"], row["e_second"]], [e_first, e_second])


@pytest.mark.parametrize("index, wavelength, atomic_number, ion_charge, "
                         "energy_lower, energy_upper, energy_lower_predicted, energy_upper_predicted",[
    (12, 67.5615, 4, 2, 983369.8, 1131383.0, False, False),
    (17, 74.6230, 4, 2, 997455.000, 1131462.0, False, False),
    (41, 16.1220, 7, 5, 3385890.000, 4006160.0, False, True)
])
def test_gfall_reader_gfall(gfall, index, wavelength, atomic_number, ion_charge,
                               energy_lower, energy_upper, energy_lower_predicted, energy_upper_predicted):
    row = gfall.loc[index]
    assert row["atomic_number"] == atomic_number
    assert row["ion_charge"] == ion_charge
    assert_allclose([row["wavelength"], row["energy_lower"], row["energy_upper"]],
                    [wavelength, energy_lower, energy_upper])
    assert row["energy_lower_predicted"] == energy_lower_predicted
    assert row["energy_upper_predicted"] == energy_upper_predicted


def test_gfall_reader_gfall_ignore_labels(gfall):
    ignored_labels = ["AVERAGE", "ENERGIES", "CONTINUUM"]
    assert len(gfall.loc[(gfall["label_lower"].isin(ignored_labels)) |
                         (gfall["label_upper"].isin(ignored_labels))]) == 0


def test_gfall_reader_clean_levels_labels(levels):
    # One label for the ground level of Be III has an extra space
    levels0402 = levels.loc[(4, 2)]
    assert len(levels0402.loc[(np.isclose(levels0402["energy"], 0.0))]) == 1


@pytest.mark.parametrize("atomic_number, ion_charge, level_index, "
                         "energy, j, method",[
    (4, 2, 0, 0.0, 0.0, "meas"),
    (4, 2, 11, 1128300.0, 2.0, "meas"),
    (7, 5, 7, 4006160.0, 0.0,  "theor")
])
def test_gfall_reader_levels(levels, atomic_number, ion_charge, level_index,
                             energy, j, method):
    row = levels.loc[(atomic_number, ion_charge, level_index)]
    assert_almost_equal(row["energy"], energy)
    assert_almost_equal(row["j"], j)
    assert row["method"] == method


@pytest.mark.parametrize("atomic_number, ion_charge, level_index_lower, level_index_upper,"
                         "wavelength, gf",[
    (4, 2, 0, 16, 8.8309, 0.12705741),
    (4, 2, 6, 15, 74.6230, 2.1330449131)
])
def test_gfall_reader_lines(lines, atomic_number, ion_charge,
                            level_index_lower, level_index_upper, wavelength, gf):
    row = lines.loc[(atomic_number, ion_charge, level_index_lower, level_index_upper)]
    assert_almost_equal(row["wavelength"], wavelength)
    assert_almost_equal(row["gf"], gf)


@pytest.mark.remote_data
def test_gfall_hash(gfall_rdr):
    gf = gfall_rdr
    # Need to generate `gfall_raw` lazy attribute to get `md5`.
    gf_raw = gf.gfall_raw

    assert gf.version == 'e2149a67d52b7cb05fa5d35e6912cc98'


@pytest.mark.remote_data
def test_gfall_hash_http(gfall_rdr_http):
    gf = gfall_rdr_http
    gf_raw = gf.gfall_raw

    assert gf.version == 'e2149a67d52b7cb05fa5d35e6912cc98'