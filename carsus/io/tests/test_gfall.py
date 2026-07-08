import sqlite3

import pytest
import numpy as np

from numpy.testing import assert_almost_equal, assert_allclose
from carsus.io.kurucz import GFALLReader, SQLiteGFALLReader



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

@pytest.mark.parametrize("index, wavelength, element_code, e_first, e_second",[
    (14, 72.5537, 4.02, 983355.0, 1121184.0),
    (37, 2.4898, 7.05, 0.0, 4016390.0)
])
@pytest.mark.remote_data
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


def test_gfall_reader_normalizes_label_whitespace_for_unique_levels(gfall_fname):
    reader = GFALLReader(
        fname=gfall_fname,
        unique_level_identifier=["energy", "j", "label"],
    )

    levels0402 = reader.levels.loc[(4, 2)]
    assert len(levels0402.loc[(np.isclose(levels0402["energy"], 0.0))]) == 1


def test_sqlite_gfall_reader_preserves_historical_o_iv_fine_structure(tmp_path):
    sqlite_path = tmp_path / "gfall.db3"
    rows = [
        (24.7010, -0.55, 8, 3, 659998.0, 255155.9, 0.5, 1.5, '3d"" 2P', "2p3 *2D"),
        (24.7028, -0.86, 8, 3, 659998.0, 255184.9, 0.5, 0.5, '3d"" 2P', "2p3 *2D"),
        (133.8612, -0.09, 8, 3, 255184.9, 180480.8, 1.5, 0.5, "2p3 *2D", "s2p2 2P"),
        (134.3512, 0.17, 8, 3, 255155.9, 180724.2, 2.5, 1.5, "2p3 *2D", "s2p2 2P"),
    ]
    with sqlite3.connect(sqlite_path) as connection:
        connection.execute(
            """
            CREATE TABLE gfall (
                wavelength REAL,
                loggf REAL,
                atomic_number INTEGER,
                ion_number INTEGER,
                e_upper REAL,
                e_lower REAL,
                j_upper REAL,
                j_lower REAL,
                label_upper TEXT,
                label_lower TEXT
            )
            """
        )
        connection.executemany(
            "INSERT INTO gfall VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

    reader = SQLiteGFALLReader(
        fname=str(sqlite_path),
        unique_level_identifier=["energy", "j", "label"],
    )
    levels = reader.levels.loc[(8, 3)].reset_index()
    o_iv_levels = levels.loc[
        levels["energy"].isin([255155.9, 255184.9]), ["energy", "j", "label"]
    ].sort_values(["energy", "j"])

    assert o_iv_levels.to_records(index=False).tolist() == [
        (255155.9, 1.5, "2p3 *2D"),
        (255155.9, 2.5, "2p3 *2D"),
        (255184.9, 0.5, "2p3 *2D"),
        (255184.9, 1.5, "2p3 *2D"),
    ]


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
