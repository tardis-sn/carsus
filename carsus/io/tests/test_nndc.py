from pathlib import Path
import shutil

import pytest

from carsus.io.nuclear import NNDCReader


@pytest.fixture()
def nndc_reader(nndc_dirname):
    return NNDCReader(dirname=nndc_dirname)


@pytest.fixture()
def nndc_reader_http():
    return NNDCReader(remote=True)


@pytest.fixture()
def decay_data(nndc_reader):
    return nndc_reader.decay_data


@pytest.fixture()
def decay_data_http(nndc_reader_http):
    return nndc_reader_http.decay_data


@pytest.mark.parametrize("element, z, parent_e_level, decay_mode, "
                         "decay_mode_value, radiation, rad_subtype, rad_energy", [
                             ("Ni", 28, 0.0, "EC", 100.00, "g", "XR ka2", 6.915),
                             ("Mn", 25, 377.749, "IT", 1.75, "e", "Auger K", 5.190)
                         ])
def test_get_nuclear_dataframe(nndc_reader, element, z, parent_e_level, decay_mode, decay_mode_value,
                               radiation, rad_subtype, rad_energy):
    decay_data = NNDCReader._get_nuclear_decay_dataframe(nndc_reader)
    df = decay_data[(decay_data.Element == element) & (decay_data["Decay Mode"] == decay_mode) &
                    (decay_data["Parent E(level)"] == parent_e_level) &
                    (decay_data["Rad subtype"] == rad_subtype)]

    row = df.iloc[0]
    assert row.Z == z
    assert row["Decay Mode Value"] == decay_mode_value
    assert row.Radiation == radiation

    # Isotope column is not yet defined in _get_nuclear_decay_dataframe()
    with pytest.raises(KeyError):
        assert row.loc["Isotope"]


@pytest.mark.parametrize("index, element, z, parent_e_level, metastable, "
                         "decay_mode, decay_mode_value, radiation, rad_subtype, rad_energy", [
                             ("Ni56", "Ni", 28, 0.0, False, "EC", 100.00, "g", "XR ka2", 6.915),
                             ("Mn52", "Mn", 25, 377.749, True, "IT", 1.75, "g", "XR ka1", 5.899)
                         ])
def test_prepare_nuclear_dataframes(nndc_reader, index, element, z, parent_e_level, metastable,
                                    decay_mode, decay_mode_value, radiation, rad_subtype, rad_energy):
    decay_data = NNDCReader._prepare_nuclear_dataframes(nndc_reader)
    df = decay_data[(decay_data.index == index) & (decay_data["Decay Mode"] == decay_mode) &
                    (decay_data["Decay Mode Value"] == decay_mode_value) &
                    (decay_data["Rad subtype"] == rad_subtype)]

    row = df.loc[index]
    assert row["Element"] == element
    assert row["Z"] == z
    assert row["Parent E(level)"] == parent_e_level
    assert row["Radiation"] == radiation
    assert row["Rad Energy"] == rad_energy


@pytest.mark.parametrize("index, element, z, parent_e_level, metastable, "
                         "decay_mode, decay_mode_value, radiation, rad_subtype, rad_energy", [
                             ("Ni56", "Ni", 28, 0.0, False, "EC", 100.00, "g", "XR ka2", 6.915),
                             ("Mn52", "Mn", 25, 377.749, True, "IT", 1.75, "g", "XR ka1", 5.899)
                         ])
def test_nndc_reader_decay_data(decay_data, index, element, z, parent_e_level, metastable,
                                decay_mode, decay_mode_value, radiation, rad_subtype, rad_energy):
    df = decay_data[(decay_data.index == index) & (decay_data["Decay Mode"] == decay_mode) &
                    (decay_data["Decay Mode Value"] == decay_mode_value) &
                    (decay_data["Rad subtype"] == rad_subtype)]

    row = df.loc[index]
    assert row["Element"] == element
    assert row["Z"] == z
    assert row["Parent E(level)"] == parent_e_level
    assert row["Radiation"] == radiation
    assert row["Rad Energy"] == rad_energy


@pytest.mark.parametrize("index, parent_e_level, decay_mode, decay_mode_value, "
                         "half_life, metastable", [
                             ("Mn52", 0.0, "EC", 100.00, 483062.4, False),
                             ("Mn52", 377.749, "IT", 1.75, 1266.0, True),
                             ("Mn52", 377.749, "EC", 98.22, 1266.0, True)
                         ])
def test_nndc_reader_metastable(decay_data, index, parent_e_level, decay_mode,
                                decay_mode_value, half_life, metastable):
    df = decay_data[(decay_data.index == index) & (decay_data["Decay Mode"] == decay_mode) &
                    (decay_data["Parent E(level)"] == parent_e_level)]

    # picking just the first row of the dataframe for a particular nuclei
    row = df.iloc[0]
    assert row["Decay Mode Value"] == decay_mode_value
    assert row["T1/2 (sec)"] == half_life
    assert row["Metastable"] == metastable


@pytest.mark.parametrize("index, element, z, parent_e_level, metastable, "
                         "decay_mode, decay_mode_value, radiation, rad_subtype, rad_energy", [
                             ("Ni56", "Ni", 28, 0.0, False, "EC", 100.00, "g", "XR ka2", 6.915),
                             ("Mn52", "Mn", 25, 377.749, True, "IT", 1.75, "g", "XR ka1", 5.899)
                         ])
def test_nndc_reader_decay_data_http(decay_data_http, index, element, z, parent_e_level, metastable,
                                     decay_mode, decay_mode_value, radiation, rad_subtype, rad_energy):
    df = decay_data_http[(decay_data_http.index == index) & (decay_data_http["Decay Mode"] == decay_mode) &
                         (decay_data_http["Decay Mode Value"] == decay_mode_value) &
                         (decay_data_http["Rad subtype"] == rad_subtype)]

    row = df.loc[index]
    assert row["Element"] == element
    assert row["Z"] == z
    assert row["Parent E(level)"] == parent_e_level
    assert row["Radiation"] == radiation
    assert row["Rad Energy"] == rad_energy

    # Deleting the repository cloned during the test
    shutil.rmtree(Path.home().joinpath("Downloads", "carsus-data-nndc"))
