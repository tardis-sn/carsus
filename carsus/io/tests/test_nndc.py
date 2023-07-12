import pytest

from carsus.io.nuclear import NNDCReader


@pytest.fixture()
def nndc_reader(nndc_fname):
    return NNDCReader(fname=nndc_fname)


@pytest.fixture()
def decay_data(nndc_reader):
    return nndc_reader.decay_data


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
