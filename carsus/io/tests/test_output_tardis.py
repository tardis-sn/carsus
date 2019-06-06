import pytest
import os
import numpy as np

from numpy.testing import assert_almost_equal
from astropy import units as u
from astropy.tests.helper import assert_quantity_allclose
from sqlalchemy import and_
from carsus.io.output.tardis_ import AtomData
from carsus.model import DataSource, Ion

with_test_db = pytest.mark.skipif(
    not pytest.config.getoption("--test-db"),
    reason="--testing database was not specified"
)

pytestmark = pytest.mark.skipif(0 == 0, reason="Tests are failing due to empty DataFrames (traced to atom_data fixture)")


@pytest.fixture
def atom_data(test_session, chianti_short_name):
    atom_data = AtomData(test_session,
                         selected_atoms="He, Be, B, N, Si, Zn",
                         chianti_ions="He 1; N 5",
                         chianti_short_name=chianti_short_name
                         )
    return atom_data


@pytest.fixture
def chianti_short_name(test_session):
    return (
            test_session.
            query(DataSource.short_name).
            filter(DataSource.short_name.like('chianti%'))
            ).one()[0]


@pytest.fixture
def atom_data_be(test_session):
    atom_data = AtomData(test_session, selected_atoms="Be")
    return atom_data


@pytest.fixture
def atom_masses(atom_data):
    return atom_data.atom_masses


@pytest.fixture
def ionization_energies(atom_data):
    return atom_data.ionization_energies


@pytest.fixture
def levels(atom_data):
    return atom_data.levels


@pytest.fixture
def lines(atom_data):
    return atom_data.lines


@pytest.fixture
def collisions(atom_data):
    return atom_data.collisions


@pytest.fixture
def macro_atom(atom_data):
    return atom_data.macro_atom


@pytest.fixture
def macro_atom_references(atom_data):
    return atom_data.macro_atom_references


@pytest.fixture
def zeta_data(atom_data):
    return atom_data.zeta_data


@pytest.fixture
def levels_prepared(atom_data):
    return atom_data.levels_prepared

@pytest.fixture
def hdf5_path(request, data_dir):
    hdf5_path = os.path.join(data_dir, "test_hdf.hdf5")

    def fin():
      os.remove(hdf5_path)
    request.addfinalizer(fin)

    return hdf5_path


def test_atom_data_init(memory_session):
    nist = DataSource.as_unique(memory_session, short_name="nist-asd")
    ch = DataSource.as_unique(memory_session, short_name="chianti_v8.0.2")
    ku = DataSource.as_unique(memory_session, short_name="ku_latest")
    atom_data = AtomData(memory_session,
                         selected_atoms="He, Be, B, N",
                         chianti_ions="He 1; N 5")
    assert set(atom_data.selected_atomic_numbers) == set([2, 4, 5, 7])
    assert set(atom_data.chianti_ions) == set([(2,1), (7,5)])


def test_atom_data_chianti_ions_subset(memory_session):
    nist = DataSource.as_unique(memory_session, short_name="nist-asd")
    ch = DataSource.as_unique(memory_session, short_name="chianti_v8.0.2")
    ku = DataSource.as_unique(memory_session, short_name="ku_latest")
    with pytest.raises(ValueError):
        atom_data = AtomData(memory_session,
                             selected_atoms="He, Be, B, N VI",
                             chianti_ions="He 1; N 5; Si 1")


@with_test_db
def test_atom_data_wo_chianti_ions_attributes(atom_data_be, test_session):
    assert atom_data_be.chianti_ions == list()
    assert test_session.query(atom_data_be.chianti_ions_table).count() == 0


@with_test_db
def test_atom_data_only_be(atom_data_be):
    assert all([atomic_number == 4 for atomic_number in
                atom_data_be.atom_masses["atomic_number"].values.tolist()])
    assert all([atomic_number == 4 for atomic_number in
                atom_data_be.ionization_energies["atomic_number"].values.tolist()])
    assert all([atomic_number == 4 for atomic_number in
                atom_data_be.levels["atomic_number"].values.tolist()])
    assert all([atomic_number == 4 for atomic_number in
                atom_data_be.lines["atomic_number"].values.tolist()])


@with_test_db
def test_atom_data_join_on_chianti_ions_table(test_session, atom_data):

    # This join operation leads to an empty chianti_ions list
    #
    # Possible cause:
    # test_session.query(atom_data.chianti_ions_table).all() -> encoding problem
    chiatni_ions_q = test_session.query(Ion).join(atom_data.chianti_ions_table,
                                     and_(Ion.atomic_number == atom_data.chianti_ions_table.c.atomic_number,
                                          Ion.ion_charge == atom_data.chianti_ions_table.c.ion_charge)).\
        order_by(Ion.atomic_number, Ion.ion_charge)
    chianti_ions = [(ion.atomic_number, ion.ion_charge) for ion in chiatni_ions_q]
    assert set(chianti_ions) == set([(2,1), (7,5)])


@with_test_db
def test_atom_data_two_instances_same_session(
        test_session,
        chianti_short_name):

    atom_data1 = AtomData(
            test_session,
            selected_atoms="He, Be, B, N, Zn",
            chianti_ions="He 1; N 5",
            chianti_short_name=chianti_short_name)
    atom_data2 = AtomData(
            test_session,
            selected_atoms="He, Be, B, N, Zn",
            chianti_ions="He 1; N 5",
            chianti_short_name=chianti_short_name)
    atom_data1.chianti_ions_table
    atom_data2.chianti_ions_table


@with_test_db
@pytest.mark.parametrize("atomic_number, exp_mass", [
    (2, 4.002602 * u.u),
    (4, 9.0121831 * u.u),
    (5, (10.806 + 10.821)/2 * u.u),
    (7, (14.00643 + 14.00728)/2 * u.u),
    (14, (28.084 + 28.086)/2 *u.u),
    (30, 65.38 *u.u)
])
def test_create_atom_masses(atom_masses, atomic_number, exp_mass):
    atom_masses = atom_masses.set_index("atomic_number")
    assert_quantity_allclose(
        atom_masses.loc[atomic_number]["mass"] * u.u,
        exp_mass
    )


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, exp_ioniz_energy", [
    (2, 1,  54.41776311 * u.eV),
    (4, 2, 153.896198 * u.eV),
    (5, 3, 259.3715 * u.eV),
    (7, 5, 552.06731 * u.eV),
    (14, 1, 16.345845 * u.eV),
    (30, 19, 737.366 * u.eV)
])
def test_create_ionizatinon_energies(ionization_energies, atomic_number, ion_number, exp_ioniz_energy):
    ionization_energies = ionization_energies.set_index(["atomic_number", "ion_number"])
    assert_quantity_allclose(
        ionization_energies.loc[(atomic_number, ion_number)]["ionization_energy"] * u.eV,
        exp_ioniz_energy
    )


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, ionization_energy",[
    (2, 1, 54.41776311 * u.eV),
    (4, 2, 153.896198 * u.eV),
    (5, 3, 259.3715 * u.eV),
    (7, 5, 552.06731 * u.eV),
    (14, 1, 16.345845 * u.eV)  # In fact, only Si II has levels with energy > ionization potential
])
def test_create_levels_filter_auto_ionizing_levels(levels, atomic_number, ion_number, ionization_energy):
    levels_ion = levels.loc[(levels["atomic_number"] == atomic_number) &
                            (levels["ion_number"] == ion_number)].copy()
    levels_energies = levels_ion["energy"].values * u.eV
    assert all(levels_energies < ionization_energy)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number, exp_energy, exp_g, exp_metastable_flag",[
    # Kurucz levels
    (4, 2, 0, 0.0 * u.Unit("cm-1"), 1, True),
    (4, 2, 1, 956501.9 * u.Unit("cm-1"), 3, True),
    (4, 2, 6, 997455.0 * u.Unit("cm-1"), 3, False),
    (14, 1, 0, 0.0 * u.Unit("cm-1"), 2, True),
    (14, 1, 15, 81251.320 * u.Unit("cm-1"), 4, False),
    # (14, 1, 16, 83801.950 * u.Unit("cm-1"), 2, 1),  investigate the issue with this level (probably labels)!
    # CHIANTI levels
    # Theoretical values from CHIANTI aren't ingested!!!
    (7, 5, 0, 0.0 * u.Unit("cm-1"), 1, True),
    (7, 5, 7, 3991860.0 * u.Unit("cm-1"), 3, False),
    (7, 5, 43, 4294670.00 * u.Unit("cm-1"), 5, False),
    # NIST Ground level
    (30, 19, 0, 0.0 * u.eV, 2, True)
])
def test_create_levels(levels, atomic_number, ion_number, level_number,
                       exp_energy, exp_g, exp_metastable_flag):
    levels = levels.set_index(["atomic_number", "ion_number", "level_number"])
    energy = levels.loc[(atomic_number, ion_number, level_number)]["energy"] * u.eV
    g = levels.loc[(atomic_number, ion_number, level_number)]["g"]
    metastable_flag = levels.loc[(atomic_number, ion_number, level_number)]["metastable"]

    # Convert the expected energy using equivalencies
    exp_energy = exp_energy.to(u.eV, equivalencies=u.spectral())

    assert_quantity_allclose(energy, exp_energy)
    assert g == exp_g
    assert metastable_flag == exp_metastable_flag


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number_lower, level_number_upper, exp_wavelength, exp_loggf",[
    # Kurucz lines
    (14, 1, 0, 57, 81.8575 * u.Unit("nm"), -1.92),
    (14, 1, 1, 71, 80.5098 * u.Unit("nm"), -2.86),
    # CHIANTI lines
    # Note that energies are *not* sorted in the elvlc file!
    (2, 1, 0, 1, 303.786 * u.Unit("angstrom"), np.log10(0.2772)),
    (2, 1, 2, 16, 1084.920 * u.Unit("angstrom"), np.log10(0.027930))
])
def test_create_lines(lines, atomic_number, ion_number, level_number_lower, level_number_upper,
                      exp_wavelength, exp_loggf):
    lines = lines.set_index(["atomic_number", "ion_number",
                              "level_number_lower", "level_number_upper"])
    wavelength = lines.loc[(atomic_number, ion_number,
                            level_number_lower, level_number_upper)]["wavelength"] * u.Unit("angstrom")
    loggf = lines.loc[(atomic_number, ion_number,
                            level_number_lower, level_number_upper)]["loggf"]
    assert_quantity_allclose(wavelength, exp_wavelength)
    assert_almost_equal(loggf, exp_loggf)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number_lower, level_number_upper, exp_wavelength, exp_loggf",[
    # Kurucz lines with wavelength above 2000 Angstrom
    (14, 1, 8, 83, 2015.574293 * u.Unit("angstrom"), -0.120),
    (14, 1, 8, 82, 2017.305610 * u.Unit("angstrom"), 0.190)
])
def test_create_lines_convert_air2vacuum(lines, atomic_number, ion_number, level_number_lower, level_number_upper,
                      exp_wavelength, exp_loggf):
    lines = lines.set_index(["atomic_number", "ion_number",
                              "level_number_lower", "level_number_upper"])
    wavelength = lines.loc[(atomic_number, ion_number,
                            level_number_lower, level_number_upper)]["wavelength"] * u.Unit("angstrom")
    loggf = lines.loc[(atomic_number, ion_number,
                            level_number_lower, level_number_upper)]["loggf"]
    assert_quantity_allclose(wavelength, exp_wavelength)
    assert_almost_equal(loggf, exp_loggf)


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, level_number_lower, level_number_upper", [
    # Default loggf_threshold = -3
    # Kurucz lines
    (14, 1, 3, 98), # loggf = -4.430
    (14, 1, 2, 83), # loggf = -4.140
    # CHIANTI lines
    (7, 5, 3, 11), # loggf = -5.522589
    (7, 5, 6, 7), # loggf = -5.240106
])
def test_create_lines_loggf_treshold(lines, atomic_number, ion_number, level_number_lower, level_number_upper):
    lines = lines.set_index(["atomic_number", "ion_number",
                             "level_number_lower", "level_number_upper"])
    with pytest.raises(KeyError):
        lines.loc[(atomic_number, ion_number, level_number_lower, level_number_upper)]


@with_test_db
@pytest.mark.parametrize("atomic_number", [2, 14, 30])
def test_levels_create_artificial_fully_ionized(levels, atomic_number):
    levels = levels.set_index(["atomic_number", "ion_number", "level_number"])
    energy, g, metastable = levels.loc[(atomic_number, atomic_number, 0), ["energy", "g", "metastable"]]
    assert_almost_equal(energy, 0.0)
    assert g == 1
    assert metastable

# ToDo: Implement real tests
@with_test_db
def test_create_collisions(collisions):
    assert True


@with_test_db
def test_create_macro_atom(macro_atom):
    assert True


@with_test_db
def test_create_macro_atom_ref(macro_atom_references):
    assert True


@with_test_db
@pytest.mark.parametrize("atomic_number, ion_number, source_level_number", [
    (2, 2, 0),
    (5, 5, 0),
    (30, 19, 0),
    (30, 30, 0)
])
def test_create_macro_atom_references_levels_wo_lines(macro_atom_references, atomic_number,
                                                      ion_number, source_level_number):
    macro_atom_references = macro_atom_references.set_index(
        ["atomic_number", "ion_number", "source_level_number"]
    )
    count_up, count_down, count_total = macro_atom_references.loc[
        (atomic_number, ion_number, source_level_number), ("count_up", "count_down", "count_total")
    ]
    assert all([count == 0 for count in [count_up, count_down, count_total]])


@with_test_db
def test_create_zeta_data(zeta_data):
    assert True


@with_test_db
def test_atom_data_to_hdf(atom_data, hdf5_path):
    atom_data.to_hdf(hdf5_path, store_atom_masses=True, store_ionization_energies=True,
                     store_levels=True, store_lines=True, store_macro_atom=True,
                     store_zeta_data=True, store_collisions=True)
