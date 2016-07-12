import pytest
from carsus.model import Atom, AtomWeight, DataSource,\
    Ion, IonizationEnergy, Level, LevelEnergy, Line, ECollision
from astropy import units as u
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal
from astropy.tests.helper import assert_quantity_allclose


@pytest.mark.parametrize("atomic_number, expected_symbol",[
    (1, "H"),
    (10, "Ne")
])
def test_atom_query(foo_session, atomic_number, expected_symbol):
    atom = foo_session.query(Atom).get(atomic_number)
    assert atom.symbol == expected_symbol


def test_data_source_as_unique(foo_session):
    nist = DataSource.as_unique(foo_session, short_name="nist",
                                name="National Institute of Standards and Technology")
    nist2 = DataSource.as_unique(foo_session, short_name="nist",
                                 name="National Institute of Standards and Technology")
    assert nist is nist2


def test_data_source_unique_constraint(foo_session):
    nist_duplicate = DataSource(short_name="nist")
    foo_session.add(nist_duplicate)
    with pytest.raises(IntegrityError):
        foo_session.commit()


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight",[
    (1, "ku", 1.00811*u.u),
    (1, "nist", 1.00784*u.u)
])
def test_atomic_weights_query(foo_session, atomic_number, ds_short_name, expected_weight):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom, atom_weight = foo_session.query(Atom, AtomWeight).\
        filter(Atom.atomic_number == atomic_number).\
        join(Atom.weights).\
        filter(AtomWeight.data_source == data_source).one()
    assert_quantity_allclose([atom_weight.quantity], expected_weight)


@pytest.mark.parametrize("atomic_number, ds_short_name, expected_weight_value",[
    (1, "ku", 8.41364096027349e-58),
    (1, "nist", 8.415894971881755e-58)
])
def test_atomic_weights_convert_to(foo_session, atomic_number, ds_short_name, expected_weight_value):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    atom, atom_weight_value = foo_session.query(Atom, AtomWeight.quantity.to(u.solMass).value). \
        filter(Atom.atomic_number == atomic_number). \
        join(Atom.weights). \
        filter(AtomWeight.data_source == data_source).one()
    assert_almost_equal(atom_weight_value, expected_weight_value)


@pytest.mark.parametrize("atomic_number, ion_charge, symbol",[
    (1, 0, "H"),
    (10, 1, "Ne")
])
def test_ion_query(foo_session, atomic_number, ion_charge, symbol):
    ion = foo_session.query(Ion).get((atomic_number, ion_charge))
    assert ion.atom.symbol == symbol


@pytest.mark.parametrize("atomic_number, ion_charge",[
    (1, 0)
])
def test_ion_as_unique(foo_session, atomic_number, ion_charge):
    ion = foo_session.query(Ion).get((atomic_number, ion_charge))
    ion2 = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    assert ion is ion2


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, "
                         "method, expected_ionization_energy",[
    (1, 0, "nist", "th", 13.5984*u.eV),
    (10, 1, "nist", "th", 40.96296*u.eV),
    (10, 1, "nist", "m", 40.97*u.eV),
])
def test_ionization_energies_query(foo_session, atomic_number, ion_charge, ds_short_name,
                                   method, expected_ionization_energy):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion, ionization_energy = foo_session.query(Ion, IonizationEnergy).\
        filter(and_(Ion.atomic_number == atomic_number,
                    Ion.ion_charge == ion_charge)).\
        join(Ion.ionization_energies).\
        filter(and_(IonizationEnergy.data_source == data_source,
                    IonizationEnergy.method == method)).one()
    assert_quantity_allclose([ionization_energy.quantity], expected_ionization_energy)



@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, level_index,"
                         "configuration, term, L, J, spin_multiplicity, parity",[
    (10, 2, "ku", 1, "2s2.2p5", "2P1.5", "P", 1.5, 2, 1),
    (10, 2, "ku", 2, "2s2.2p5", "2P0.5", "P", 0.5, 2, 1),
    (10, 2, "chianti", 2, "2s2.2p5", "2P0.5", "P", 0.5, 2, 1),
])
def test_level_query(foo_session, atomic_number, ion_charge, ds_short_name, level_index,
                         configuration, term, L, J, spin_multiplicity, parity):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    level = foo_session.query(Level).\
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == level_index)).one()
    assert level.configuration == configuration
    assert level.term == term
    assert level.L == L
    assert_almost_equal(level.J, J)
    assert level.spin_multiplicity == spin_multiplicity
    assert level.parity == parity


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, level_index,"
                         "method, expected_level_energy_value", [
    (10, 2, "ku", 1, "m", 0),
    (10, 2, "ku", 1, "th", 0),
    (10, 2, "ku", 2, "m", 780.4),
    (10, 2, "chianti", 2, "m", 780.2)  # Unit cm-1
])
def test_ionization_energies_query(foo_session, atomic_number, ion_charge,  ds_short_name,
                                   level_index, method, expected_level_energy_value):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    level, level_energy_value = foo_session.query(Level,
                                        LevelEnergy.quantity.to(u.Unit("cm-1"), equivalencies=u.spectral()).value). \
                                filter(and_(Level.ion == ion,
                                            Level.data_source == data_source),
                                            Level.level_index == level_index). \
                                join(Level.energies). \
                                filter(and_(LevelEnergy.data_source == data_source,
                                            LevelEnergy.method == method)).one()
    assert_almost_equal(level_energy_value, expected_level_energy_value)


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, lower_level_index, upper_level_index,"
                         "expected_wavelength, expected_a_value, expected_gf_value", [
                        (10, 2, "ku", 1, 2, 155545.188*u.AA, 5.971e-03*u.Unit("s-1"), 8.792e-01),
])
def test_line_quantities_query(foo_session, atomic_number, ion_charge, ds_short_name,
                               lower_level_index, upper_level_index,
                               expected_wavelength, expected_a_value, expected_gf_value):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    lower_level = foo_session.query(Level).\
        filter(and_(Level.data_source==data_source,
                    Level.ion == ion,
                    Level.level_index == lower_level_index)).one()
    upper_level = foo_session.query(Level). \
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == upper_level_index)).one()
    line = foo_session.query(Line).\
        filter(and_(Line.data_source == data_source,
                    Line.lower_level == lower_level,
                    Line.upper_level == upper_level)).one()

    wavelength = line.wavelengths[0].quantity
    a_value = line.a_values[0].quantity
    gf_value = line.gf_values[0].quantity

    assert_quantity_allclose(wavelength, expected_wavelength)
    assert_quantity_allclose(a_value, expected_a_value)
    assert_quantity_allclose(gf_value, expected_gf_value)


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, level_index, exp_g", [
    (10, 2, "ku", 1, 4),
    (10, 2, "ku", 2, 2),
])
def test_level_g_hybrid_attribute(foo_session, atomic_number, ion_charge, ds_short_name, level_index, exp_g):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    level = foo_session.query(Level). \
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == level_index)).one()
    assert level.g == exp_g


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, level_index, exp_g", [
    (10, 2, "ku", 1, 4),
    (10, 2, "ku", 2, 2),
])
def test_level_g_hybrid_attribute_expression(foo_session, atomic_number, ion_charge, ds_short_name, level_index, exp_g):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    g = foo_session.query(Level.g). \
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == level_index)).scalar()
    assert g == exp_g


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, lower_level_index, upper_level_index,"
                         "expected_energy, expected_temp_strengths", [
                             (10, 2, "ku", 1, 2, 0.007108*u.rydberg, [(0.0, 0.255), (0.07394, 0.266)]),
                         ])
def test_e_collision_quantities_query(foo_session, atomic_number, ion_charge, ds_short_name,
                               lower_level_index, upper_level_index,
                               expected_energy, expected_temp_strengths):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    lower_level = foo_session.query(Level). \
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == lower_level_index)).one()
    upper_level = foo_session.query(Level). \
        filter(and_(Level.data_source == data_source,
                    Level.ion == ion,
                    Level.level_index == upper_level_index)).one()
    e_col = foo_session.query(ECollision). \
        filter(and_(ECollision.data_source == data_source,
                    ECollision.lower_level == lower_level,
                    ECollision.upper_level == upper_level)).one()

    energy = e_col.energies[0].quantity

    assert_quantity_allclose(energy, expected_energy)

    for temp_strength, expected_temp_strength in zip(e_col.temp_strengths_tuple, expected_temp_strengths):
        assert_allclose(temp_strength, expected_temp_strength)


@pytest.mark.parametrize("atomic_number, ion_charge, ds_short_name, expected_energy, expected_J", [
    (10, 2, "nist", 0.0, 1.5),
])
def test_ion_ground_levels(foo_session, atomic_number, ion_charge, ds_short_name, expected_energy, expected_J):
    data_source = DataSource.as_unique(foo_session, short_name=ds_short_name)
    ion = Ion.as_unique(foo_session, atomic_number=atomic_number, ion_charge=ion_charge)
    ground_lvl = ion.ground_levels[0]
    assert ground_lvl.data_source_id == data_source.data_source_id
    assert_almost_equal(ground_lvl.energies[0].quantity.value, expected_energy)
    assert_almost_equal(ground_lvl.J, expected_J)