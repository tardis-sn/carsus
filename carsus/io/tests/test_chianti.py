import pytest
from ..chianti_ import ChiantiIonReader, ChiantiIngester
from carsus.model import Level, LevelEnergy, Ion, ChiantiLevel,\
    Line, LineWavelength, LineAValue, LineGFValue, \
    ECollision, ECollisionTemp, ECollisionStrength, \
    ECollisionGFValue, ECollisionTempStrength, ECollisionEnergy
from numpy.testing import assert_almost_equal


@pytest.fixture(scope="module")
def ch_ion_reader():
    return ChiantiIonReader("ne_6")


@pytest.fixture
def ch_ingester(test_session):
    ions_list = ['ne_2', 'cl_4', 'ne_6']
    ingester = ChiantiIngester(test_session, ions_list=ions_list)
    return ingester


@pytest.mark.parametrize("level_index, energy, energy_theoretical",[
    (1, 0, 0),
    (20, 816294.000, 815133.312),
])
def test_chianti_reader_read_levels(ch_ion_reader, level_index, energy, energy_theoretical):
    row = ch_ion_reader.levels_df.loc[level_index]
    assert_almost_equal(row['energy'], energy)
    assert_almost_equal(row['energy_theoretical'], energy_theoretical)


@pytest.mark.parametrize("level_index, wavelength, method",[
    ((10, 13), 913.195, "m"),
    ((74, 204), 4.091, "th"),
])
def test_chianti_reader_read_lines(ch_ion_reader, level_index, wavelength, method):
    row = ch_ion_reader.lines_df.loc[level_index]
    assert_almost_equal(row['wavelength'], wavelength)
    assert row['method'] == method


# @pytest.mark.parametrize("atomic_number, ion_charge, levels_count",[
#     (10, 1, 138),
#     (10, 5, 204),
#     (17, 3, 5)
# ])
# def test_chianti_ingest_levels_count(test_session, ch_ingester, atomic_number, ion_charge, levels_count):
#     ch_ingester.ingest(levels=True, lines=False)
#     test_session.commit()
#     ne_1 = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
#     assert len(ne_1.levels) == levels_count
#
#
# @pytest.mark.parametrize("atomic_number, ion_charge, lines_count",[
#     (10, 1, 1999)
# ])
# def test_chianti_ingest_lines_count(test_session, ch_ingester, atomic_number, ion_charge, lines_count):
#     ch_ingester.ingest(levels=True, lines=True)
#     ne_1 = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
#     cnt = test_session.query(Level).filter(Level.ion == ne_1).\
#         join(Level.source_transitions.of_type(Line)).count()
#     assert cnt == lines_count


@pytest.mark.parametrize("atomic_number, ion_charge, e_col_count",[
    (10, 1, 9453)
])
def test_chianti_ingest_e_col_count(test_session, ch_ingester, atomic_number, ion_charge, e_col_count):
    ch_ingester.ingest(levels=True, collisions=True)
    ne_1 = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
    cnt = test_session.query(Level).filter(Level.ion == ne_1).\
        join(Level.source_transitions.of_type(ECollision)).count()
    assert cnt == e_col_count
