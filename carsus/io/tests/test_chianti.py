import pytest
from ..chianti_ import ChiantiReader, ChiantiIngester, masterlist_ions
from carsus.model import Level, LevelEnergy, Ion
from numpy.testing import assert_almost_equal


@pytest.fixture(scope="module")
def chianti_reader():
    ions = ['ne_2', 'cl_4', 'ne_6']
    return ChiantiReader(ions)


@pytest.fixture(scope="module")
def levels_df(chianti_reader):
    return chianti_reader.read_levels()


@pytest.fixture(scope="module")
def lines_df(chianti_reader):
    return chianti_reader.read_lines()


@pytest.fixture
def chianti_ingester(test_session):
    ions_list = ['ne_2', 'cl_4', 'ne_6']
    ingester = ChiantiIngester(test_session, ions_list=ions_list)
    return ingester


def test_chianti_reader_init(chianti_reader):
    assert len(chianti_reader.ions) == 3


def test_chianti_reader_init_w_bad_ions():
    chianti_rdr = ChiantiReader(['ne_2', 'cl_4', 'ne_6', 'au_1'])
    assert len(chianti_rdr.ions) == 3


@pytest.mark.parametrize("index, energy, energy_theoretical",[
    ((10, 5, 1), 0, 0),
    ((10, 5, 20), 816294.000, 815133.312),
])
def test_chianti_reader_read_levels(levels_df, index, energy, energy_theoretical):
    row = levels_df.loc[index]
    assert_almost_equal(row['energy'], energy)
    assert_almost_equal(row['energy_theoretical'], energy_theoretical)


@pytest.mark.parametrize("index, wavelength, method",[
    ((10, 5, 10, 13), 913.195, "m"),
    ((10, 5, 74, 204), 4.091, "th"),
])
def test_chianti_reader_read_lines(lines_df, index, wavelength, method):
    row = lines_df.loc[index]
    assert_almost_equal(row['wavelength'], wavelength)
    assert row['method'] == method


@pytest.mark.parametrize("atomic_number, ion_charge, count",[
    (10, 1, 138),
    (10, 5, 204),
    (17, 3, 5)
])
def test_chianti_ingest_levels_count(test_session, chianti_ingester, atomic_number, ion_charge, count):
    chianti_ingester.ingest(levels=True, lines=False)
    test_session.commit()
    ne_1 = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
    assert len(ne_1.levels) == count


@pytest.mark.parametrize("atomic_number, ion_charge, lines_count",[
    (10, 1, 1999)
])
def test_chianti_ingest_lines_count(test_session, chianti_ingester, atomic_number, ion_charge, lines_count):
    chianti_ingester.ingest(levels=True, lines=True)
    ne_1 = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
    assert len(ne_1.lines) == lines_count
