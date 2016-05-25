import pytest
from ..chianti_ import ChiantiReader, ChiantiIngester, masterlist_ions
from carsus.model import Level, LevelEnergy, Ion
from numpy.testing import assert_almost_equal


@pytest.fixture
def chianti_reader():
    ions = ['ne_2', 'cl_4', 'ne_6']
    return ChiantiReader(ions)

@pytest.fixture
def chianti_ingester(test_session):
    ions_list = masterlist_ions[:3]  # ['ne_2', 'cl_4', 'ne_6']
    ingester = ChiantiIngester(test_session, ions_list=ions_list)
    return ingester

def test_chianti_reader_init(chianti_reader):
    assert len(chianti_reader.ions) == 3


def test_chianti_reader_init_w_bad_ions():
    chianti_rdr = ChiantiReader(['ne_2', 'cl_4', 'ne_6', 'au_1'])
    assert len(chianti_rdr.ions) == 3


def test_chianti_reader_read_levels(chianti_reader):
    levels_df = chianti_reader.read_levels()
    assert_almost_equal(levels_df.loc[10,1,1]['energy'], 0)


@pytest.mark.parametrize("atomic_number, ion_charge, count",[
    (10, 1, 138),
    (10, 5, 204),
    (17, 3, 5)
])
def test_chianti_ingest_levels_count(test_session, chianti_ingester, atomic_number, ion_charge, count):
    chianti_ingester.ingest()
    test_session.commit()
    ne_1 = Ion.as_unique(test_session, atomic_number=atomic_number, ion_charge=ion_charge)
    assert len(ne_1.levels) == count