import pytest
from ..chiantidb import ChiantiReader, ChiantiIngestor, masterlist_ions
from carsus.model import Level, LevelEnergy, Ion
from numpy.testing import assert_almost_equal


@pytest.fixture
def chianti_rdr():
    return ChiantiReader(["si_2", "si_3"])


def test_chianti_reader_init(chianti_rdr):
    assert len(chianti_rdr.ions) == 2


def test_chianti_reader_init_w_bad_ions():
    chianti_rdr = ChiantiReader(["si_2", "si_3", "au_1"])
    assert len(chianti_rdr.ions) == 2


def test_chianti_reader_read_levels(chianti_rdr):
    levels_df = chianti_rdr.read_levels()
    assert_almost_equal(levels_df.loc[14,3,1]['energy'], 0)


def test_chianti_ingest(test_session):
    ions = masterlist_ions[:3]  # ['ne_2', 'cl_4', 'ne_6']
    rdr = ChiantiReader(ions)
    levels_df = rdr.read_levels()
    ingester = ChiantiIngestor(test_session)
    ingester.ingest_levels(levels_df)