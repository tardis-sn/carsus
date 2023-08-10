import pytest

from numpy.testing import assert_almost_equal
from carsus.io.chianti_ import ChiantiIonReader, ChiantiIngester, ChiantiReader
from carsus.model import Level, Ion, Line, ECollision



@pytest.fixture
def ch_ingester(memory_session):
    ions = "ne 1; cl 3"
    ingester = ChiantiIngester(memory_session, ions=ions)
    return ingester


class TestChiantiIonReader:
    @pytest.fixture(scope="class", params=["ne_2", "n_5"])
    def ch_ion_reader(self, request):
        return ChiantiIonReader(request.param)

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_chianti_bound_levels(self, ch_ion_reader):
        bound_levels = ch_ion_reader.bound_levels
        return bound_levels

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_chianti_bound_lines(self, ch_ion_reader):
        bound_lines = ch_ion_reader.bound_lines
        return bound_lines

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_chianti_reader_read_levels(self, ch_ion_reader):
        return ch_ion_reader.levels

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_chianti_reader_read_lines(self, ch_ion_reader):
        return ch_ion_reader.lines

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_chianti_reader_read_collisions(self, ch_ion_reader):
        return ch_ion_reader.collisions


@pytest.mark.slow
@pytest.mark.parametrize(
    "atomic_number, ion_charge, levels_count", [(10, 1, 138), (17, 3, 5)]
)
def test_chianti_ingest_levels_count(
    memory_session, ch_ingester, atomic_number, ion_charge, levels_count
):
    ch_ingester.ingest(levels=True)
    ion = Ion.as_unique(
        memory_session, atomic_number=atomic_number, ion_charge=ion_charge
    )
    assert len(ion.levels) == levels_count


@pytest.mark.slow
@pytest.mark.parametrize("atomic_number, ion_charge, lines_count", [(10, 1, 1999)])
def test_chianti_ingest_lines_count(
    memory_session, ch_ingester, atomic_number, ion_charge, lines_count
):
    ch_ingester.ingest(levels=True, lines=True)
    ion = Ion.as_unique(
        memory_session, atomic_number=atomic_number, ion_charge=ion_charge
    )
    cnt = (
        memory_session.query(Line)
        .join(Line.lower_level)
        .filter(Level.ion == ion)
        .count()
    )
    assert cnt == lines_count


@pytest.mark.slow
@pytest.mark.parametrize("atomic_number, ion_charge, e_col_count", [(10, 1, 9453)])
def test_chianti_ingest_e_col_count(
    memory_session, ch_ingester, atomic_number, ion_charge, e_col_count
):
    ch_ingester.ingest(levels=True, collisions=True)
    ion = Ion.as_unique(
        memory_session, atomic_number=atomic_number, ion_charge=ion_charge
    )
    cnt = (
        memory_session.query(ECollision)
        .join(ECollision.lower_level)
        .filter(Level.ion == ion)
        .count()
    )
    assert cnt == e_col_count


class TestChiantiReader:
    @pytest.fixture(scope="class", params=["H-He", "N"])
    def ch_reader(self, request):
        return ChiantiReader(ions=request.param, collisions=True, priority=20)

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_levels(self, ch_reader):
        return ch_reader.levels

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_lines(self, ch_reader):
        return ch_reader.lines

    @pytest.mark.array_compare(file_format="pd_hdf")
    def test_cols(self, ch_reader):
        return ch_reader.collisions
