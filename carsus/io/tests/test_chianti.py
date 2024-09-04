import pytest

from carsus.io.chianti_ import ChiantiIonReader, ChiantiReader


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
