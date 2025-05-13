import pytest
import pandas as pd

from carsus.io.chianti_ import ChiantiIonReader, ChiantiReader


class TestChiantiIonReader:
    @pytest.fixture(scope="class", params=["ne_2", "n_5"])
    def ch_ion_reader(self, request):
        return ChiantiIonReader(request.param)

    def test_chianti_bound_levels(self, ch_ion_reader, regression_data):
        bound_levels = ch_ion_reader.bound_levels
        expected = regression_data.sync_dataframe(bound_levels)
        pd.testing.assert_frame_equal(bound_levels, expected)

    def test_chianti_bound_lines(self, ch_ion_reader, regression_data):
        bound_lines = ch_ion_reader.bound_lines
        expected = regression_data.sync_dataframe(bound_lines)
        pd.testing.assert_frame_equal(bound_lines, expected)

    def test_chianti_reader_read_levels(self, ch_ion_reader, regression_data):
        levels = ch_ion_reader.levels
        expected = regression_data.sync_dataframe(levels)
        pd.testing.assert_frame_equal(levels, expected)

    def test_chianti_reader_read_lines(self, ch_ion_reader, regression_data):
        lines = ch_ion_reader.lines
        expected = regression_data.sync_dataframe(lines)
        pd.testing.assert_frame_equal(lines, expected)

    def test_chianti_reader_read_collisions(self, ch_ion_reader, regression_data):
        collisions = ch_ion_reader.collisions
        expected = regression_data.sync_dataframe(collisions)
        pd.testing.assert_frame_equal(collisions, expected)


class TestChiantiReader:
    @pytest.fixture(scope="class", params=["H-He", "N"])
    def ch_reader(self, request):
        return ChiantiReader(ions=request.param, collisions=True, priority=20)

    def test_levels(self, ch_reader, regression_data):
        levels = ch_reader.levels
        expected = regression_data.sync_dataframe(levels)
        pd.testing.assert_frame_equal(levels, expected)

    def test_lines(self, ch_reader, regression_data):
        lines = ch_reader.lines
        expected = regression_data.sync_dataframe(lines)
        pd.testing.assert_frame_equal(lines, expected)

    def test_cols(self, ch_reader, regression_data):
        collisions = ch_reader.collisions
        expected = regression_data.sync_dataframe(collisions)
        pd.testing.assert_frame_equal(collisions, expected)
