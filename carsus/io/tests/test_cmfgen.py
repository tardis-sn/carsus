from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from carsus.io.cmfgen import (
    CMFGENEnergyLevelsParser,
    CMFGENOscillatorStrengthsParser,
    CMFGENCollisionalStrengthsParser,
    CMFGENPhoCrossSectionsParser,
    CMFGENHydLParser,
    CMFGENHydGauntBfParser,
    CMFGENReader,
)

from carsus.io.cmfgen.util import *

data_dir = Path(__file__).parent / "data"


@pytest.fixture()
def si1_reader():
    return CMFGENReader.from_config(
        "Si 0-1",
        atomic_path="/tmp/atomic",
        collisions=True,
        cross_sections=True,
        ionization_energies=True,
        temperature_grid=np.arange(2000, 50000, 2000),
        drop_mismatched_labels=True,
    )


@pytest.fixture()
def cmfgen_regression_data_fname(carsus_regression_path, path):
    subdirectory, fname = path
    return Path(carsus_regression_path) / "cmfgen" / subdirectory / fname


@pytest.mark.with_regression_data
@pytest.mark.parametrize(
    "path",
    [
        ["energy_levels", "si2_osc_kurucz"],
    ],
)
def test_CMFGENEnergyLevelsParser(cmfgen_regression_data_fname, regression_data):
    cmfgen_regression_data_fname = str(cmfgen_regression_data_fname)
    parser = CMFGENEnergyLevelsParser(cmfgen_regression_data_fname)
    n = int(parser.header["Number of energy levels"])
    assert parser.base.shape[0] == n
    
    expected = regression_data.sync_dataframe(parser.base)
    pd.testing.assert_frame_equal(parser.base, expected)


@pytest.mark.with_regression_data
@pytest.mark.parametrize(
    "path",
    [
        ["oscillator_strengths", "fevi_osc_kb_rk.dat"],
        ["oscillator_strengths", "p2_osc"],
        ["oscillator_strengths", "vi_osc"],
    ],
)
def test_CMFGENOscillatorStrengthsParser(cmfgen_regression_data_fname, regression_data):
    cmfgen_regression_data_fname = str(cmfgen_regression_data_fname)
    parser = CMFGENOscillatorStrengthsParser(cmfgen_regression_data_fname)
    n = int(parser.header["Number of transitions"])
    assert parser.base.shape[0] == n
    
    expected = regression_data.sync_dataframe(parser.base)
    pd.testing.assert_frame_equal(parser.base, expected)


@pytest.mark.with_regression_data
@pytest.mark.parametrize(
    "path",
    [
        ["collisional_strengths", "he2col.dat"],
        ["collisional_strengths", "col_ariii"],
    ],
)
def test_CMFGENCollisionalStrengthsParser(cmfgen_regression_data_fname, regression_data):
    cmfgen_regression_data_fname = str(cmfgen_regression_data_fname)
    parser = CMFGENCollisionalStrengthsParser(cmfgen_regression_data_fname)
    
    expected = regression_data.sync_dataframe(parser.base)
    pd.testing.assert_frame_equal(parser.base, expected)


@pytest.mark.with_regression_data
@pytest.mark.parametrize(
    "path",
    [
        ["photoionization_cross_sections", "phot_nahar_A"],
        ["photoionization_cross_sections", "phot_data_gs"],
    ],
)
def test_CMFGENPhoCrossSectionsParser(cmfgen_regression_data_fname, regression_data):
    cmfgen_regression_data_fname = str(cmfgen_regression_data_fname)
    parser = CMFGENPhoCrossSectionsParser(cmfgen_regression_data_fname)
    n = int(parser.header["Number of energy levels"])
    assert len(parser.base) == n
    
    # Test the first item in the base collection
    first_item = parser.base[0]
    expected = regression_data.sync_dataframe(first_item)
    pd.testing.assert_frame_equal(first_item, expected)


@pytest.mark.with_regression_data
@pytest.mark.parametrize(
    "path",
    [
        ["photoionization_cross_sections", "hyd_l_data.dat"],
    ],
)
def test_CMFGENHydLParser(cmfgen_regression_data_fname, regression_data):
    cmfgen_regression_data_fname = str(cmfgen_regression_data_fname)
    parser = CMFGENHydLParser(cmfgen_regression_data_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    
    expected = regression_data.sync_dataframe(parser.base)
    pd.testing.assert_frame_equal(parser.base, expected)


@pytest.mark.with_regression_data
@pytest.mark.parametrize(
    "path",
    [
        ["photoionization_cross_sections", "gbf_n_data.dat"],
    ],
)
def test_CMFGENHydGauntBfParser(cmfgen_regression_data_fname, regression_data):
    cmfgen_regression_data_fname = str(cmfgen_regression_data_fname)
    parser = CMFGENHydGauntBfParser(cmfgen_regression_data_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    
    expected = regression_data.sync_dataframe(parser.base)
    pd.testing.assert_frame_equal(parser.base, expected)


@pytest.mark.with_regression_data
def test_reader_lines(si1_reader, regression_data):
    lines = si1_reader.lines
    expected = regression_data.sync_dataframe(lines)
    pd.testing.assert_frame_equal(lines, expected)


@pytest.mark.with_regression_data
def test_reader_levels(si1_reader, regression_data):
    levels = si1_reader.levels
    expected = regression_data.sync_dataframe(levels)
    pd.testing.assert_frame_equal(levels, expected)


@pytest.mark.with_regression_data
def test_reader_collisions(si1_reader, regression_data):
    collisions = si1_reader.collisions
    expected = regression_data.sync_dataframe(collisions)
    pd.testing.assert_frame_equal(collisions, expected)


@pytest.mark.with_regression_data
def test_reader_cross_sections_squeeze(si1_reader, regression_data):
    cross_sections = si1_reader.cross_sections
    expected = regression_data.sync_dataframe(cross_sections)
    pd.testing.assert_frame_equal(cross_sections, expected)


@pytest.mark.with_regression_data
def test_reader_ionization_energies(si1_reader, regression_data):
    ionization_energies = si1_reader.ionization_energies.to_frame()
    expected = regression_data.sync_dataframe(ionization_energies)
    pd.testing.assert_frame_equal(ionization_energies, expected)


@pytest.mark.parametrize("threshold_energy_ryd", [0.053130732819562695])
@pytest.mark.parametrize("fit_coeff_list", [[34.4452, 1.0, 2.0]])
def test_get_seaton_phixs_table(threshold_energy_ryd, fit_coeff_list, regression_data):
    phixs_table = get_seaton_phixs_table(threshold_energy_ryd, *fit_coeff_list)
    expected = regression_data.sync_ndarray(phixs_table)
    np.testing.assert_allclose(phixs_table, expected)


@pytest.mark.parametrize("hyd_gaunt_energy_grid_ryd", [{1: list(range(1, 4))}])
@pytest.mark.parametrize("hyd_gaunt_factor", [{1: list(range(3, 6))}])
@pytest.mark.parametrize("threshold_energy_ryd", [0.5])
@pytest.mark.parametrize("n", [1])
@pytest.mark.parametrize("hyd_n_phixs_stop2start_energy_ratio", [20])
@pytest.mark.parametrize("hyd_n_phixs_num_points", [20])
def test_get_hydrogenic_n_phixs_table(
    hyd_gaunt_energy_grid_ryd,
    hyd_gaunt_factor,
    threshold_energy_ryd,
    n,
    hyd_n_phixs_stop2start_energy_ratio,
    hyd_n_phixs_num_points,
    regression_data
):
    hydrogenic_n_phixs_table = get_hydrogenic_n_phixs_table(
        hyd_gaunt_energy_grid_ryd,
        hyd_gaunt_factor,
        threshold_energy_ryd,
        n,
        hyd_n_phixs_stop2start_energy_ratio,
        hyd_n_phixs_num_points,
    )
    expected = regression_data.sync_ndarray(hydrogenic_n_phixs_table)
    np.testing.assert_allclose(hydrogenic_n_phixs_table, expected)


@pytest.mark.parametrize("hyd_phixs_energy_grid_ryd", [{(4, 1): np.linspace(1, 3, 5)}])
@pytest.mark.parametrize("hyd_phixs", [{(4, 1): np.linspace(1, 3, 5)}])
@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("l_start", [1])
@pytest.mark.parametrize("l_end", [1])
@pytest.mark.parametrize("nu_0", [0.2])
def test_get_hydrogenic_nl_phixs_table(
    hyd_phixs_energy_grid_ryd, 
    hyd_phixs, 
    threshold_energy_ryd, 
    n, 
    l_start, 
    l_end, 
    nu_0,
    regression_data
):
    phixs_table = get_hydrogenic_nl_phixs_table(
        hyd_phixs_energy_grid_ryd,
        hyd_phixs,
        threshold_energy_ryd,
        n,
        l_start,
        l_end,
        nu_0,
    )
    expected = regression_data.sync_ndarray(phixs_table)
    np.testing.assert_allclose(phixs_table, expected)


@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("vars", [[3, 4, 5, 6, 7]])
@pytest.mark.parametrize("n_points", [50])
def test_get_opproject_phixs_table(threshold_energy_ryd, vars, n_points, regression_data):
    phixs_table = get_opproject_phixs_table(threshold_energy_ryd, *vars, n_points)
    expected = regression_data.sync_ndarray(phixs_table)
    np.testing.assert_allclose(phixs_table, expected)


@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("vars", [[2, 3, 4, 5, 6, 7, 8, 9]])
@pytest.mark.parametrize("n_points", [50])
def test_get_hummer_phixs_table(threshold_energy_ryd, vars, n_points, regression_data):
    phixs_table = get_hummer_phixs_table(threshold_energy_ryd, *vars, n_points)
    expected = regression_data.sync_ndarray(phixs_table)
    np.testing.assert_allclose(phixs_table, expected)


@pytest.mark.parametrize("threshold_energy_ryd", [10])
@pytest.mark.parametrize(
    "fit_coeff_table",
    [
        pd.DataFrame.from_dict(
            {
                "E": [1, 2],
                "E_0": [1, 2],
                "P": [2, 2],
                "l": [2, 2],
                "sigma_0": [1, 2],
                "y(a)": [1, 3],
                "y(w)": [1, 4],
            }
        )
    ],
)
@pytest.mark.parametrize("n_points", [50])
def test_get_vy95_phixs_table(threshold_energy_ryd, fit_coeff_table, n_points, regression_data):
    phixs_table = get_vy95_phixs_table(threshold_energy_ryd, fit_coeff_table, n_points)
    expected = regression_data.sync_ndarray(phixs_table)
    np.testing.assert_allclose(phixs_table, expected)


@pytest.mark.skip(reason="Not implemented yet")
def test_get_leibowitz_phixs_table():
    pass


@pytest.mark.parametrize("threshold_energy_ryd", [50])
def test_get_null_phixs_table(threshold_energy_ryd, regression_data):
    phixs_table = get_null_phixs_table(threshold_energy_ryd)
    expected = regression_data.sync_ndarray(phixs_table)
    np.testing.assert_allclose(phixs_table, expected)
