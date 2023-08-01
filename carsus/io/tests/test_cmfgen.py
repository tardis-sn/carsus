import os
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

from carsus.conftest import withrefdata
from carsus.io.cmfgen.util import *

with_refdata = pytest.mark.skipif(
    not withrefdata, reason="--refdata folder not specified"
)
data_dir = os.path.join(os.path.dirname(__file__), "data")


@with_refdata
@pytest.fixture()
def si1_reader():
    print(with_refdata)
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
def cmfgen_refdata_fname(refdata_path, path):
    subdirectory, fname = path
    return os.path.join(refdata_path, "cmfgen", subdirectory, fname)


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
@pytest.mark.parametrize(
    "path",
    [
        ["energy_levels", "si2_osc_kurucz"],
    ],
)
def test_CMFGENEnergyLevelsParser(cmfgen_refdata_fname):
    parser = CMFGENEnergyLevelsParser(cmfgen_refdata_fname)
    n = int(parser.header["Number of energy levels"])
    assert parser.base.shape[0] == n
    return parser.base


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
@pytest.mark.parametrize(
    "path",
    [
        ["oscillator_strengths", "fevi_osc_kb_rk.dat"],
        ["oscillator_strengths", "p2_osc"],
        ["oscillator_strengths", "vi_osc"],
    ],
)
def test_CMFGENOscillatorStrengthsParser(cmfgen_refdata_fname):
    parser = CMFGENOscillatorStrengthsParser(cmfgen_refdata_fname)
    n = int(parser.header["Number of transitions"])
    assert parser.base.shape[0] == n
    return parser.base


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
@pytest.mark.parametrize(
    "path",
    [
        ["collisional_strengths", "he2col.dat"],
        ["collisional_strengths", "col_ariii"],
    ],
)
def test_CMFGENCollisionalStrengthsParser(cmfgen_refdata_fname):
    parser = CMFGENCollisionalStrengthsParser(cmfgen_refdata_fname)
    return parser.base


@with_refdata
@pytest.mark.parametrize(
    "path",
    [
        ["photoionization_cross_sections", "phot_nahar_A"],
        ["photoionization_cross_sections", "phot_data_gs"],
    ],
)
@pytest.mark.array_compare(file_format="pd_hdf")
def test_CMFGENPhoCrossSectionsParser(cmfgen_refdata_fname):
    parser = CMFGENPhoCrossSectionsParser(cmfgen_refdata_fname)
    n = int(parser.header["Number of energy levels"])
    assert len(parser.base) == n
    return parser.base[0]


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
@pytest.mark.parametrize(
    "path",
    [
        ["photoionization_cross_sections", "hyd_l_data.dat"],
    ],
)
def test_CMFGENHydLParser(cmfgen_refdata_fname):
    parser = CMFGENHydLParser(cmfgen_refdata_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    return parser.base


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
@pytest.mark.parametrize(
    "path",
    [
        ["photoionization_cross_sections", "gbf_n_data.dat"],
    ],
)
def test_CMFGENHydGauntBfParser(cmfgen_refdata_fname):
    parser = CMFGENHydGauntBfParser(cmfgen_refdata_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    return parser.base


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
def test_reader_lines(si1_reader):
    return si1_reader.lines


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
def test_reader_levels(si1_reader):
    return si1_reader.levels


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
def test_reader_collisions(si1_reader):
    return si1_reader.collisions


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
def test_reader_cross_sections_squeeze(si1_reader):
    return si1_reader.cross_sections


@with_refdata
@pytest.mark.array_compare(file_format="pd_hdf")
def test_reader_ionization_energies(si1_reader):
    return si1_reader.ionization_energies.to_frame()


@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [0.053130732819562695])
@pytest.mark.parametrize("fit_coeff_list", [[34.4452, 1.0, 2.0]])
def test_get_seaton_phixs_table(threshold_energy_ryd, fit_coeff_list):
    phixs_table = get_seaton_phixs_table(threshold_energy_ryd, *fit_coeff_list)
    return phixs_table


@pytest.mark.array_compare
@pytest.mark.parametrize("hyd_gaunt_energy_grid_ryd", [{1: list(range(1, 3))}])
@pytest.mark.parametrize("hyd_gaunt_factor", [{1: list(range(3, 6))}])
@pytest.mark.parametrize("threshold_energy_ryd", [0.5])
@pytest.mark.parametrize("n", [1])
def test_get_hydrogenic_n_phixs_table(
    hyd_gaunt_energy_grid_ryd, hyd_gaunt_factor, threshold_energy_ryd, n
):
    hydrogenic_n_phixs_table = get_hydrogenic_n_phixs_table(
        hyd_gaunt_energy_grid_ryd, hyd_gaunt_factor, threshold_energy_ryd, n
    )
    return hydrogenic_n_phixs_table


@pytest.mark.array_compare
@pytest.mark.parametrize("hyd_phixs_energy_grid_ryd", [{(4, 1): np.linspace(1, 3, 5)}])
@pytest.mark.parametrize("hyd_phixs", [{(4, 1): np.linspace(1, 3, 5)}])
@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("n", [4])
@pytest.mark.parametrize("l_start", [1])
@pytest.mark.parametrize("l_end", [1])
@pytest.mark.parametrize("nu_0", [0.2])
def test_get_hydrogenic_nl_phixs_table(
    hyd_phixs_energy_grid_ryd, hyd_phixs, threshold_energy_ryd, n, l_start, l_end, nu_0
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
    return phixs_table


@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("vars", [[3, 4, 5, 6, 7]])
@pytest.mark.parametrize("n_points", [50])
def test_get_opproject_phixs_table(threshold_energy_ryd, vars, n_points):
    phixs_table = get_opproject_phixs_table(threshold_energy_ryd, *vars, n_points)
    return phixs_table


@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("vars", [[2, 3, 4, 5, 6, 7, 8, 9]])
@pytest.mark.parametrize("n_points", [50])
def test_get_hummer_phixs_table(threshold_energy_ryd, vars, n_points):
    phixs_table = get_hummer_phixs_table(threshold_energy_ryd, *vars, n_points)
    return phixs_table


@pytest.mark.array_compare
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
def test_get_vy95_phixs_table(threshold_energy_ryd, fit_coeff_table, n_points):
    phixs_table = get_vy95_phixs_table(threshold_energy_ryd, fit_coeff_table, n_points)
    return phixs_table


@pytest.mark.skip(reason="Not implemented yet")
def test_get_leibowitz_phixs_table():
    pass


@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [50])
def test_get_null_phixs_table(threshold_energy_ryd):
    phixs_table = get_null_phixs_table(threshold_energy_ryd)
    return phixs_table
