import os
import pytest
import numpy as np
import pandas as pd
from io import StringIO
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from carsus.io.cmfgen import (CMFGENEnergyLevelsParser,
                              CMFGENOscillatorStrengthsParser,
                              CMFGENCollisionalStrengthsParser,
                              CMFGENPhoCrossSectionsParser,
                              CMFGENHydLParser,
                              CMFGENHydGauntBfParser,
                              CMFGENReader
                             )

from carsus.io.cmfgen.util import *

with_refdata = pytest.mark.skipif(
    not pytest.config.getoption("--refdata"),
    reason="--refdata folder not specified"
)
data_dir = os.path.join(os.path.dirname(__file__), 'data')

@with_refdata
@pytest.fixture()
def si2_osc_kurucz_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'energy_levels', 'si2_osc_kurucz')

@with_refdata
@pytest.fixture()
def fevi_osc_kb_rk_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'oscillator_strengths', 'fevi_osc_kb_rk.dat')

@with_refdata
@pytest.fixture()
def p2_osc_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'oscillator_strengths', 'p2_osc')

@with_refdata
@pytest.fixture()
def vi_osc_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'oscillator_strengths', 'vi_osc')

@with_refdata
@pytest.fixture()
def he2_col_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'collisional_strengths', 'he2col.dat')

@with_refdata
@pytest.fixture()
def ariii_col_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'collisional_strengths', 'col_ariii')

@with_refdata
@pytest.fixture()
def si2_col_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'collisional_strengths', 'si2_col')

@with_refdata
@pytest.fixture()
def si2_pho_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'photoionization_cross_sections', 'phot_nahar_A')

@with_refdata
@pytest.fixture()
def coiv_pho_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'photoionization_cross_sections', 'phot_data_gs')


@with_refdata
@pytest.fixture()
def hyd_l_fname(refdata_path):
    return os.path.join(
        refdata_path,
        "cmfgen",
        "photoionization_cross_sections",
        "hyd_l_data.dat",
    )


@with_refdata
@pytest.fixture()
def gbf_n_fname(refdata_path):
    return os.path.join(
        refdata_path,
        "cmfgen",
        "photoionization_cross_sections",
        "gbf_n_data.dat",
    )

@with_refdata
@pytest.fixture()
def si1_data_dict(si2_osc_kurucz_fname, si2_col_fname):
    si1_levels = CMFGENEnergyLevelsParser(si2_osc_kurucz_fname).base  #  (carsus) Si 1 == Si II
    si1_lines = CMFGENOscillatorStrengthsParser(si2_osc_kurucz_fname).base
    si1_col = CMFGENCollisionalStrengthsParser(si2_col_fname).base
    return {(14,1): dict(levels = si1_levels, lines = si1_lines, collisions = si1_col)}

@with_refdata
@pytest.fixture()
def si1_reader(si1_data_dict):
    return CMFGENReader(si1_data_dict, collisions=True)


@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_si2_osc_kurucz(si2_osc_kurucz_fname):
    parser = CMFGENEnergyLevelsParser(si2_osc_kurucz_fname)
    n = int(parser.header['Number of energy levels'])
    assert parser.base.shape[0] == n
    return parser.base

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_fevi_osc_kb_rk(fevi_osc_kb_rk_fname):
    parser = CMFGENOscillatorStrengthsParser(fevi_osc_kb_rk_fname)
    n = int(parser.header['Number of transitions'])
    assert parser.base.shape[0] == n
    return parser.base

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_p2_osc(p2_osc_fname):
    parser = CMFGENOscillatorStrengthsParser(p2_osc_fname)
    n = int(parser.header['Number of transitions'])
    assert parser.base.shape[0] == n
    return parser.base


@with_refdata
def test_vi_osc(vi_osc_fname):
    parser = CMFGENOscillatorStrengthsParser(vi_osc_fname)
    assert parser.base.empty

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_he2_col(he2_col_fname):
    parser = CMFGENCollisionalStrengthsParser(he2_col_fname)
    return parser.base

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_ariii_col(ariii_col_fname):
    parser = CMFGENCollisionalStrengthsParser(ariii_col_fname)
    n = int(parser.header['Number of transitions'])
    assert parser.base.shape == (n, 13)
    return parser.base

@with_refdata
def test_si2_pho(si2_pho_fname):
    parser = CMFGENPhoCrossSectionsParser(si2_pho_fname)
    n = int(parser.header['Number of energy levels'])
    m = int(parser.base[0].attrs['Number of cross-section points'])
    assert len(parser.base) == n
    assert parser.base[0].shape == (m, 2)

@with_refdata
def test_coiv_pho(coiv_pho_fname):
    parser = CMFGENPhoCrossSectionsParser(coiv_pho_fname)
    n = int(parser.header['Number of energy levels'])
    assert len(parser.base) == n
    assert parser.base[0].shape == (3, 8)


@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_hyd_l(hyd_l_fname):
    parser = CMFGENHydLParser(hyd_l_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    return parser.base

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_gbf_n(gbf_n_fname):
    parser = CMFGENHydGauntBfParser(gbf_n_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    return parser.base

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_reader_lines(si1_reader):
    return si1_reader.lines

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_reader_levels(si1_reader):
    return si1_reader.levels

@with_refdata
@pytest.mark.array_compare(file_format='pd_hdf')
def test_reader_collisions(si1_reader):
    return si1_reader.collisions


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
@pytest.mark.parametrize("a", [2])
@pytest.mark.parametrize("b", [3])
@pytest.mark.parametrize("c", [4])
@pytest.mark.parametrize("d", [5])
@pytest.mark.parametrize("e", [6])
@pytest.mark.parametrize("n_points", [50])
def test_get_opproject_phixs_table(
    threshold_energy_ryd, a, b, c, d, e, n_points
):
    phixs_table = get_opproject_phixs_table(
        threshold_energy_ryd, a, b, c, d, e, n_points
    )
    return phixs_table

@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [2])
@pytest.mark.parametrize("a", [2])
@pytest.mark.parametrize("b", [3])
@pytest.mark.parametrize("c", [4])
@pytest.mark.parametrize("d", [5])
@pytest.mark.parametrize("e", [6])
@pytest.mark.parametrize("f", [7])
@pytest.mark.parametrize("g", [8])
@pytest.mark.parametrize("n_points", [50])
def test_get_hummer_phixs_table(
    threshold_energy_ryd, a, b, c, d, e, f, g, n_points
):
    phixs_table = get_hummer_phixs_table(
        threshold_energy_ryd, a, b, c, d, e, f, g, n_points
    )
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
    phixs_table = get_vy95_phixs_table(
        threshold_energy_ryd, fit_coeff_table, n_points
    )
    return phixs_table


# TODO: should this exist? skip?
# def test_get_leibowitz_phixs_table():
#     pass

@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [50])
def test_get_null_phixs_table(threshold_energy_ryd):
    phixs_table = get_null_phixs_table(threshold_energy_ryd)
    return phixs_table


