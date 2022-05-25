import os
import glob
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

si2_levels_head = """
0.00      0.5  3s2_3p_2Po[1/2]   meas        10
287.24    1.5  3s2_3p_2Po[3/2]   meas        10
42824.29  0.5  3s_3p2_4Pe[1/2]   meas        10
42932.62  1.5  3s_3p2_4Pe[3/2]   meas        10
43107.91  2.5  3s_3p2_4Pe[5/2]   meas        10
"""

si2_lines_head = """
0.0      42824.29  1.148200e-05      0.5      0.5    233.5123
0.0      42932.62  7.128000e-08      0.5      1.5    232.9231
0.0      55309.35  1.527600e-03      0.5      1.5    180.8013
0.0      65500.47  2.558000e-01      0.5      0.5    152.6707
0.0      76665.35  2.124000e-01      0.5      0.5    130.4370
"""

si2_col_head = """
3.1550 3.1150 3.0750 3.0450 3.0300 3.0100 2.9850 2.9450 2.8850 2.7900 2.5250 2.1800 1.835 1.5450
0.2330 0.2530 0.2590 0.2575 0.2555 0.2535 0.2510 0.2480 0.2435 0.2380 0.2215 0.1995 0.172 0.1420
0.4060 0.4090 0.4035 0.3945 0.3910 0.3875 0.3845 0.3805 0.3745 0.3665 0.3425 0.3075 0.264 0.2175
0.3205 0.3115 0.3035 0.2975 0.2960 0.2955 0.2945 0.2925 0.2885 0.2815 0.2595 0.2290 0.194 0.1565
1.3450 1.3800 1.3900 1.3700 1.3450 1.3050 1.2500 1.1850 1.1100 1.0300 0.8650 0.7150 0.585 0.4770
"""

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
@pytest.fixture()
def si2_levels_head_df():
    return pd.read_csv(StringIO(si2_levels_head), delim_whitespace=True, names=['energy', 'j', 'label', 'method', 'priority'])

@with_refdata
@pytest.fixture()
def si2_lines_head_df():
    return pd.read_csv(StringIO(si2_lines_head), delim_whitespace=True, names=['energy_lower', 'energy_upper', 'gf', 'j_lower', 
                                                                                'j_upper', 'wavelength'])
@with_refdata
@pytest.fixture()
def si2_col_head_df():
    return pd.read_csv(StringIO(si2_col_head), delim_whitespace=True, names=range(14))

@with_refdata
def test_si2_osc_kurucz(si2_osc_kurucz_fname):
    parser = CMFGENEnergyLevelsParser(si2_osc_kurucz_fname)
    n = int(parser.header['Number of energy levels'])
    assert parser.base.shape[0] == n
    assert list(parser.base.columns) == ['label', 'g', 'E(cm^-1)', '10^15 Hz', 'eV', 'Lam(A)', 'ID', 'ARAD', 'C4', 'C6']

@with_refdata
def test_fevi_osc_kb_rk(fevi_osc_kb_rk_fname):
    parser = CMFGENOscillatorStrengthsParser(fevi_osc_kb_rk_fname)
    n = int(parser.header['Number of transitions'])
    assert parser.base.shape[0] == n
    assert list(parser.base.columns) == ['label_lower', 'label_upper', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    assert np.isclose(parser.base.iloc[0,2], 1.94e-02)

@with_refdata
def test_p2_osc(p2_osc_fname):
    parser = CMFGENOscillatorStrengthsParser(p2_osc_fname)
    n = int(parser.header['Number of transitions'])
    assert parser.base.shape[0] == n
    assert list(parser.base.columns) == ['label_lower', 'label_upper', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    assert np.isnan(parser.base.iloc[0,7])
    assert np.isclose(parser.base.iloc[0,8], 3.)
    assert np.isnan(parser.base.iloc[1,7])
    assert np.isclose(parser.base.iloc[1,8], 25.)
    assert np.isclose(parser.base.iloc[2,7], 1532.51)
    assert np.isclose(parser.base.iloc[3,7], 1301.87)

@with_refdata
def test_vi_osc(vi_osc_fname):
    parser = CMFGENOscillatorStrengthsParser(vi_osc_fname)
    assert parser.base.empty

@with_refdata
def test_he2_col(he2_col_fname):
    parser = CMFGENCollisionalStrengthsParser(he2_col_fname)
    assert parser.base.shape[0] == 465
    assert parser.base.shape[1] == 11
    assert parser.base.iloc[-1,0] == '30___'
    assert parser.base.iloc[-1,1] == 'I'

@with_refdata
def test_ariii_col(ariii_col_fname):
    parser = CMFGENCollisionalStrengthsParser(ariii_col_fname)
    n = int(parser.header['Number of transitions'])
    assert parser.base.shape == (n, 13)

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
def test_hyd_l(hyd_l_fname):
    parser = CMFGENHydLParser(hyd_l_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    assert parser.base.shape == (465, 97)
    assert parser.base.loc[(11, 3)].values[5] == -6.226968
    assert parser.base.loc[(21, 20)].values[2] == -10.3071
    assert_allclose(
        parser.base.columns[:4], [1.1 ** 0, 1.1 ** 1, 1.1 ** 2, 1.1 ** 3]
    )

@with_refdata
def test_gbf_n(gbf_n_fname):
    parser = CMFGENHydGauntBfParser(gbf_n_fname)
    assert parser.header["Maximum principal quantum number"] == "30"
    assert parser.base.shape == (30, 145)
    assert (
        round(parser.base.loc[3].values[3], 7) == 0.9433558
    )  # Rounding is needed as a result of undoing the unit conversion
    assert round(parser.base.loc[18].values[11], 7) == 1.008855
    assert_allclose(
        parser.base.columns[:4], [1.1 ** 0, 1.1 ** 1, 1.1 ** 2, 1.1 ** 3]
    )

@with_refdata
def test_reader_levels_shape(si1_reader):
    assert si1_reader.levels.shape == (157, 5)

@with_refdata
def test_reader_lines_shape(si1_reader):
    assert si1_reader.lines.shape == (4196, 6)

@with_refdata
def test_reader_levels_head(si1_reader, si2_levels_head_df):
    assert_frame_equal(si1_reader.levels.head(5).reset_index(drop=True), 
                        si2_levels_head_df)

@with_refdata
def test_reader_lines_head(si1_reader, si2_lines_head_df):
    assert_frame_equal(si1_reader.lines.head(5).reset_index(drop=True), 
                        si2_lines_head_df)

@with_refdata
def test_reader_col_head(si1_reader, si2_col_head_df):
    assert_frame_equal(si1_reader.collisions.head(5).reset_index(drop=True), 
                        si2_col_head_df)


# test with different combinations of input args
@pytest.mark.array_compare
@pytest.mark.parametrize("threshold_energy_ryd", [0.053130732819562695])
@pytest.mark.parametrize("fit_coeff_list", [[34.4452, 1.0, 2.0]])
def test_get_seaton_phixs_table(threshold_energy_ryd, fit_coeff_list):
    phixs_table = get_seaton_phixs_table(threshold_energy_ryd, *fit_coeff_list)
    
    # TODO: generate reference

    # phixs_table_seaton_fname = os.path.join(data_dir, 'phixs_table_seaton.npy')
    # phixs_table_seaton = np.load(phixs_table_seaton_fname)
    # assert_allclose(phixs_table_seaton, phixs_table)
    return phixs_table
    
