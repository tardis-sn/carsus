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
                              CMFGENPhotoionizationCrossSectionParser,
                              CMFGENHydLParser,
                              CMFGENHydGauntBfParser,
                              CMFGENReader
                             )

with_refdata = pytest.mark.skipif(
    not pytest.config.getoption("--refdata"),
    reason="--refdata folder not specified"
)

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
def si1_data_dict(si2_osc_kurucz_fname):
    si1_levels = CMFGENEnergyLevelsParser(si2_osc_kurucz_fname)  #  (carsus) Si 1 == Si II
    si1_lines = CMFGENOscillatorStrengthsParser(si2_osc_kurucz_fname)
    return {'Si 1': dict(levels = si1_levels, lines = si1_lines)}

@with_refdata
@pytest.fixture()
def si1_reader(si1_data_dict):
    return CMFGENReader(si1_data_dict)

@with_refdata
@pytest.fixture()
def si2_levels_head_df():
    return pd.read_csv(StringIO(si2_levels_head), delim_whitespace=True, names=['energy', 'j', 'label', 'method', 'priority'])

@with_refdata
@pytest.fixture()
def si2_lines_head_df():
    return pd.read_csv(StringIO(si2_lines_head),delim_whitespace=True, names=['energy_lower', 'energy_upper', 'gf', 'j_lower', 
                                                                                'j_upper', 'wavelength'])

@with_refdata
@pytest.fixture()
def si1_data_dict(si2_osc_kurucz_fname):
    si1_levels = CMFGENEnergyLevelsParser(si2_osc_kurucz_fname).base  #  (carsus) Si 1 == Si II
    si1_lines = CMFGENOscillatorStrengthsParser(si2_osc_kurucz_fname).base
    return {(14,1): dict(levels = si1_levels, lines = si1_lines)}

@with_refdata
@pytest.fixture()
def si1_reader(si1_data_dict):
    return CMFGENReader(si1_data_dict)

@with_refdata
@pytest.fixture()
def si2_levels_head_df():
    return pd.read_csv(StringIO(si2_levels_head), delim_whitespace=True, names=['energy', 'j', 'label', 'method', 'priority'])

@with_refdata
@pytest.fixture()
def si2_lines_head_df():
    return pd.read_csv(StringIO(si2_lines_head),delim_whitespace=True, names=['energy_lower', 'energy_upper', 'gf', 'j_lower', 
                                                                                'j_upper', 'wavelength'])

@with_refdata
def test_si2_osc_kurucz(si2_osc_kurucz_fname):
    parser = CMFGENEnergyLevelsParser(si2_osc_kurucz_fname)
    n = int(parser.meta['Number of energy levels'])
    assert parser.base.shape[0] == n
    assert parser.columns == ['Configuration', 'g', 'E(cm^-1)', '10^15 Hz', 'eV', 'Lam(A)', 'ID', 'ARAD', 'C4', 'C6']

@with_refdata
def test_fevi_osc_kb_rk(fevi_osc_kb_rk_fname):
    parser = CMFGENOscillatorStrengthsParser(fevi_osc_kb_rk_fname)
    n = int(parser.meta['Number of transitions'])
    assert parser.base.shape[0] == n
    assert parser.columns == ['label_lower', 'label_upper', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    assert np.isclose(parser.base.iloc[0,2], 1.94e-02)

@with_refdata
def test_p2_osc(p2_osc_fname):
    parser = CMFGENOscillatorStrengthsParser(p2_osc_fname)
    n = int(parser.meta['Number of transitions'])
    assert parser.base.shape[0] == n
    assert parser.columns == ['label_lower', 'label_upper', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
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
    n = int(parser.meta['Number of transitions'])
    assert parser.base.shape == (n, 13)

@with_refdata
def test_si2_pho(si2_pho_fname):
    parser = CMFGENPhotoionizationCrossSectionParser(si2_pho_fname)
    n = int(parser.meta['Number of energy levels'])
    m = int(parser.base[0].attrs['Points'])
    assert len(parser.base) == n
    assert parser.base[0].shape == (m, 2)

@with_refdata
def test_coiv_pho(coiv_pho_fname):
    parser = CMFGENPhotoionizationCrossSectionParser(coiv_pho_fname)
    n = int(parser.meta['Number of energy levels'])
    assert len(parser.base) == n
    assert parser.base[0].shape == (3, 8)


@with_refdata
def test_hyd_l(hyd_l_fname):
    parser = CMFGENHydLParser(hyd_l_fname)
    assert parser.meta["Maximum principal quantum number"] == "30"
    assert parser.base.shape == (465, 97)
    assert parser.base.loc[(11, 3)].values[5] == -16.226968
    assert parser.base.loc[(21, 20)].values[2] == -20.3071
    assert_allclose(
        parser.columns[:4], [1.1 ** 0, 1.1 ** 1, 1.1 ** 2, 1.1 ** 3]
    )

@with_refdata
def test_gbf_n(gbf_n_fname):
    parser = CMFGENHydGauntBfParser(gbf_n_fname)
    assert parser.meta["Maximum principal quantum number"] == "30"
    assert parser.base.shape == (30, 145)
    assert (
        round(parser.base.loc[3].values[3], 7) == 0.9433558
    )  # Rounding is needed as a result of undoing the unit conversion
    assert round(parser.base.loc[18].values[11], 7) == 1.008855
    assert_allclose(
        parser.columns[:4], [1.1 ** 0, 1.1 ** 1, 1.1 ** 2, 1.1 ** 3]
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
