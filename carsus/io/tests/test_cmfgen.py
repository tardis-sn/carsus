import os
import glob
import pytest
import numpy as np
from numpy.testing import assert_allclose
from carsus.io.cmfgen import (CMFGENEnergyLevelsParser,
                              CMFGENOscillatorStrengthsParser,
                              CMFGENCollisionalDataParser,
                              CMFGENPhotoionizationCrossSectionParser,
                             )

with_refdata = pytest.mark.skipif(
    not pytest.config.getoption("--refdata"),
    reason="--refdata folder not specified"
)

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
    return os.path.join(refdata_path, 'cmfgen', 'collisional_data', 'he2col.dat')

@with_refdata
@pytest.fixture()
def ariii_col_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'collisional_data', 'col_ariii')

@with_refdata
@pytest.fixture()
def si2_pho_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'photoionization_cross_sections', 'phot_nahar_A')

@with_refdata
@pytest.fixture()
def coiv_pho_fname(refdata_path):
    return os.path.join(refdata_path, 'cmfgen', 'photoionization_cross_sections', 'phot_data_gs')

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
    assert parser.columns == ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    assert np.isclose(parser.base.iloc[0,2], 1.94e-02)

@with_refdata
def test_p2_osc(p2_osc_fname):
    parser = CMFGENOscillatorStrengthsParser(p2_osc_fname)
    n = int(parser.meta['Number of transitions'])
    assert parser.base.shape[0] == n
    assert parser.columns == ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
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
    parser = CMFGENCollisionalDataParser(he2_col_fname)
    assert parser.base.shape[0] == 465
    assert parser.base.shape[1] == 11
    assert parser.base.iloc[-1,0] == '30___'
    assert parser.base.iloc[-1,1] == 'I'

@with_refdata
def test_ariii_col(ariii_col_fname):
    parser = CMFGENCollisionalDataParser(ariii_col_fname)
    n = int(parser.meta['Number of transitions'])
    assert parser.base.shape == (n, 13)

@with_refdata
def test_si2_pho(si2_pho_fname):
    parser = CMFGENPhotoionizationCrossSectionParser(si2_pho_fname)
    n = int(parser.meta['Number of energy levels'])
    m = int(parser.base[0]._meta['Points'])
    assert len(parser.base) == n
    assert parser.base[0].shape == (m, 2)

@with_refdata
def test_coiv_pho(coiv_pho_fname):
    parser = CMFGENPhotoionizationCrossSectionParser(coiv_pho_fname)
    n = int(parser.meta['Number of energy levels'])
    assert len(parser.base) == n
    assert parser.base[0].shape == (3, 8)
