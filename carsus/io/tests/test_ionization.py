import pytest
import pandas as pd

from pandas.testing import assert_series_equal

from carsus.io.nist.ionization import (NISTIonizationEnergiesParser,
                                       NISTIonizationEnergies)


test_data = """
<h2> Be specta </h2>
<pre>
--------------------------------------------------------------------------------------------
At. num | Ion Charge | Ground Shells | Ground Level |      Ionization Energy (a) (eV)      |
--------|------------|---------------|--------------|--------------------------------------|
      4 |          0 | 1s2.2s2       | 1S0          |                 9.3226990(70)        |
      4 |         +1 | 1s2.2s        | 2S*<1/2>      |                18.211153(40)         |
      4 |         +2 | 1s2           | (1,3/2)<2>          |   <a class=bal>[</a>153.8961980(40)<a class=bal>]</a>       |
      4 |         +3 | 1s            | 2S<1/2>      |   <a class=bal>(</a>217.7185766(10)<a class=bal>)</a>       |
--------------------------------------------------------------------------------------------
</pre>
"""

expected_at_num = [4, 4, 4, 4]

expected_ion_charge = [0, 1, 2, 3]

expected_indices = list(zip(expected_at_num, expected_ion_charge))

expected_ground_shells = ('ground_shells',
                          ['1s2.2s2', '1s2.2s', '1s2', '1s']
                          )

expected_ground_level = ('ground_level',
                         ['1S0', '2S*<1/2>', '(1,3/2)<2>', '2S<1/2>']
                         )

expected_j = ('J', [0.0, 0.5, 2.0, 0.5])

expected_ioniz_energy_value = ('ionization_energy_value',
                               [9.3226990, 18.211153, 153.8961980, 217.7185766]
                               )

expected_ioniz_energy_uncert = ('ionization_energy_uncert',
                                [7e-6, 4e-5, 4e-6, 1e-6]
                                )

expected_ioniz_energy_method = ('ionization_energy_method',
                                ['meas', 'meas', 'intrpl', 'theor']
                                )


@pytest.fixture
def ioniz_energies_parser():
    parser = NISTIonizationEnergiesParser(input_data=test_data)
    return parser


@pytest.fixture
def ioniz_energies(ioniz_energies_parser):
    return ioniz_energies_parser.prepare_ioniz_energies()


@pytest.fixture
def ground_levels(ioniz_energies_parser):
    return ioniz_energies_parser.prepare_ground_levels()


@pytest.fixture(params=[expected_ground_shells,
                        expected_ground_level, expected_ioniz_energy_value,
                        expected_ioniz_energy_uncert, expected_ioniz_energy_method])
def expected_series_ioniz_energies(request):
    index = pd.MultiIndex.from_tuples(tuples=expected_indices,
                                       names=['atomic_number', 'ion_charge'])
    name, data = request.param
    return pd.Series(data=data, name=name, index=index)


@pytest.fixture(params=[expected_j])
def expected_series_ground_levels(request):
    index = pd.MultiIndex.from_tuples(tuples=expected_indices,
                                       names=['atomic_number', 'ion_charge'])
    name, data = request.param
    return pd.Series(data=data, name=name, index=index)


def test_prepare_ioniz_energies_null_values(ioniz_energies):
    assert all(pd.notnull(ioniz_energies["ionization_energy_value"]))


def test_prepare_ioniz_energies(ioniz_energies, expected_series_ioniz_energies):
    series = ioniz_energies[expected_series_ioniz_energies.name]
    assert_series_equal(series, expected_series_ioniz_energies)



def test_prepare_ground_levels(ground_levels, expected_series_ground_levels):
    series = ground_levels[expected_series_ground_levels.name]
    assert_series_equal(series, expected_series_ground_levels)




def test_ground_levels_missing_j():
    ionization_energies = NISTIonizationEnergies(spectra="Nd")
    ground_levels = ionization_energies.get_ground_levels()
    ground_levels = ground_levels.set_index(['atomic_number', 'ion_charge'])

    assert ground_levels.loc[(60, 5)]['g'] == 1
    assert ground_levels.loc[(60, 6)]['g'] == 1
    assert ground_levels.loc[(60, 7)]['g'] == 1
    assert ground_levels.loc[(60, 8)]['g'] == 1
    assert ground_levels.loc[(60, 9)]['g'] == 1
    assert ground_levels.loc[(60, 10)]['g'] == 1

def test_nist_asd_version():
    nist_ionization = NISTIonizationEnergies('H')
    version = nist_ionization.version
    version_split = version.split('.')

    assert len(version_split) > 1
    to_int = [ int(i) for i in version_split ]

