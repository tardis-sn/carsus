import pytest
import pandas as pd

from pandas.util.testing import assert_series_equal
from numpy.testing import assert_almost_equal
from sqlalchemy.orm import joinedload
from carsus.model import Ion
from carsus.io.nist.ionization import  NISTIonizationEnergiesParser, NISTIonizationEnergiesIngester


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


@pytest.fixture
def ioniz_energies_ingester(memory_session):
    ingester = NISTIonizationEnergiesIngester(memory_session)
    ingester.parser(test_data)
    return ingester

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


@pytest.mark.parametrize("index, value, uncert",
                         zip(expected_indices,
                             expected_ioniz_energy_value[1],
                             expected_ioniz_energy_uncert[1]))
def test_ingest_ionization_energies(index, value, uncert, memory_session, ioniz_energies_ingester):

    ioniz_energies_ingester.ingest(ionization_energies=True, ground_levels=False)

    atomic_number, ion_charge = index
    ion = memory_session.query(Ion).options(joinedload('ionization_energies')).get((atomic_number, ion_charge))

    ion_energy = ion.ionization_energies[0]
    assert_almost_equal(ion_energy.quantity.value, value)
    assert_almost_equal(ion_energy.uncert, uncert)


@pytest.mark.parametrize("index, exp_j", zip(expected_indices, expected_j[1]))
def test_ingest_ground_levels(index, exp_j, memory_session, ioniz_energies_ingester):
    ioniz_energies_ingester.ingest(ionization_energies=True, ground_levels=True)

    atomic_number, ion_charge = index
    ion = memory_session.query(Ion).options(joinedload('levels')).get((atomic_number, ion_charge))
    ground_level = ion.levels[0]
    assert_almost_equal(ground_level.J, exp_j)


@pytest.mark.remote_data
def test_ingest_nist_asd_ion_data(memory_session):
    ingester = NISTIonizationEnergiesIngester(memory_session, spectra="h-uuh")
    ingester.ingest(ionization_energies=True, ground_levels=True)
