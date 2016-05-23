import pytest
import pandas as pd
from pandas.util.testing import assert_series_equal
from astropy import units as u
from numpy.testing import assert_almost_equal
from sqlalchemy import and_
from sqlalchemy.orm import joinedload
from carsus.model import Ion, Atom, DataSource, IonizationEnergy
from carsus.io.nist.ionization import download_ionization_energies, NISTIonizationEnergiesParser,\
    NISTIonizationEnergiesIngester


test_data = """
<h2> Be specta </h2>
<pre>
--------------------------------------------------------------------------------------------
At. num | Ion Charge | Ground Shells | Ground Level |      Ionization Energy (a) (eV)      |
--------|------------|---------------|--------------|--------------------------------------|
      4 |          0 | 1s2.2s2       | 1S0          |                 9.3226990(70)        |
      4 |         +1 | 1s2.2s        | 2S<1/2>      |                18.211153(40)         |
      4 |         +2 | 1s2           | 1S0          |   <a class=bal>[</a>153.8961980(40)<a class=bal>]</a>       |
      4 |         +3 | 1s            | 2S<1/2>      |   <a class=bal>(</a>217.7185766(10)<a class=bal>)</a>       |
--------------------------------------------------------------------------------------------
</pre>
"""

expected_at_num = [4, 4, 4, 4]

expected_ion_charge = [0, 1, 2, 3]

expected_indices = zip(expected_at_num, expected_ion_charge)

expected_ground_shells = ('ground_shells',
                          ['1s2.2s2', '1s2.2s', '1s2', '1s']
                          )

expected_ground_level = ('ground_level',
                         ['1S0', '2S<1/2>', '1S0', '2S<1/2>']
                         )

expected_ion_energy_value = ('ionization_energy_value',
                            [9.3226990, 18.211153, 153.8961980, 217.7185766]
                            )

expected_ion_energy_uncert = ('ionization_energy_uncert',
                             [7e-6, 4e-5, 4e-6, 1e-6]
                             )

expected_ion_energy_method = ('ionization_energy_method',
                             ['measured', 'measured', 'interpolated', 'theoretical']
                             )


@pytest.fixture
def ion_energies_parser():
    parser = NISTIonizationEnergiesParser(input_data=test_data)
    return parser


@pytest.fixture
def ion_energies_df(ion_energies_parser):
    return ion_energies_parser.prepare_ion_energies_df()


@pytest.fixture
def ion_energies_ingester():
    ingester = NISTIonizationEnergiesIngester()
    ingester.parser(test_data)
    return ingester

@pytest.fixture(params=[expected_ground_shells,
                        expected_ground_level, expected_ion_energy_value,
                        expected_ion_energy_uncert, expected_ion_energy_method])
def expected_series(request):
    index = pd.MultiIndex.from_tuples(tuples=expected_indices,
                                       names=['atomic_number', 'ion_charge'])
    name, data = request.param
    return pd.Series(data=data, name=name, index=index)


def test_prepare_ion_energies_df(ion_energies_df, expected_series):
    series = ion_energies_df[expected_series.name]
    assert_series_equal(series, expected_series)


@pytest.mark.parametrize("index, value, uncert",
                         zip(expected_indices,
                             expected_ion_energy_value[1],
                             expected_ion_energy_uncert[1]))
def test_ingest_test_data(index, value, uncert, test_session, ion_energies_ingester):
    ds = DataSource.as_unique(test_session, short_name="nist-asd-ionenergy")
    Be = test_session.query(Atom).get(4)
    Be_0 = Ion(atom=Be, ion_charge=0)
    Be_1 = Ion(atom=Be, ion_charge=1)
    Be_0.ionization_energies = [
        IonizationEnergy(quantity=13.6180540 * u.eV, data_source=ds, std_dev=6e-6)
    ]

    test_session.add_all([ds, Be_0, Be_1])

    ion_energies_ingester.ingest(test_session)

    atomic_number, ion_charge = index
    q = test_session.query(Ion).options(joinedload('ionization_energies')).\
            filter(and_(Ion.atomic_number==atomic_number,
                        Ion.ion_charge==ion_charge))
    for ion in q:
        ion_energy = ion.ionization_energies[0]
        assert_almost_equal(ion_energy.quantity.value, value)
        assert_almost_equal(ion_energy.std_dev, uncert)


@pytest.mark.remote_data
def test_ingest_nist_asd_ion_data(test_session):
    ingester = NISTIonizationEnergiesIngester()
    ingester.download()
    ingester.ingest(test_session)