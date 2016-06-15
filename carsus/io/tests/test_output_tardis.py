import pytest

from carsus.io.output.tardis import create_basic_atom_data, create_ionization_data
from carsus.io.nist import NISTWeightsCompIngester, NISTIonizationEnergiesIngester
from carsus.model import AtomWeight, Atom

@pytest.fixture
def weightscomp_ingester(test_session):
    ingester = NISTWeightsCompIngester(test_session)
    return ingester


@pytest.fixture
def ioniz_energies_ingester(test_session):
    ingester = NISTIonizationEnergiesIngester(test_session)
    return ingester

@pytest.mark.remote_data
def test_create_basic_atomic_data(weightscomp_ingester, test_session):
    weightscomp_ingester.download()
    weightscomp_ingester.ingest()
    test_session.commit()
    basic_atom_data_df = create_basic_atom_data(test_session)


@pytest.mark.remote_data
def test_create_ionization_energies_data(ioniz_energies_ingester, test_session):
    ioniz_energies_ingester.download()
    ioniz_energies_ingester.ingest()
    test_session.commit()
    ioniz_energies_ingester = create_ionization_data(test_session)