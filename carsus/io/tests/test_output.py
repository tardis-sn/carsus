import pytest

from carsus.io.output import read_basic_atom_data
from carsus.io.nist import NISTWeightsCompIngester
from carsus.model import AtomWeight, Atom

@pytest.fixture
def weightscomp_ingester(test_session):
    ingester = NISTWeightsCompIngester(test_session)
    return ingester


@pytest.mark.remote_data
def test_read_basic_atomic_data(weightscomp_ingester, test_session):
    weightscomp_ingester.download()
    weightscomp_ingester.ingest()
    test_session.commit()
    basic_atom_data_df = read_basic_atom_data(test_session)