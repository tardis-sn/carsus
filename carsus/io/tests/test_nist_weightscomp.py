import pytest
import pandas as pd

from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_almost_equal
from carsus.io.nist import (NISTWeightsCompIngester, 
                            NISTWeightsCompPyparser,
                            NISTWeightsComp)
from carsus.io.nist.weightscomp_grammar import *
from carsus.model import AtomWeight

test_input = """
Atomic Number = 35
Atomic Symbol = Br
Mass Number = 79
Relative Atomic Mass = 78.9183376(14)
Isotopic Composition = 0.5069(7)
Standard Atomic Weight = [79.901,79.907]
Notes =

Atomic Number = 38
Atomic Symbol = Sr
Mass Number = 84
Relative Atomic Mass = 83.9134191(13)
Isotopic Composition = 0.0056(1)
Standard Atomic Weight = 87.62(1)
Notes = g,r

Atomic Number = 38
Atomic Symbol = Sr
Mass Number = 86
Relative Atomic Mass = 85.9092606(12)
Isotopic Composition = 0.0986(1)
Standard Atomic Weight = 87.62(1)
Notes = g,r

Atomic Number = 43
Atomic Symbol = Tc
Mass Number = 98
Relative Atomic Mass = 97.9072124(36)
Isotopic Composition =
Standard Atomic Weight = [98]
Notes =

Atomic Number = 43
Atomic Symbol = Tc
Mass Number = 99
Relative Atomic Mass = 98.9062508(10)
Isotopic Composition =
Standard Atomic Weight = [98]
Notes =
"""

expected_dict=[
    {'atomic_number': 35, 'atomic_weight_nominal_value': 79.904, 'atomic_weight_std_dev': 3e-3},
    {'atomic_number': 38, 'atomic_weight_nominal_value': 87.62, 'atomic_weight_std_dev': 1e-2},
    {'atomic_number': 43, 'atomic_weight_nominal_value': 97.9072124, 'atomic_weight_std_dev': 36e-7}
]

expected_tuples = [
    (43, 97.9072124, 36e-7),
    (35, 79.904, 3e-3),
    (38, 87.62, 1e-2)
]


@pytest.fixture
def weightscomp_pyparser():
    return NISTWeightsCompPyparser(input_data=test_input)

@pytest.fixture
def atomic(weightscomp_pyparser):
    return weightscomp_pyparser.prepare_atomic_dataframe()


@pytest.fixture
def expected():
    return pd.DataFrame(data=expected_dict, columns=[ATOM_NUM_COL, AW_VAL_COL, AW_SD_COL]).set_index(ATOM_NUM_COL)


@pytest.fixture
def weightscomp_ingester(memory_session):
    ingester = NISTWeightsCompIngester(memory_session)
    ingester.parser(test_input)
    return ingester


def test_weightscomp_pyparser_base_index(weightscomp_pyparser):
    assert weightscomp_pyparser.base.index.names == [ATOM_NUM_COL, MASS_NUM_COL]


def test_weightscomp_pyparser_prepare_atomic_index(atomic):
    assert atomic.index.name == ATOM_NUM_COL


def test_weightscomp_pyparser_prepare_atomic(atomic, expected):
    assert_frame_equal(atomic, expected, check_names=False)


@pytest.mark.parametrize("atomic_number, value, uncert", expected_tuples)
def test_weightscomp_ingest_nonexisting_atomic_weights(atomic_number, value, uncert, weightscomp_ingester, memory_session):
    weightscomp_ingester.ingest()
    atom_weight = memory_session.query(AtomWeight).\
        filter(AtomWeight.atomic_number==atomic_number).\
        filter(AtomWeight.data_source==weightscomp_ingester.data_source).one()
    assert_almost_equal(atom_weight.quantity.value, value)
    assert_almost_equal(atom_weight.uncert, uncert)


@pytest.mark.remote_data
def test_weightscomp_ingest_default_count(memory_session):
    weightscomp_ingester = NISTWeightsCompIngester(memory_session)
    weightscomp_ingester.ingest(atomic_weights=True)
    assert memory_session.query(AtomWeight).\
               filter(AtomWeight.data_source==weightscomp_ingester.data_source).count() == 94


@pytest.mark.remote_data
def test_nist_weights_version():
    nist_weights = NISTWeightsComp()
    version = nist_weights.version
    version_split = version.split('.')

    assert len(version_split) > 1
    to_int = [ int(i) for i in version_split ]
