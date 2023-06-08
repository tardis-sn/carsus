import pytest

from pyparsing import Word, Dict, Group, alphas, nums, Suppress
from carsus.io.base import BasePyparser, BaseIngester
from carsus.model import DataSource


@pytest.fixture
def entry():
    name = Word(alphas, f'{alphas}_')
    value = Word(nums, f"{nums}.").setResultsName('nominal_value')
    uncert = Word(nums).setResultsName('std_dev')
    value_uncert = value + Suppress("(") + uncert + Suppress(")")
    return Dict( Group(name + Suppress("=") + value_uncert ) )


@pytest.fixture
def aw_pyparser(entry):
    columns = ["atomic_weight_nominal_value", "atomic_weight_std_dev"]
    return BasePyparser(entry, columns)


class ConcreteBaseIngester(BaseIngester):

    ds_short_name = "cncrt"

    def download(self):
        return "Downloaded"

    def ingest(self, session):
        data_source = DataSource.as_unique(session, short_name=self.ds_short_name)
        return "Ingested"


@pytest.fixture
def ingester():
    return ConcreteBaseIngester(parser=object(), downloader=object())