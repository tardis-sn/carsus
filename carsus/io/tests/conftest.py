import pytest

from pyparsing import Word, Dict, Group, alphas, nums, Suppress
from carsus.io.base import BasePyparser


@pytest.fixture
def entry():
    name = Word(alphas, alphas+'_')
    value = Word(nums, nums+".").setResultsName('nominal_value')
    uncert = Word(nums).setResultsName('std_dev')
    value_uncert = value + Suppress("(") + uncert + Suppress(")")
    return Dict( Group(name + Suppress("=") + value_uncert ) )


@pytest.fixture
def aw_pyparser(entry):
    columns = ["atomic_weight_nominal_value", "atomic_weight_std_dev"]
    return BasePyparser(entry, columns)