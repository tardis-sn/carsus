"""This module defines base classes for parsers and ingesters."""

import pandas as pd

from carsus.io.util import to_flat_dict
from abc import ABCMeta, abstractmethod


class ParserError(ValueError):
    pass


class BaseParser(object):
    """
    Abstract base class for parsers.

    Attributes
    ----------
    base : pandas.DataFrame
        Contains parsed results from the provided input data.

    Methods
    -------
    load(input_data)
        Parses the input data and stores the results in the `base` attribute

    __call__(input_data)
        Call an instance with input data to invoke the `load` method.

    """
    __metaclass__ = ABCMeta

    def __init__(self, input_data=None):
        self.base = None
        if input_data is not None:
            self.load(input_data)

    @abstractmethod
    def load(self, input_data):
        pass

    def __call__(self, input_data):
        self.load(input_data)


class BasePyparser(BaseParser):
    """
    Abstract base class for parsers that use pyparsing grammar.

    Attributes
    ----------
    base : pandas.DataFrame
        Contains parsed results from the provided input data.

    grammar : pyparsing.ParseElement
        The grammar used to parse input.
        Its labeled tokens correspond to the columns of the `base`

    columns : list of str
        The column names of the `base`

    Methods
    -------
    load(input_data)
        Parses the input data and stores the results in the `base` attribute

    Notes
    -----
    Rationale: pyparsers have a specific load workflow illustrated below.

    Suppose a `base` of some parser has three columns::

        atomic_mass_nominal_value | atomic_mass_std_dev | notes

    The `load` method scans the input data with parser's `grammar`.
    The returned matches have nested labeled tokens that correspond to the columns.
    Say, one of the matches has the following nested tokens list::

        - atomic_mass: ['37.96273211', '(', '21', ')']
            - nominal_value: 37.96273211
            - std_dev: 2.1e-07

    The `load` method then infers the columns' values from
    the nested labels and adds the following row to the `base`::

        atomic_mass_nominal_value            37.9627
        atomic_mass_std_dev                  2.1e-07
        notes                                NaN

    """
    __metaclass__ = ABCMeta

    def __init__(self, grammar, columns, input_data=None):
        self.grammar = grammar
        self.columns = columns
        super(BasePyparser, self).__init__(input_data)

    def load(self, input_data):
        results = self.grammar.scanString(input_data)
        base = list()  # list of dicts that will be passed to the base
        for tokens, start, end in results:
            tokens_dict = to_flat_dict(tokens)  # make a flattened dict with the column names as keys
            base.append(tokens_dict)
        self.base = pd.DataFrame(data=base, columns=self.columns)

