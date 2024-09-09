import os
import re
import carsus
import numpy as np

from collections import OrderedDict


def get_data_path(fname):
    return os.path.join(
        os.path.dirname(carsus.__file__), 'data', fname
    )

ATOMIC_SYMBOLS_DATA = np.genfromtxt(get_data_path('basic_atomic_data.csv'), skip_header=1,
                                    delimiter=',', usecols=(0, 1), names=['atomic_number', 'symbol'], encoding='utf-8', dtype=None)

SYMBOL2ATOMIC_NUMBER = OrderedDict(zip(ATOMIC_SYMBOLS_DATA['symbol'],
                                       ATOMIC_SYMBOLS_DATA['atomic_number']))
ATOMIC_NUMBER2SYMBOL = OrderedDict(zip(ATOMIC_SYMBOLS_DATA['atomic_number'],
                                       ATOMIC_SYMBOLS_DATA['symbol']))


def convert_camel2snake(name):
    """
    Convert CamelCase to snake_case.

    http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-snake-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def convert_wavelength_vacuum2air(wavelength_vacuum):
    """
    Convert vacuum wavelength to air wavelength
    Parameters
    -----------
    wavelength_vacuum: float
        Vacuum wavelength in Angstroms

    Returns
    --------
    float
        Air wavelength in Angstroms
    """
    sigma2 = (1e4/wavelength_vacuum)**2.
    fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return wavelength_vacuum/fact


def convert_wavelength_air2vacuum(wavelength_air):
    """
    Convert air wavelength to vacuum wavelength
    Parameters
    -----------
    wavelength_air: float
        Air wavelength in Angstroms

    Returns
    --------
    float
        Vacuum wavelength in Angstroms
    """
    sigma2 = (1e4/wavelength_air)**2.
    fact = 1.0 + 5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/(57.362 - sigma2)

    return wavelength_air * fact


def convert_atomic_number2symbol(atomic_number):
    return ATOMIC_NUMBER2SYMBOL[atomic_number]


def convert_symbol2atomic_number(symbol):
    return SYMBOL2ATOMIC_NUMBER[symbol]


def query_columns(query):
    return [v['name'] for v in query.column_descriptions]
