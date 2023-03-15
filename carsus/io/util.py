import hashlib
import requests
from io import BytesIO
from pyparsing import ParseResults
from carsus.util import convert_atomic_number2symbol

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

def to_flat_dict(tokens, parent_key='', sep='_'):
    """
    Creates a flattened dictionary from the named values in tokens.

    E.g. suppose tokens.dump() output is
        - isotopic_comp: ['0.000629', '(', '7', ')']
            - nominal_value: 0.000629
            - std_dev: 7e-06
    Then the new dictionary is {'isotopic_comp_nominal_value': 0.000629, 'isotopic_comp_std_dev':7e-06}

    Parameters
    ----------
    tokens: ~pyparsing.ParseResults
    parent_key: ~str -- is used in recursive calls; you don't need to pass this
    sep: ~str -- is used to concatenate keys (default: "_")

    Returns: ~dict
    -------

    """
    tokens_dict = dict()
    for key, item in tokens.items():
        new_key = parent_key + sep + key if parent_key else key
        if isinstance(item, ParseResults):
            tokens_dict.update(to_flat_dict(item, parent_key=new_key, sep=sep))
        else:
            tokens_dict[new_key] = item
    return tokens_dict


def to_nom_val_and_std_dev(interval):
    """
    For a given interval [mu - sigma, mu + sigma] returns (mu, sigma)
    (Here mu is nominal value and sigma is standard deviation)

    Parameters
    ----------
    interval: ~list [lwr_bnd, upr_bnd]

    Returns: ~tuple

    """
    lwr_bnd, upr_bnd = interval
    sigma = (upr_bnd - lwr_bnd)/2
    mu = lwr_bnd + sigma
    return (mu, sigma)


def convert_species_tuple2chianti_str(species):
    """
    Convert a species tuple to the ion name format used in `chiantipy`.

    Parameters
    -----------
    species: tuple (atomic_number, ion_number)

    Returns
    --------
    str
        ion name in the chiantipy format

    Examples
    ---------
    >>> convert_species_tuple2chianti_str((1,0))
    'h_1'

    >>> convert_species_tuple2chianti_str((14,1))
    'si_2'

    """
    atomic_number, ion_number = species
    chianti_ion_name = convert_atomic_number2symbol(atomic_number).lower() + '_' + str(ion_number + 1)
    return chianti_ion_name


def read_from_buffer(fname):
    """Read a local or remote file into a buffer and get its MD5 
    checksum. To be used with `pandas.read_csv` or `pandas.read_fwf`
    functions.

    Parameters
    ----------
    fname : str
        local path or remote url

    Returns
    -------
    bytes, str
        data from text file, MD5 checksum
    """    
    if fname.startswith("http"):
        response = requests.get(fname, verify=False)
        data = response.content

    else:
        with open(fname, 'rb') as f:
            data = f.read()

    buffer = BytesIO(data)
    checksum = hashlib.md5(buffer.getbuffer()).hexdigest()

    return buffer, checksum
