import hashlib
import requests
from io import BytesIO
import pandas as pd
import numpy as np
from pyparsing import ParseResults
from carsus.util import convert_atomic_number2symbol
import requests
from requests.adapters import HTTPAdapter, Retry

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
    species: tuple (atomic_number, ion_charge)

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
    atomic_number, ion_charge = species
    chianti_ion_name = convert_atomic_number2symbol(atomic_number).lower() + '_' + str(ion_charge + 1)
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
        response = retry_request(fname, "get")
        data = response.content

    else:
        with open(fname, 'rb') as f:
            data = f.read()

    buffer = BytesIO(data)
    checksum = hashlib.md5(buffer.getbuffer()).hexdigest()

    return buffer, checksum


def retry_request(
    url,
    method,
    n_retry=15,
    backoff_factor=1,
    status_forcelist=[502, 503, 504, 400, 495],
    **kwargs
):
    """Retry an HTTP request.

    Parameters
    ----------
    url : str
        URL to send request to.
    method : str
        HTTP request method.
    n_retry : int, default: 15, optional
    backoff_factor : int, default: 1, optional
    status_forcelist : list, default: [502, 503, 504, 400, 495], optional
    **kwargs : dict, optional

    Returns
    -------
    response : requests.Response
    """
    sess = requests.Session()
    retries = Retry(
        total=n_retry, backoff_factor=backoff_factor, status_forcelist=status_forcelist
    )
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    requests_method = getattr(sess, method)
    response = requests_method(url, **kwargs)
    sess.close()
    return response

def get_lvl_index2id(df, levels_all):
    """
    Matches `level_index` with level IDs for a given DataFrame.

    """
    # TODO: re-write this method without a for loop
    ion = df.index.unique()
    lvl_index2id = levels_all.set_index(["atomic_number", "ion_number"]).loc[ion]
    lvl_index2id = lvl_index2id.reset_index()

    lower_level_id = []
    upper_level_id = []

    df = df.reset_index()
    for row in df.itertuples():
        llid = row.level_index_lower
        ulid = row.level_index_upper

        upper = lvl_index2id.at[ulid, "level_id"]
        lower = lvl_index2id.at[llid, "level_id"]

        lower_level_id.append(lower)
        upper_level_id.append(upper)

    df["lower_level_id"] = pd.Series(lower_level_id)
    df["upper_level_id"] = pd.Series(upper_level_id)

    return df

def create_artificial_fully_ionized(levels):
    """
    Returns a DataFrame with fully ionized levels.

    """
    fully_ionized_levels = []

    for atomic_number, _ in levels.groupby("atomic_number"):
        fully_ionized_levels.append(
            (-1, atomic_number, atomic_number, 0, 0.0, 1, True)
        )

    levels_columns = [
        "level_id",
        "atomic_number",
        "ion_number",
        "level_number",
        "energy",
        "g",
        "metastable",
    ]

    fully_ionized_levels_dtypes = [
        (key, levels.dtypes[key]) for key in levels_columns
    ]

    fully_ionized_levels = np.array(
        fully_ionized_levels, dtype=fully_ionized_levels_dtypes
    )

    return pd.DataFrame(data=fully_ionized_levels)

def exclude_artificial_levels(levels_df):
    """Removes artificially created levels from a dataframe of levels

    Parameters
    ----------
    levels_df : pandas.DataFrame
        Levels dataframe

    Returns
    -------
    pandas.DataFrame
        Filtered levels dataframe
    """
    return levels_df.loc[levels_df["level_id"] != -1].set_index("level_id")