import pandas as pd
import hashlib


def hash_pandas_object(pd_object, algorithm="md5"):
    """Hash Pandas objects.

    Parameters
    ----------
    pd_object : pandas.Series or pandas.DataFrame
        Pandas object to be hashed.
    algorithm : str, optional
        Algorithm available in `hashlib`, by default "md5"

    Returns
    -------
    str
        Hash values.

    Raises
    ------
    ValueError
        If `algorithm` is not available in `hashlib`.
    """
    algorithm = algorithm.lower()

    if hasattr(hashlib, algorithm):
        hash_func = getattr(hashlib, algorithm)

    else:
        raise ValueError('algorithm not supported')

    return hash_func(pd.util.hash_pandas_object(pd_object).values).hexdigest()