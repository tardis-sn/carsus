import hashlib
import pickle

def serialize_pandas_object(pd_object):
    """Serialize Pandas objects with Pickle.

    Parameters
    ----------
    pd_object : pandas.Series or pandas.DataFrame
        Pandas object to be serialized with Pickle.

    Returns
    -------
    Pickle serialized Python object.
    """
    return pickle.dumps(pd_object)


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

    return hash_func(serialize_pandas_object(pd_object)).hexdigest()