import pyarrow as pa
import hashlib

def serialize_pandas_object(pd_object):
    """Serialize Pandas objects with PyArrow.

    Parameters
    ----------
    pd_object : pandas.Series or pandas.DataFrame
        Pandas object to be serialized with PyArrow.

    Returns
    -------
    pyarrow.lib.SerializedPyObject
        PyArrow serialized Python object.
    """

    context = pa.default_serialization_context()
    return context.serialize(pd_object)


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

    buffer = serialize_pandas_object(pd_object).to_buffer()

    return hash_func(buffer).hexdigest()