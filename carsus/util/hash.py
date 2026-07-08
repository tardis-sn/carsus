import hashlib
import json

def serialize_pandas_object(pd_object):
    """Serialize Pandas objects in a deterministic format.

    Parameters
    ----------
    pd_object : pandas.Series or pandas.DataFrame
        Pandas object to be serialized.

    Returns
    -------
    bytes
        Serialized pandas object.
    """
    if hasattr(pd_object, "columns"):
        dtypes = [str(dtype) for dtype in pd_object.dtypes]
    else:
        dtypes = [str(pd_object.dtype)]

    payload = {
        "type": type(pd_object).__name__,
        "index_names": list(pd_object.index.names),
        "dtypes": dtypes,
        "data": pd_object.to_json(
            orient="split",
            date_format="iso",
            double_precision=15,
            default_handler=str,
        ),
    }

    if hasattr(pd_object, "columns"):
        payload["column_names"] = list(pd_object.columns.names)
    else:
        payload["name"] = pd_object.name

    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


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
