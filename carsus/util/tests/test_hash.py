import pytest
import hashlib
import pandas as pd
from carsus.util import serialize_pandas_object, hash_pandas_object


@pytest.mark.parametrize(
    "values, md5, sha1",
    [
        ([(0, 1), (1, 2), (2, 3), (3, 4)], "a703629383", "d733a2c2dc"),
        (["apple", "banana", "orange"], "24e45baf79", "dd9a9d4b88"),
    ],
)
def test_hash_pd(values, md5, sha1):
    assert hash_pandas_object(pd.DataFrame(values))[:10] == md5
    assert hash_pandas_object(pd.DataFrame(values), algorithm="sha1")[:10] == sha1

