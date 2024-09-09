import pytest
import hashlib
import pandas as pd
from carsus.util import hash_pandas_object


@pytest.mark.parametrize(
    "values, md5",
    [
        ([(0, 1), (1, 2), (2, 3), (3, 4)], "9a6b127b25"),
        (["apple", "banana", "orange"], "924a349b83"),
    ],
)
def test_hash_pd(values, md5):
    assert hash_pandas_object(pd.DataFrame(values))[:10] == md5
