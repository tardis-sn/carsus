import pytest
import pandas as pd
from carsus.util import hash_pandas_object


@pytest.mark.parametrize(
    "values, md5",
    [
        ([(0, 1), (1, 2), (2, 3), (3, 4)], "443a502045"),
        (["apple", "banana", "orange"], "f90e8fb058"),
    ],
)
def test_hash_pd(values, md5):
    assert hash_pandas_object(pd.DataFrame(values))[:10] == md5
