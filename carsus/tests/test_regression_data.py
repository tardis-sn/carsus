import pandas as pd
import numpy as np

def test_np_regression(regression_data):
  numpy_array = np.array([1, 2, 3, 4, 5])
  expected = regression_data.sync_ndarray(numpy_array)
  np.testing.assert_allclose(expected, numpy_array)
