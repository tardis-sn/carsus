import pytest
import pandas as pd

from pandas.testing import assert_frame_equal
from carsus.io.nist.weightscomp_grammar import AW_SD_COL, AW_VAL_COL


@pytest.mark.parametrize("test_input,expected",[
    ("""atomic_weight = 6.8083492038(23)
         atomic_weight = 8.8239833(11)
         atomic_weight = 2.19802(3)""",
     [{AW_VAL_COL: "6.8083492038", AW_SD_COL: '23'},
      {AW_VAL_COL: "8.8239833", AW_SD_COL: '11'},
      {AW_VAL_COL: "2.19802", AW_SD_COL: '3'}])
])
def test_pyparser_load(test_input, expected, aw_pyparser):
    aw_pyparser.load(test_input)
    assert_frame_equal(aw_pyparser.base,
        pd.DataFrame(data=expected, columns=[AW_VAL_COL, AW_SD_COL]), check_names=False)


def test_pyparser_callable(aw_pyparser):
    aw_pyparser(input_data="atomic_weight = 6.8083492038(23)")
    assert aw_pyparser.base.loc[0, AW_VAL_COL] == "6.8083492038"
    assert aw_pyparser.base.loc[0, AW_SD_COL] == "23"