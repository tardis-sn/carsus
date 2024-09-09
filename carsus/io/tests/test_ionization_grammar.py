import pytest

from numpy.testing import assert_almost_equal
from carsus.io.nist.ionization_grammar import *


@pytest.mark.parametrize("test_input,exp_j",[
    ("0", 0.0),
    ("2", 2.0),
    ("3/2", 1.5)
])
def test_j(test_input, exp_j):
    tkns = J.parse_string(test_input)
    assert_almost_equal(tkns[0], exp_j)


@pytest.mark.parametrize("test_input, exp_mult, exp_l",[
    ("1S", 1, "S"),
    ("2P", 2, "P"),
    ("1S", 1, "S")
])
def test_ls_term(test_input, exp_mult, exp_l):
    tkns = ls_term.parse_string(test_input)
    assert tkns["mult"] == exp_mult
    assert tkns["L"] == exp_l


@pytest.mark.parametrize("test_input, exp_first_j, exp_second_j",[
    ("(0,1/2)", 0.0, 0.5),
    ("(3/2,1)", 1.5, 1.0),
    ("(0,2)", 0.0, 2.0)
])
def test_jj_term(test_input, exp_first_j, exp_second_j):
    tkns = jj_term.parse_string(test_input)
    assert_almost_equal(tkns["first_J"], exp_first_j)
    assert_almost_equal(tkns["second_J"], exp_second_j)


@pytest.mark.parametrize("test_input, exp_mult, exp_l, exp_parity, exp_j",[
    ("1S0", 1, "S", 0, 0.0),
    ("2P<1/2>", 2, "P", 0, 0.5),
    ("1S*<4>", 1, "S", 1, 4.0)
])
def test_level_w_ls_term(test_input, exp_mult, exp_l, exp_parity, exp_j):
    tkns = level.parse_string(test_input)
    assert tkns["ls_term"]["mult"] == exp_mult
    assert tkns["ls_term"]["L"] == exp_l
    assert tkns["parity"] == exp_parity
    assert_almost_equal(tkns["J"][0], exp_j)  # This assertion fails because tkns["J"] is a list and exp_j and integer. e.g. [2] == 2
                                           # Same thing on lines 62 and 73.

@pytest.mark.parametrize("test_input, exp_first_j, exp_second_j, exp_parity, exp_j",[
    ("(0,1/2)0", 0.0, 0.5, 0, 0.0),
    ("(3/2,5/2)<1/2>", 1.5, 2.5, 0, 0.5),
    ("(1/2, 2)*<2>", 0.5, 2.0, 1, 2.0)
])
def test_level_w_jj_term(test_input, exp_first_j, exp_second_j, exp_parity, exp_j):
    tkns = level.parse_string(test_input)
    assert tkns["jj_term"]["first_J"] == exp_first_j
    assert tkns["jj_term"]["second_J"] == exp_second_j
    assert tkns["parity"] == exp_parity
    assert_almost_equal(tkns["J"][0], exp_j)


@pytest.mark.parametrize("test_input, exp_parity, exp_j",[
    ("0", 0, 0.0),
    ("<1/2>", 0,  0.5),
    ("*<2>", 1, 2.0)
])
def test_level_wo_term(test_input, exp_parity, exp_j):
    tkns = level.parse_string(test_input)
    assert tkns["parity"] == exp_parity
    assert_almost_equal(tkns["J"][0], exp_j)


@pytest.mark.parametrize("test_input, exp_parity",[
    ("", 0),
    ("*", 1)
])
def test_level_wo_term_and_j(test_input, exp_parity):
    tkns = level.parse_string(test_input)
    assert tkns["parity"] == exp_parity