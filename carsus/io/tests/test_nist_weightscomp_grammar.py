import pytest

from carsus.io.nist.weightscomp_grammar import *
from carsus.io.util import to_flat_dict
from numpy.testing import assert_almost_equal, assert_allclose


@pytest.mark.parametrize("test_input,expected",[
    ("1.00784", 1.00784),
    ("6.0151228874", 6.0151228874)
])
def test_float_(test_input, expected):
    tkns = float_.parse_string(test_input)
    assert_almost_equal(tkns[0], expected)


@pytest.mark.parametrize("test_input,expected",[
    ("16.025750(22)", [16.025750, 2.2e-5]),
    ("18.998403163(6)", [18.998403163, 6e-9])
])
def test_ufloat(test_input, expected):
    tkns = ufloat.parse_string(test_input)
    assert_almost_equal(tkns.nominal_value, expected[0])
    assert_almost_equal(tkns.std_dev, expected[1])


@pytest.mark.parametrize("test_input,expected",[
    ("29.04254(54#)", [29.04254, 54e-5, True]),
    ("27.02644(20)", [27.02644, 20e-5, False])
])
def test_ufloat_theor(test_input, expected):
    tkns = ufloat_theor.parse_string(test_input)
    assert_almost_equal(tkns.nominal_value, expected[0])
    assert_almost_equal(tkns.std_dev, expected[1])
    assert tkns.theoretical == expected[2]


@pytest.mark.parametrize("test_input,expected",[
    ("m", 'm'),
    ("g, r", 'g r')
])
def test_notes(test_input, expected):
    tkns = notes.parse_string(test_input)
    assert tkns[0] == expected


@pytest.mark.parametrize("test_input,expected",[
    ("20.1797(6)", [20.1797, 6e-4, VAL_SD]),
    ("22.98976928(2)", [22.98976928, 2e-8, VAL_SD])
])
def test_atomic_weight_value_uncertainty(test_input, expected):
    tkns = atomic_weight.parse_string(test_input)
    assert_almost_equal(tkns.nominal_value, expected[0])
    assert_almost_equal(tkns.std_dev, expected[1])
    assert tkns.type == expected[2]


@pytest.mark.parametrize("test_input,expected",[
    ("[24.304,24.307]", [[24.304,24.307], INTERVAL])
])
def test_atomic_weight_interval(test_input, expected):
    tkns = atomic_weight.parse_string(test_input)
    assert_allclose([tkns.lwr_bnd, tkns.upr_bnd],expected[0])
    assert tkns.type == expected[1]


@pytest.mark.parametrize("test_input,expected",[
    ("[226]", [226, STABLE_MASS_NUM])
])
def test_atomic_weight_stable_mass_num(test_input, expected):
    tkns = atomic_weight.parse_string(test_input)
    assert tkns.stable_mass_number == expected[0]
    assert tkns.type == expected[1]



@pytest.mark.parametrize("test_input,expected",[
    ("0.96941(156)", [0.96941, 156e-5])
])
def test_isotopic_comp(test_input, expected):
    tkns = isotopic_comp.parse_string(test_input)
    assert_almost_equal(tkns.nominal_value, expected[0])
    assert_almost_equal(tkns.std_dev, expected[1])


def test_isotopic_comp_one():
    tkns = isotopic_comp.parse_string("1")
    assert tkns.nominal_value == 1
    assert tkns.std_dev == ''


@pytest.mark.parametrize("test_input,expected",[
    ("55.00076(75#)", [55.00076, 75e-5, True]),
    ("39.963998166(60)", [39.963998166, 60e-9, False])
])
def test_atomic_mass(test_input, expected):
    tkns = atomic_mass.parse_string(test_input)
    assert_almost_equal(tkns.nominal_value, expected[0])
    assert_almost_equal(tkns.std_dev, expected[1])
    assert tkns.theoretical == expected[2]


@pytest.mark.parametrize("test_input,expected",[
    ("He", "He"),
    ("Uuo", "Uuo")
])
def test_symbol(test_input, expected):
    tkns = symbol.parse_string(test_input)
    assert tkns[0] == expected


@pytest.mark.parametrize("test_input,expected",[
    ("Atomic Number", COLUMN_NAMES_MAPPING['Atomic Number']),
    ("Standard Atomic Weight", COLUMN_NAMES_MAPPING['Standard Atomic Weight']),
    ("Notes", COLUMN_NAMES_MAPPING["Notes"])
])
def test_column_name(test_input, expected):
    tkns = column_name.parse_string(test_input)
    assert tkns[0] == expected


@pytest.mark.parametrize("test_input,expected",[
    ("""Atomic Number = 18
        Atomic Symbol = Ar
        Mass Number = 38
        Relative Atomic Mass = 37.96273211(21)
        Isotopic Composition = 0.000629(7)
        Standard Atomic Weight = 39.948(1)
        Notes = g,r)""",
     {ATOM_NUM_COL:18, SYMBOL_COL: 'Ar', MASS_NUM_COL: 38,
      AM_VAL_COL: 37.96273211, AM_SD_COL: 21e-8, AM_THEOR_COL: False,
      IC_VAL_COL: 0.000629, IC_SD_COL: 7e-6,
      AW_VAL_COL: 39.948, AW_SD_COL: 1e-3, AW_TYPE_COL: VAL_SD,
      NOTES_COL: 'g r'}),

    ("""Atomic Number = 19
        Atomic Symbol = K
        Mass Number = 54
        Relative Atomic Mass = 53.99463(64#)
        Isotopic Composition =
        Standard Atomic Weight = 39.0983(1)
        Notes =  """,
    {ATOM_NUM_COL: 19, SYMBOL_COL: 'K', MASS_NUM_COL: 54,
     AM_VAL_COL: 53.99463, AM_SD_COL: 64e-5, AM_THEOR_COL: True,
     AW_VAL_COL: 39.0983, AW_SD_COL: 1e-4, AW_TYPE_COL: VAL_SD}),

    ("""Atomic Number = 3
        Atomic Symbol = Li
        Mass Number = 5
        Relative Atomic Mass = 5.012538(54)
        Isotopic Composition =
        Standard Atomic Weight = [6.938,6.997]
        Notes = m""",
    {ATOM_NUM_COL: 3, SYMBOL_COL: 'Li', MASS_NUM_COL: 5,
     AM_VAL_COL: 5.012538, AM_SD_COL: 54e-6, AM_THEOR_COL: False,
     AW_LWR_BND_COL: 6.938, AW_UPR_BND_COL: 6.997, AW_TYPE_COL: INTERVAL,
     NOTES_COL: "m"}),

    ("""Atomic Number = 95
        Atomic Symbol = Am
        Mass Number = 230
        Relative Atomic Mass = 230.04609(14#)
        Isotopic Composition =
        Standard Atomic Weight =
        Notes =  """,
    {ATOM_NUM_COL: 95, SYMBOL_COL: 'Am', MASS_NUM_COL: 230,
     AM_VAL_COL: 230.04609, AM_SD_COL: 14e-5, AM_THEOR_COL: True}),

    ("""Atomic Number = 86
        Atomic Symbol = Rn
        Mass Number = 211
        Relative Atomic Mass = 210.9906011(73)
        Isotopic Composition =
        Standard Atomic Weight = [222]
        Notes =  """,
    {ATOM_NUM_COL: 86, SYMBOL_COL: 'Rn', MASS_NUM_COL: 211,
     AM_VAL_COL: 210.9906011, AM_SD_COL: 73e-7, AM_THEOR_COL: False,
     AW_TYPE_COL: STABLE_MASS_NUM, AW_STABLE_MASS_NUM_COL: 222})
])
def test_isotope(test_input, expected):
    tkns = isotope.parse_string(test_input)
    tkns_dict = to_flat_dict(tkns)
    for key, item in tkns_dict.items():
        if isinstance(item, float):
            assert_almost_equal(item, expected[key])
        else:
            assert item == expected[key]
