import pytest

from numpy.testing import assert_allclose
from carsus.io.util import to_flat_dict, to_nom_val_and_std_dev

@pytest.mark.parametrize("test_input,expected",[
    ('isotopic_comp = 2.1234132(12)',
     {'isotopic_comp_nominal_value':'2.1234132', 'isotopic_comp_std_dev':'12'}),

    ('atomic_weight = 6.8083492038(23)',
     {'atomic_weight_nominal_value':'6.8083492038', 'atomic_weight_std_dev':'23'})
])
def test_to_flat_dict(test_input, expected, entry):
    tkns = entry.parse_string(test_input)
    tkns_dict = to_flat_dict(tkns)
    assert tkns_dict == expected


@pytest.mark.parametrize("test_input,expected",[
    ([4.10923, 4.11364], (4.111435, 0.002205))
])
def test_to_nom_val_and_std_dev(test_input, expected):
    mu, sigma = to_nom_val_and_std_dev(test_input)
    assert_allclose((mu, sigma), expected)