import pytest

from carsus.util.selected import element, element_range, \
    selected_atoms, parse_selected_atoms, species_entry, \
    parse_selected_species


@pytest.mark.parametrize("test_input, exp_atomic_number",[
    ("H", 1),
    ("Zn", 30),
    ("Uuo", 118),
    ("h", 1)
])
def test_element(test_input, exp_atomic_number):
    tokens = element.parse_string(test_input)
    assert tokens[0] == exp_atomic_number


@pytest.mark.parametrize("test_input, exp_atomic_numbers",[
    ("H-Li", [1, 2, 3]),
    ("H-Zn", list(range(1,31))),
    ("si-s", [14, 15, 16])
])
def test_element_range(test_input, exp_atomic_numbers):
    tokens = element_range.parse_string(test_input)
    assert tokens.asList() == exp_atomic_numbers


@pytest.mark.parametrize("test_input, exp_atomic_numbers",[
    ("H", [1,]),
    ("H-Zn", list(range(1,31))),
    ("h, si-s", [1, 14, 15, 16]),
    ('he, h-li', [1, 2, 3])
])
def test_selected_atoms(test_input, exp_atomic_numbers):
    tokens = selected_atoms.parse_string(test_input)
    assert tokens.asList() == exp_atomic_numbers


@pytest.mark.parametrize("selected_atoms, expected_list", [
    ("H", [1]),
    ("H-Li", [1, 2, 3]),
    ("H, Be-B", [1, 4, 5]),
    ("h, be-b", [1, 4, 5]),
    (" h ,  be - b ", [1, 4, 5])
])
def test_parse_selected_atoms(selected_atoms, expected_list):
    assert parse_selected_atoms(selected_atoms) == expected_list


@pytest.mark.parametrize("invalid_selected_atoms", [
    "Foo", "H-Foo", "H, Al-Foo"
])
def test_parse_selected_atoms_raises_invalid(invalid_selected_atoms):
    with pytest.raises(ValueError):
        parse_selected_atoms(invalid_selected_atoms)


@pytest.mark.parametrize("test_input, expected_list", [
    ("H 0", [(1, 0)]),
    ("H, Li 0", [(1, 0), (3, 0)]),
    ("H-Li 0-1", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]),
    ("H-C 0, 4", [(1, 0), (2, 0), (3, 0), (4, 0),
                  (5, 0), (5, 4), (6, 0), (6, 4)]),
    ('H-Li', [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
])
def test_parse_species_entry(test_input, expected_list):
    tokens = species_entry.parse_string(test_input)
    assert tokens.asList() == expected_list


@pytest.mark.parametrize("test_species, expected_list", [
    ("H 0; li 0", [(1, 0), (3, 0)]),
    ("H, Li 0", [(1, 0), (3, 0)]),
    ("li 0; h", [(1, 0), (3, 0)]),
    ("h 0; h-li 0", [(1, 0), (2, 0), (3, 0)]),
    ("H-Li 0-1", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1)]),
    ("H-C 0, 4", [(1, 0), (2, 0), (3, 0), (4, 0),
                  (5, 0), (5, 4), (6, 0), (6, 4)]),
    ('H-Li', [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]),
    ('fe 12; ni-zn 23-25', [(26, 12), (28, 23), (28, 24), (28, 25),
                            (29, 23), (29, 24), (29, 25),
                            (30, 23), (30, 24), (30, 25)])
])
def test_parse_selected_species(test_species, expected_list):
    assert parse_selected_species(test_species) == expected_list
