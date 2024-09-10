"""
BNF grammar for parsing spectra

element ::= 'H' .. 'Uuh'
element_range ::= element + '-' + element

atom_entry = element | element_range
selected_atoms = atom_entry + [',' + atom_entry]*

ion_number ::= decimal
ion_number_range ::= ion_number - ion_number
ion_numbers = ion_number | ion_number_range + [',' + ion_number | ion_number_range]

species_entry =  selected_atoms + [ion_numbers]
selected_species = species_entry + [';' + species_entry]*
"""

from carsus.util.helpers import convert_symbol2atomic_number
from pyparsing import Literal, Suppress, delimitedList,\
    Word, alphas, nums, Optional, Group

hyphen = Suppress(Literal('-'))

element = Word(alphas)

def parse_element(tokens):
    symbol = tokens[0]
    symbol = symbol[:1].upper() + symbol[1:].lower()
    try:
        atomic_number = convert_symbol2atomic_number(symbol)
    except KeyError:
        raise ValueError("Unrecognized atomic symbol {}".format(symbol))

    return atomic_number

element.setParseAction(parse_element)

element_range = element + hyphen + element
element_range.setParseAction(lambda x: list(range(x[0], x[1] + 1)))

atom_entry = element ^ element_range
selected_atoms = delimitedList(atom_entry)
selected_atoms.setParseAction(lambda x: sorted(set(x)))


ion_number = Word(nums)
ion_number.setParseAction(lambda x: int(x[0]))

ion_number_range = ion_number + hyphen + ion_number
ion_number_range.setParseAction(lambda x: list(range(x[0], x[1] + 1)))

ion_numbers = delimitedList(ion_number ^ ion_number_range)
ion_numbers.setParseAction(lambda x: sorted(set(x)))

species_entry = Group(selected_atoms).setResultsName('atomic_numbers') + \
                Group(Optional(ion_numbers)).setResultsName('ion_numbers')


def parse_species_entry(tokens):
    species = list()

    if tokens['ion_numbers']:
        species = [(atomic_number, ion_number)
                   for atomic_number in tokens['atomic_numbers']
                   for ion_number in tokens['ion_numbers']
                   if atomic_number > ion_number]
    else:
        species = [(atomic_number, ion_number)
                   for atomic_number in tokens['atomic_numbers']
                   for ion_number in range(atomic_number)]

    return species

species_entry.setParseAction(parse_species_entry)

selected_species = delimitedList(species_entry, delim=';')
selected_species.setParseAction(lambda x: sorted(set(x)))


def parse_selected_atoms(atoms):
    """
    Parse the sting specifying selected atoms to the list of atomic numbers.

    Parameters
    ----------
    atoms: str
        Sting that specifies selected atoms. It should consist of comma-separated entries
        that are either single atoms (e.g. "H") or ranges (indicated by using a hyphen between, e.g "H-Zn").
        Element symbols need **not** to be capitalized.

    Returns
    -------
    list of int
        List of atomic numbers

    Examples
    --------

    >>> parse_selected_atoms("H")
    [1]

    >>> parse_selected_atoms("H, Li-N")
    [1, 3, 4, 5, 6, 7]

    >>> parse_selected_atoms("H, Li-N, Si, S")
    [1, 3, 4, 5, 6, 7, 14, 16]

    """
    return selected_atoms.parseString(atoms).asList()


def parse_selected_species(species):
    """
    Parse the sting specifying selected species to the list of tuples in the
    form (atomic_number, ion_charge).

    Parameters
    ----------
    species: str
        Sting that specifies selected species. It should consist of semicolon-separated entries.
        The entries can be just element symbols or ranges of elements symbols. In this case
        all ions of the elements are selected. Also, ion numbers (starting from 0)
        can be specified - either as numbers or ranges. In this case only ions that have
        the corresponding ionization stage are selected.

    Returns
    -------
    list of tuples (atomic_number, ion_charge)
        List of selected ions

    Examples
    --------

    >>> parse_selected_species('H')
    [(1, 0)]

    >>> parse_selected_species('H-Li')
    [(1, 0), (2, 0), (2, 1), (3, 0), (3, 1), (3, 2)]

    >>> parse_selected_species('h-li 0')
    [(1, 0), (2, 0), (3, 0)]

    >>> parse_selected_species('b 3-5')
    [(5, 3), (5, 4)]

    >>> parse_selected_species('Li 3; B-O 4-5')
    [(5, 4), (6, 4), (6, 5), (7, 4), (7, 5), (8, 4), (8, 5)]

    """
    return selected_species.parseString(species).asList()
