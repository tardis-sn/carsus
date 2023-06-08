"""
BNF grammar and column names for the NIST Atomic Weights and Isotopic Compostitions database
http://www.nist.gov/pml/data/comp.cfm (linearized ASCII output)

isotope  ::=  column_name eq atomic_number \
              column_name eq symbol \
              column_name eq mass_number \
              column_name eq atomic_mass \
              column_name eq [isotopic_comp] \
              column_name eq [atomic_weight] \
              column_name eq [notes]

column_name    ::= 'Atomic Number' | 'Atomic Symbol' | 'Mass Number' | 'Relative Atomic Mass' | \
                   'Isotopic Composition' | 'Standard Atomic Weight' | 'Notes'

atomic_number  ::=  decimal
symbol         ::=  letter+
atomic_mass    ::=  ufloat_theor
isotopic_comp  ::=  ufloat | "1"
atomic_weight  ::=  ufloat | ( "[" float "," float "]" ) | ( "[" mass_number "]" )
mass_number    ::=  decimal
notes          ::=  note_value ("," note_value)*

ufloat_theor   ::=  float "("  decimal ["#"] ")"
ufloat         ::=  float "("  decimal ")"
float          ::=  decimal "." digit*
decimal        ::=  digit+
digit          ::=  "0"..."9"
note_value     ::=  "g" | "m" | "r"
letter         ::=  "a"..."z" | "A"..."Z"
eq             ::=  "="

The grammar follows python BNF notation.
"""


from pyparsing import Word, Literal, Suppress, Group, Dict, Optional, delimitedList, oneOf, nums, alphas
from uncertainties import ufloat_fromstr


__all__ = ['ATOM_NUM_COL', 'SYMBOL_COL', 'MASS_NUM_COL', \
           'AM_VAL_COL', 'AM_SD_COL', 'AM_THEOR_COL', \
           'IC_VAL_COL', 'IC_SD_COL', 'AW_TYPE_COL', 'AW_VAL_COL', 'AW_SD_COL', \
           'AW_LWR_BND_COL', 'AW_UPR_BND_COL', 'AW_STABLE_MASS_NUM_COL',\
           'NOTES_COL', 'GENERAL_COLS', 'ATOM_MASS_COLS', 'ISOTOP_COMP_COLS', \
           'ATOM_WEIGHT_COLS', 'COLUMNS', 'COLUMN_NAMES_MAPPING', \
           'VAL_SD', 'INTERVAL', 'STABLE_MASS_NUM',\
           'float_', 'ufloat', 'ufloat_theor', 'notes', 'atomic_weight', \
           'isotopic_comp', 'atomic_mass', 'symbol', 'column_name', 'isotope']

GENERAL_COLS = ['atomic_number', 'symbol', 'mass_number']
ATOM_NUM_COL, SYMBOL_COL, MASS_NUM_COL = GENERAL_COLS

ATOM_MASS_COLS = ['atomic_mass_nominal_value', 'atomic_mass_std_dev', 'atomic_mass_theoretical']
AM_VAL_COL, AM_SD_COL, AM_THEOR_COL = ATOM_MASS_COLS

ISOTOP_COMP_COLS = ['isotopic_comp_nominal_value', 'isotopic_comp_std_dev']
IC_VAL_COL, IC_SD_COL = ISOTOP_COMP_COLS

ATOM_WEIGHT_COLS = ['atomic_weight_type', 'atomic_weight_nominal_value', 'atomic_weight_std_dev',
           'atomic_weight_lwr_bnd', 'atomic_weight_upr_bnd', 'atomic_weight_stable_mass_number']
AW_TYPE_COL, AW_VAL_COL, AW_SD_COL, \
AW_LWR_BND_COL, AW_UPR_BND_COL, AW_STABLE_MASS_NUM_COL = ATOM_WEIGHT_COLS

NOTES_COL = 'notes'

COLUMNS = GENERAL_COLS + ATOM_MASS_COLS + ISOTOP_COMP_COLS + ATOM_WEIGHT_COLS + [NOTES_COL]


COLUMN_NAMES_MAPPING = {
    'Atomic Number': 'atomic_number',
    'Atomic Symbol': 'symbol',
    'Mass Number': 'mass_number',
    'Relative Atomic Mass': 'atomic_mass',
    'Isotopic Composition': 'isotopic_comp',
    'Standard Atomic Weight': 'atomic_weight',
    'Notes': 'notes'
}

ATOMIC_WEIGHT_TYPES = [0, 1, 2]
VAL_SD, INTERVAL, STABLE_MASS_NUM = ATOMIC_WEIGHT_TYPES

EQ = Suppress(Literal("="))
LPAREN, RPAREN, LBRACK, RBRACK = "()[]"

# letter ::=  "a"..."z" | "A"..."Z"
# use alphas

# note_value :: = "g" | "m" | "r"
note_value = oneOf("g m r")

# digit ::= "0" ... "9"
# use nums

# decimal  ::= digit +
decimal = Word(nums)
to_int = lambda t: int(t[0])
decimal.setParseAction(to_int)

# float  ::=  digit+ "." digit*
float_ = Word(nums, f'{nums}.')
to_float = lambda t: float(t[0])
float_.setParseAction(to_float)

# ufloat ::=  float "("  decimal ")"
ufloat = Word(nums, f'{nums}.') + LPAREN + Word(nums) + RPAREN


def to_nom_value_and_std_dev(tokens):
    u = ufloat_fromstr("".join(tokens))
    tokens['nominal_value'] = u.nominal_value
    tokens['std_dev'] = u.std_dev
    return tokens

ufloat.setParseAction(to_nom_value_and_std_dev)

# ufloat_theor ::=  float "("  decimal ["#"] ")"
ufloat_theor = (
    Word(nums, f'{nums}.')
    + LPAREN
    + Word(nums)
    + Optional(Literal("#"))
    + RPAREN
)


def to_nom_val_and_std_dev_theor(tokens):
    if "#" in tokens.asList():
        tokens["theoretical"] = True
        del tokens[3]
    else:
        tokens["theoretical"] = False
    to_nom_value_and_std_dev(tokens)

ufloat_theor.setParseAction(to_nom_val_and_std_dev_theor)

# notes ::=  note_value ("," note_value)*
notes = delimitedList(note_value).setParseAction(lambda t: " ".join(t))

# mass_number  ::=  decimal
mass_number = decimal

# atomic_weight ::= ufloat | ( "[" float "," float "]" ) | ( "[" mass_number "]" )
atomic_weight = ufloat | \
            ( LBRACK + float_.setResultsName("lwr_bnd") + Suppress(",") + float_.setResultsName("upr_bnd") + RBRACK ) | \
            ( LBRACK + mass_number.setResultsName("stable_mass_number") + RBRACK )


def set_atomic_weight_type(tokens):
    if tokens.nominal_value != '':
        tokens['type'] = VAL_SD
    elif tokens.lwr_bnd != '':
        tokens['type'] = INTERVAL
    else:
        tokens['type'] = STABLE_MASS_NUM
    return tokens

atomic_weight.setParseAction( set_atomic_weight_type)


# isotopic_comp  ::=  ufloat | "1"
isotopic_comp = ufloat | Literal("1").setParseAction(to_int).setResultsName('nominal_value')

# atomic_mass  ::=  ufloat_theor
atomic_mass = ufloat_theor

# symbol  ::=  letter+
symbol = Word(alphas)

# atomic_number  ::=  decimal
atomic_number = decimal

#  column_name = 'Atomic Number' | 'Atomic Symbol' | 'Mass Number' | 'Relative Atomic Mass' | \
#                'Isotopic Composition' | 'Standard Atomic Weight' | 'Notes'
column_name = oneOf(COLUMN_NAMES_MAPPING.keys()).setParseAction(lambda t: COLUMN_NAMES_MAPPING[t[0]])

#  isotope  ::=  column_name eq atomic_number \
#                column_name eq symbol \
#                column_name eq mass_number \
#                column_name eq atomic_mass \
#                column_name eq [isotopic_comp] \
#                column_name eq [atomic_weight] \
#                column_name eq [notes]
isotope = Dict( Group(column_name + EQ + atomic_number) ) + \
          Dict( Group(column_name + EQ + symbol )) + \
          Dict( Group(column_name + EQ + mass_number) ) + \
          Dict( Group(column_name + EQ + atomic_mass) ) + \
          Dict( Group(column_name + EQ + Optional(isotopic_comp)) ) + \
          Dict( Group(column_name + EQ + Optional(atomic_weight)) ) + \
          Dict( Group(column_name + EQ + Optional(notes)) )


def remove_empty_keys(tokens):
    for key, item in list(tokens.items()):
        if item == '':
            del tokens[key]
    return tokens

isotope.setParseAction(remove_empty_keys)
