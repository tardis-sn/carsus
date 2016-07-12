"""
BNF grammar for parsing Ground Levels

level  ::= [ ls_term | jj_term ] + (J | <J>)

ls_term  ::= mult + L + ["*"]
jj_term  ::= "(" + J + "," + J + ")"

mult   ::= decimal
L      ::= "A" .. "Z"
J      ::= decimal | (decimal + "/" + decimal)

decimal  ::= digit+
digit    ::= "0"..."9"
"""

from pyparsing import Word, Literal, Suppress, Group, Dict, Optional, delimitedList, oneOf, nums, srange


LPAREN, RPAREN = map(Suppress, "()")

# digit ::= "0" ... "9"
# use nums

# decimal  ::= digit +
decimal = Word(nums)
to_int = lambda t: int(t[0])
decimal.setParseAction(to_int)

# J  ::= decimal | (decimal + "/" + decimal)
J = decimal ^ decimal + Suppress("/") + decimal


def parse_j(tokens):
    if len(tokens) == 1:
        return float(tokens[0])
    else:
        return float(tokens[0])/tokens[1]

J.setParseAction(parse_j)

# mult ::= decimal
mult = decimal

# L  ::= "A" .. "Z"
L = Word(srange("[A-Z]"))

# ls_term  ::= mult + L + ["*"]
ls_term = mult.setResultsName("mult") + L.setResultsName("L") + Optional("*")


def parse_ls_term(tokens):
    if "*" in tokens.asList():
        tokens["parity"] = 1
    else:
        tokens["parity"] = 0

ls_term.setParseAction(parse_ls_term)

# jj_term  ::= "(" + J + "," + J + ")"
jj_term = Literal("(") + J.setResultsName("first_J") + "," + \
          J.setResultsName("second_J") + Literal(")")

# level  ::= [ ls_term | jj_term ] + (J | <J>)
level = Optional(
            Group(ls_term).setResultsName("ls_term") |
            Group(jj_term).setResultsName("jj_term")) + \
        (J | Suppress("<") + J + Suppress(">")).setResultsName("J")
