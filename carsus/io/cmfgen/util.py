import gzip
import itertools

def open_cmfgen_file(fname, encoding='ISO-8859-1'):
    return gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname, encoding=encoding) 

def to_float(string):
    """
    String to float, also deals with Fortran 'D' type.

    Parameters
    ----------
    string : str

    Returns
    -------
    float
    """
    try:
        value = float(string.replace('D', 'E'))

    except ValueError:

        # Typo in `MG/VIII/23oct02/phot_sm_3000`, line 23340
        if string == '1-.00':
            value = 10.00

        # Typo in `SUL/V/08jul99/phot_op.big`, lines 9255-9257
        if string == '*********':
            value = np.nan

    return value

def find_row(fname, string1, string2=None, how='AND', row_number=False):
    """
    Search for strings in plain text files and returns the matching\
    line (or row number).

    Parameters
    ----------
    fname : str
        Path to plain text file.
    string1 : str
        String to search.
    string2 : str
        Secondary string to search (default is None).
    how : {'OR', 'AND', 'AND NOT'}
        Search method: `string1` <method> `string2`
            (default is 'AND').
    row_number : bool
        If true, returns row number (default is False).

    Returns
    -------
    str or int
        Returns matching line or match row number.
    """

    if string2 is None:
        string2 = ''

    with open_cmfgen_file(fname) as f:
        n = 0
        for line in f:

            n += 1
            if how == 'OR':
                if string1 in line or string2 in line:
                    break

            if how == 'AND':
                if string1 in line and string2 in line:
                    break

            if how == 'AND NOT':
                if string1 in line and string2 not in line:
                    break

        else:
            n, line = None, None

    if row_number is True:
        return n

    return line


def parse_header(fname, keys, start=0, stop=50):
    """
    Parse header information from CMFGEN files.

    Parameters
    ----------
    fname : str
        Path to plain text file.
    keys : list of str
        Entries to search.
    start : int
        First line to search in (default is 0).
    stop : int
        Last line to search in (default is 50).

    Returns
    -------
    dict
        Dictionary containing metadata.
    """
    meta = {k.strip('!'): None for k in keys}

    with open_cmfgen_file(fname) as f:
        for line in itertools.islice(f, start, stop):
            for k in keys:
                if k.lower() in line.lower():
                    meta[k.strip('!')] = line.split()[0]

    return meta
