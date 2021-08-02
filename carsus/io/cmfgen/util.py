import gzip
import itertools

import astropy.constants as const
import astropy.units as u
import numpy as np

RYD_TO_EV = u.rydberg.to('eV')
H_IN_EV_SECONDS = const.h.to('eV s').value
HC_IN_EV_ANGSTROM = (const.h * const.c).to('eV angstrom').value


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
    meta = {k.strip('!'):None for k in keys}

    with open_cmfgen_file(fname) as f:
        for line in itertools.islice(f, start, stop):
            for k in keys:
                if k.lower() in line.lower():
                    meta[k.strip('!')] = line.split()[0]

    return meta


def get_seaton_phixs_table(threshold_energy_ryd, sigma_t, beta, s, nu_0=None, n_points=1000):
    """ Docstring """
    energy_grid = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phixs_table = np.empty((len(energy_grid), 2))

    for i, c in enumerate(energy_grid):
        energy_div_threshold = 1 + 20 * (c ** 2)

        if nu_0 is None:
            threshold_div_energy = energy_div_threshold ** -1
            cross_section = sigma_t * (beta + (1 - beta) * threshold_div_energy) * (threshold_div_energy ** s)

        else:
            threshold_energy_ev = threshold_energy_ryd * RYD_TO_EV
            energy_offset_div_threshold = energy_div_threshold + (nu_0 * 1e15 * H_IN_EV_SECONDS) / threshold_energy_ev
            threshold_div_energy_offset = energy_offset_div_threshold ** -1

            if threshold_div_energy_offset < 1.0:
                cross_section = sigma_t * (beta + (1 - beta) * (threshold_div_energy_offset)) * \
                    (threshold_div_energy_offset ** s)

            else:
                cross_section = 0.0

        phixs_table[i] = energy_div_threshold * threshold_energy_ryd, cross_section

    return phixs_table


def get_hydrogenic_n_phixs_table(hyd_gaunt_energy_grid_ryd, hyd_gaunt_factor, threshold_energy_ryd, n):
    """ Docstring """
    energy_grid = hyd_gaunt_energy_grid_ryd[n]
    phixs_table = np.empty((len(energy_grid), 2))
    scale_factor = 7.91 / threshold_energy_ryd / n

    for i, energy_ryd in enumerate(energy_grid):
        energy_div_threshold = energy_ryd / energy_grid[0]

        if energy_div_threshold > 0:
            cross_section = scale_factor * hyd_gaunt_factor[n][i] / (energy_div_threshold) ** 3
        else:
            cross_section = 0.0

        phixs_table[i][0] = energy_div_threshold * threshold_energy_ryd
        phixs_table[i][1] = cross_section

    return phixs_table


def get_hydrogenic_nl_phixs_table(hyd_phixs_energy_grid_ryd, hyd_phixs, threshold_energy_ryd, n, l_start, l_end, nu_0=None):
    """ Docstring """

    assert l_start >= 0
    assert l_end <= n - 1

    energy_grid = hyd_phixs_energy_grid_ryd[(n, l_start)]
    phixs_table = np.empty((len(energy_grid), 2))

    threshold_energy_ev = threshold_energy_ryd * RYD_TO_EV
    scale_factor = 1 / threshold_energy_ryd / (n ** 2) / ((l_end - l_start + 1) * (l_end + l_start + 1))

    for i, energy_ryd in enumerate(energy_grid):
        energy_div_threshold = energy_ryd / energy_grid[0]
        if nu_0 is None:
            U = energy_div_threshold
        else:
            E_0 = (nu_0 * 1e15 * H_IN_EV_SECONDS)
            U = threshold_energy_ev * energy_div_threshold / (E_0 + threshold_energy_ev)
        if U > 0:
            cross_section = 0.0
            for l in range(l_start, l_end + 1):
                assert np.array_equal(hyd_phixs_energy_grid_ryd[(n, l)], energy_grid)
                cross_section += (2 * l + 1) * hyd_phixs[(n, l)][i]
            cross_section = cross_section * scale_factor
        else:
            cross_section = 0.0

        phixs_table[i][0] = energy_div_threshold * threshold_energy_ryd
        phixs_table[i][1] = cross_section

    return phixs_table


def get_opproject_phixs_table(threshold_energy_ryd, a, b, c, d, e, n_points=1000):
    """
    Peach, Saraph, and Seaton (1988).
    """
    energy_grid = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phixs_table = np.empty((len(energy_grid), 2))

    for i, c in enumerate(energy_grid):

        energy_div_threshold = 1 + 20 * (c ** 2)
        u = energy_div_threshold
        x = np.log10(min(u, e))

        cross_section = 10 ** (a + x * (b + x * (c + x * d)))
        if u > e:
            cross_section *= (e / u) ** 2

        phixs_table[i] = energy_div_threshold * threshold_energy_ryd, cross_section

    return phixs_table


def get_hummer_phixs_table(threshold_energy_ryd, a, b, c, d, e, f, g, h, n_points=1000):
    """ 
    Only applies to `He`. The threshold cross sections seems ok, but energy 
    dependence could be slightly wrong. What is the `h` parameter that is 
    not used?.
    """
    energy_grid = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phixs_table = np.empty((len(energy_grid), 2))

    for i, c in enumerate(energy_grid):
        energy_div_threshold = 1 + 20 * (c ** 2)

        x = np.log10(energy_div_threshold)
        if x < e:
            cross_section = 10 ** (((d * x + c) * x + b) * x + a)

        else:
            cross_section = 10 ** (f + g * x)

        phixs_table[i] = energy_div_threshold * threshold_energy_ryd, cross_section

    return phixs_table


def get_vy95_phixs_table(threshold_energy_ryd, fit_coeff_table, n_points=1000):
    """
    Analytic FITS for partial photoionization cross sections.
    Verner, D. A. ; Yakovlev, D. G.

    Astronomy and Astrophysics Suppl., Vol. 109, p.125-133 (1995)
    """
    energy_grid = np.linspace(0.0, 1.0, n_points, endpoint=False)
    phixs_table = np.empty((len(energy_grid), 2))

    for i, c in enumerate(energy_grid):

        energy_div_threshold = 1 + 20 * (c ** 2)
        cross_section = 0.0

        for index, row in fit_coeff_table.iterrows():
            y = energy_div_threshold * row.at['E'] / row.at['E_0']
            P = row.at['P']
            Q = 5.5 + row.at['l'] - 0.5 * row.at['P']
            y_a = row.at['y(a)']
            y_w = row.at['y(w)']
            cross_section += row.at['sigma_0'] * ((y - 1) ** 2 + y_w ** 2) * (y ** -Q) * ((1 + np.sqrt(y / y_a)) ** -P)

        phixs_table[i] = energy_div_threshold * threshold_energy_ryd, cross_section

    return phixs_table


def get_leibowitz_phixs_table():
    """
    Radiative Transition Probabilities and Recombination Coefficients 
    of the Ion C IV.

    J. Quant. Spectrosc. Radiat. Transfer. Vol 12, pp. 299-306.
    """

    return