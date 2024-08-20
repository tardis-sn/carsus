import astropy.constants as const
import numpy as np
import pandas as pd
from scipy import interpolate

def create_einstein_coeff(lines):
    """
    Create Einstein coefficients columns for the `lines` DataFrame.

    Parameters
    ----------
    lines : pandas.DataFrame
        Transition lines dataframe.

    """
    einstein_coeff = (4 * np.pi**2 * const.e.gauss.value**2) / (
        const.m_e.cgs.value * const.c.cgs.value
    )

    lines["B_lu"] = (
        einstein_coeff * lines["f_lu"] / (const.h.cgs.value * lines["nu"])
    )

    lines["B_ul"] = (
        einstein_coeff * lines["f_ul"] / (const.h.cgs.value * lines["nu"])
    )

    lines["A_ul"] = (
        2
        * einstein_coeff
        * lines["nu"] ** 2
        / const.c.cgs.value**2
        * lines["f_ul"]
    )

def calculate_collisional_strength(
        row, temperatures, kb_ev, c_ul_temperature_cols
    ):
    """
    Function to calculation upsilon from Burgess & Tully 1992 (TType 1 - 4; Eq. 23 - 38).

    """

    c = row["cups"]
    x_knots = np.linspace(0, 1, len(row["btemp"]))
    y_knots = row["bscups"]
    delta_e = row["delta_e"]
    g_u = row["g_u"]

    ttype = row["ttype"]
    if ttype > 5:
        ttype -= 5

    kt = kb_ev * temperatures

    spline_tck = interpolate.splrep(x_knots, y_knots)

    if ttype == 1:
        x = 1 - np.log(c) / np.log(kt / delta_e + c)
        y_func = interpolate.splev(x, spline_tck)
        upsilon = y_func * np.log(kt / delta_e + np.exp(1))

    elif ttype == 2:
        x = (kt / delta_e) / (kt / delta_e + c)
        y_func = interpolate.splev(x, spline_tck)
        upsilon = y_func

    elif ttype == 3:
        x = (kt / delta_e) / (kt / delta_e + c)
        y_func = interpolate.splev(x, spline_tck)
        upsilon = y_func / (kt / delta_e + 1)

    elif ttype == 4:
        x = 1 - np.log(c) / np.log(kt / delta_e + c)
        y_func = interpolate.splev(x, spline_tck)
        upsilon = y_func * np.log(kt / delta_e + c)

    elif ttype == 5:
        raise ValueError("Not sure what to do with ttype=5")

    #### 1992A&A...254..436B Equation 20 & 22 #####
    collisional_ul_factor = 8.63e-6 * upsilon / (g_u * temperatures**0.5)
    return pd.Series(data=collisional_ul_factor, index=c_ul_temperature_cols)