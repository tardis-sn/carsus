import logging
import re

import astropy.constants as const
import numpy as np
import pandas as pd
from scipy import interpolate

from carsus.io.util import get_lvl_index2id, exclude_artificial_levels

logger = logging.getLogger(__name__)

class CollisionsPreparer:
    def __init__(self, reader):
        collisions = reader.collisions.copy()
        collisions.index = collisions.index.rename(
            [
                "atomic_number",
                "ion_number",
                "level_number_lower",
                "level_number_upper",
            ]
        )

        self.collisions = collisions
        self.collisions_metadata = reader.collisional_metadata
    
    def prepare_collisions(self):
        """
        Prepare the DataFrame with electron collisions for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """

        collisions_index = [
            "atomic_number",
            "ion_number",
            "level_number_lower",
            "level_number_upper",
        ]

        if "chianti" in self.collisions_metadata.dataset:
            collisions_columns = (
                collisions_index
                + ["g_ratio", "delta_e"]
                + sorted(
                    [col for col in self.collisions.columns if re.match("^t\d+$", col)]
                )
            )

        elif "cmfgen" in self.collisions_metadata.dataset:
            collisions_columns = collisions_index + list(self.collisions.columns)

        else:
            raise ValueError("Unknown source of collisional data")

        collisions_prepared = (
            self.collisions.reset_index().loc[:, collisions_columns].copy()
        )
        self.collisions_prepared = collisions_prepared.set_index(collisions_index)
    

class ChiantiCollisionsPreparer(CollisionsPreparer):
    def __init__(
            self, 
            chianti_reader, 
            levels, 
            levels_all, 
            lines_all, 
            chianti_ions, 
            collisions_param = {"temperatures": np.arange(2000, 50000, 2000)}
            ):
        self.chianti_reader = chianti_reader
        self.levels = levels
        self.levels_all = levels_all
        self.lines_all = lines_all
        self.chianti_ions = chianti_ions

        self.collisions = self.create_chianti_collisions(**collisions_param)
        self.collisions_metadata = pd.Series(
            {
                "temperatures": collisions_param["temperatures"],
                "dataset": ["chianti"],
                "info": None,
            }
        )
        
    def create_chianti_collisions(self, temperatures=np.arange(2000, 50000, 2000)):
        """
        Generates the definitive `collisions` DataFrame by adding new columns
        and making some calculations.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData.create_collisions` method.

        """

        logger.info("Ingesting collisional strengths.")
        ch_collisions = self.chianti_reader.collisions
        ch_collisions["ds_id"] = 4

        # Not really needed because we have only one source of collisions
        collisions = pd.concat([ch_collisions], sort=True)
        ions = self.chianti_ions

        collisions = collisions.reset_index()
        collisions = collisions.rename(columns={"ion_charge": "ion_number"})
        collisions = collisions.set_index(["atomic_number", "ion_number"])

        logger.info("Matching collisions and levels.")
        col_list = [
            get_lvl_index2id(collisions.loc[ion], self.levels_all) for ion in ions
        ]
        collisions = pd.concat(col_list, sort=True)
        collisions = collisions.sort_values(by=["lower_level_id", "upper_level_id"])

        # `e_col_id` number starts after the last line id
        start = self.lines_all.index[-1] + 1
        collisions["e_col_id"] = range(start, start + len(collisions))

        # Exclude artificially created levels from levels
        levels = exclude_artificial_levels(self.levels)

        # Join atomic_number, ion_number, level_number_lower, level_number_upper
        collisions = collisions.set_index(["atomic_number", "ion_number"])
        lower_levels = levels.rename(
            columns={
                "level_number": "level_number_lower",
                "g": "g_l",
                "energy": "energy_lower",
            }
        ).loc[
            :,
            [
                "atomic_number",
                "ion_number",
                "level_number_lower",
                "g_l",
                "energy_lower",
            ],
        ]

        upper_levels = levels.rename(
            columns={
                "level_number": "level_number_upper",
                "g": "g_u",
                "energy": "energy_upper",
            }
        ).loc[:, ["level_number_upper", "g_u", "energy_upper"]]

        collisions = collisions.join(lower_levels, on="lower_level_id").join(
            upper_levels, on="upper_level_id"
        )

        # Calculate delta_e
        kb_ev = const.k_B.cgs.to("eV / K").value
        collisions["delta_e"] = (
            collisions["energy_upper"] - collisions["energy_lower"]
        ) / kb_ev

        # Calculate g_ratio
        collisions["g_ratio"] = collisions["g_l"] / collisions["g_u"]

        # Derive columns for collisional strengths
        c_ul_temperature_cols = ["t{:06d}".format(t) for t in temperatures]

        collisions = collisions.rename(
            columns={"temperatures": "btemp", "collision_strengths": "bscups"}
        )

        collisions = collisions[
            [
                "e_col_id",
                "lower_level_id",
                "upper_level_id",
                "ds_id",
                "btemp",
                "bscups",
                "ttype",
                "cups",
                "gf",
                "atomic_number",
                "ion_number",
                "level_number_lower",
                "g_l",
                "energy_lower",
                "level_number_upper",
                "g_u",
                "energy_upper",
                "delta_e",
                "g_ratio",
            ]
        ]

        collisional_ul_factors = collisions.apply(
            calculate_collisional_strength,
            axis=1,
            args=(temperatures, kb_ev, c_ul_temperature_cols),
        )

        collisions = pd.concat([collisions, collisional_ul_factors], axis=1)
        collisions = collisions.set_index("e_col_id")

        return collisions

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