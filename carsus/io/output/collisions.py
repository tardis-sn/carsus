import logging
import re

import astropy.constants as const
import numpy as np
import pandas as pd

from carsus.calculations import calculate_collisional_strength
from carsus.io.util import get_lvl_index2id

logger = logging.getLogger(__name__)

class CollisionsPreparer:
    def __init__(self, chianti_reader, levels, levels_all, lines_all, chianti_ions):
        self.chianti_reader = chianti_reader
        self.levels = levels
        self.levels_all = levels_all
        self.lines_all = lines_all
        self.chianti_ions = chianti_ions
        
    def create_collisions(self, temperatures=np.arange(2000, 50000, 2000)):
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
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")

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
    
    def prepare_collisions(self, collisions_metadata, collisions):
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

        if "chianti" in collisions_metadata.dataset:
            collisions_columns = (
                collisions_index
                + ["g_ratio", "delta_e"]
                + sorted(
                    [col for col in collisions.columns if re.match("^t\d+$", col)]
                )
            )

        elif "cmfgen" in collisions_metadata.dataset:
            collisions_columns = collisions_index + list(collisions.columns)

        else:
            raise ValueError("Unknown source of collisional data")

        collisions_prepared = (
            collisions.reset_index().loc[:, collisions_columns].copy()
        )
        collisions_prepared = collisions_prepared.set_index(collisions_index)

        return collisions_prepared