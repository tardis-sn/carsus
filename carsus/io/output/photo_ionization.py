import logging

import astropy.units as u
import pandas as pd

from carsus.io.util import get_lvl_index2id

logger = logging.getLogger(__name__)

class PhotoIonizationPreparer:
    def __init__(self, levels, levels_all, lines_all, cmfgen_reader, cmfgen_ions):
        self.levels = levels
        self.levels_all = levels_all
        self.lines_all = lines_all
        self.cmfgen_reader = cmfgen_reader
        self.cmfgen_ions = cmfgen_ions

    @property
    def cross_sections(self):
        """
        Create a DataFrame containing photoionization cross-sections.

        Returns
        -------
        pandas.DataFrame

        """

        logger.info("Ingesting photoionization cross-sections.")
        cross_sections = self.cmfgen_reader.cross_sections.reset_index()

        logger.info("Matching levels and cross sections.")
        cross_sections = cross_sections.rename(columns={"ion_charge": "ion_number"})
        cross_sections = cross_sections.set_index(["atomic_number", "ion_number"])

        cross_sections["level_index_lower"] = cross_sections["level_index"].values
        cross_sections["level_index_upper"] = cross_sections["level_index"].values
        phixs_list = [
            get_lvl_index2id(cross_sections.loc[ion], self.levels_all)
            for ion in self.cmfgen_ions
        ]

        cross_sections = pd.concat(phixs_list, sort=True)
        cross_sections = cross_sections.sort_values(
            by=["lower_level_id", "upper_level_id"]
        )
        cross_sections["level_id"] = cross_sections["lower_level_id"]

        # `x_sect_id` number starts after the last `line_id`, just a convention
        start = self.lines_all.index[-1] + 1
        cross_sections["x_sect_id"] = range(start, start + len(cross_sections))

        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")
        level_number = levels.loc[:, ["level_number"]]
        cross_sections = cross_sections.join(level_number, on="level_id")

        # Levels are already cleaned, just drop the NaN's after join
        cross_sections = cross_sections.dropna()

        cross_sections["energy"] = u.Quantity(cross_sections["energy"], "Ry").to(
            "Hz", equivalencies=u.spectral()
        )
        cross_sections["sigma"] = u.Quantity(cross_sections["sigma"], "Mbarn").to("cm2")
        cross_sections["level_number"] = cross_sections["level_number"].astype("int")
        cross_sections = cross_sections.rename(
            columns={"energy": "nu", "sigma": "x_sect"}
        )

        return cross_sections
    
    @property
    def cross_sections_prepared(self):
        """
        Prepare the DataFrame with photoionization cross-sections for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """
        cross_sections_prepared = self.cross_sections.set_index(
            ["atomic_number", "ion_number", "level_number"]
        )
        cross_sections_prepared = cross_sections_prepared[["nu", "x_sect"]]

        return cross_sections_prepared
