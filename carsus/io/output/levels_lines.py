import functools
import logging

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd

from carsus.util import convert_atomic_number2symbol, convert_wavelength_air2vacuum
from carsus.io.util import get_lvl_index2id, create_artificial_fully_ionized

# Wavelengths above this value are given in air
GFALL_AIR_THRESHOLD = 2000 * u.AA
MEDIUM_AIR = 1
MEDIUM_VACUUM = 0

logger = logging.getLogger(__name__)

class LevelsLinesPreparer:
    def __init__(self, ionization_energies, gfall_reader, chianti_reader, cmfgen_reader, lanl_ads_reader):
        self.ionization_energies = ionization_energies
        self.gfall_reader = gfall_reader
        self.chianti_reader = chianti_reader
        self.cmfgen_reader = cmfgen_reader
        self.lanl_ads_reader = lanl_ads_reader

    @staticmethod
    def solve_priorities(levels):
        """
        Returns a list of unique species per data source.


        Notes
        -----

        The `ds_id` field is the data source identifier.

        1 : NIST
        2 : GFALL
        3 : Knox Long's Zeta
        4 : Chianti Database
        5 : CMFGEN
        6 : LANL ADS

        """
        levels = levels.set_index(["atomic_number", "ion_number"])
        levels = levels.sort_index()  # To supress warnings

        lvl_list = []
        for ion in levels.index.unique():
            max_priority = levels.loc[ion, "priority"].max()
            lvl = levels.loc[ion][levels.loc[ion, "priority"] == max_priority]
            lvl_list.append(lvl)

        levels_uq = pd.concat(lvl_list, sort=True)
        gfall_ions = levels_uq[levels_uq["ds_id"] == 2].index.unique()
        chianti_ions = levels_uq[levels_uq["ds_id"] == 4].index.unique()
        cmfgen_ions = levels_uq[levels_uq["ds_id"] == 5].index.unique()
        lanl_ads_ions = levels_uq[levels_uq["ds_id"] == 6].index.unique()

        
        assert set(gfall_ions).intersection(set(chianti_ions)).intersection(
            set(cmfgen_ions)
        ).intersection(set(lanl_ads_ions)) == set([])

        return gfall_ions, chianti_ions, cmfgen_ions, lanl_ads_ions
    
    @staticmethod
    def _create_metastable_flags(levels, lines, levels_metastable_loggf_threshold=-3):
        """
        Returns metastable flag column for the `levels` DataFrame.

        Parameters
        ----------
        levels : pandas.DataFrame
           Energy levels dataframe.

        lines : pandas.DataFrame
           Transition lines dataframe.

        levels_metastable_loggf_threshold : int
           loggf threshold value.

        """
        # Filter lines on the loggf threshold value
        metastable_lines = lines.loc[lines["loggf"] > levels_metastable_loggf_threshold]

        # Count the remaining strong transitions
        metastable_lines_grouped = metastable_lines.groupby("upper_level_id")
        metastable_counts = metastable_lines_grouped["upper_level_id"].count()
        metastable_counts.name = "metastable_counts"

        # If there are no strong transitions for a level (the count is NaN)
        # then the metastable flag is True else (the count is a natural number)
        # the metastable flag is False
        levels = levels.join(metastable_counts)
        metastable_flags = levels["metastable_counts"].isnull()
        metastable_flags.name = "metastable"

        return metastable_flags
    
    def ingest_multiple_sources(self, attribute):
        """Takes dataframes from multiple readers and merges them

        Parameters
        ----------
        attribute : string
            The attribute to get from the readers

        Returns
        -------
        pandas.DataFrame
            Dataframe of the merged data
        """
        gfall = getattr(self.gfall_reader, attribute)
        gfall["ds_id"] = 2
        sources = [gfall]

        if self.chianti_reader is not None:
            chianti = getattr(self.chianti_reader, attribute)
            chianti["ds_id"] = 4
            sources.append(chianti)

        if self.cmfgen_reader is not None:
            cmfgen = getattr(self.cmfgen_reader, attribute)
            cmfgen["ds_id"] = 5
            sources.append(cmfgen)

        if self.lanl_ads_reader is not None:
            lanl_ads = getattr(self.lanl_ads_reader, attribute)
            lanl_ads["ds_id"] = 6
            sources.append(lanl_ads)
        
        return pd.concat(sources, sort=True)

    # replace with functools.cached_property with Python > 3.8
    @property
    @functools.lru_cache()
    def all_levels_data(self):
        """
        The resulting DataFrame contains stacked energy levels from GFALL,
        Chianti (optional), CMFGEN (optional) and NIST ground levels. Only
        one source of levels is kept based on priorities.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData._get_all_levels_data()`.

        """

        logger.info("Ingesting energy levels.")
        levels = self.ingest_multiple_sources("levels")

        levels["g"] = 2 * levels["j"] + 1
        levels["g"] = levels["g"].astype(np.int64)
        levels = levels.drop(columns=["j", "label", "method"])
        levels = levels.reset_index()
        levels = levels.rename(columns={"ion_charge": "ion_number"})
        levels = levels[
            ["atomic_number", "ion_number", "g", "energy", "ds_id", "priority"]
        ]
        levels["energy"] = (
            u.Quantity(levels["energy"], "cm-1")
            .to("eV", equivalencies=u.spectral())
            .value
        )

        # Solve priorities and set attributes for later use.
        self.gfall_ions, self.chianti_ions, self.cmfgen_ions, self.lanl_ads_ions = self.solve_priorities(
            levels
        )

        def to_string(x):
            return [f"{convert_atomic_number2symbol(ion[0])} {ion[1]}" for ion in sorted(x)]

        gfall_str = ", ".join(to_string(self.gfall_ions))
        logger.info(f"GFALL selected species: {gfall_str}.")

        if len(self.chianti_ions) > 0:
            chianti_str = ", ".join(to_string(self.chianti_ions))
            logger.info(f"Chianti selected species: {chianti_str}.")

        if len(self.cmfgen_ions) > 0:
            cmfgen_str = ", ".join(to_string(self.cmfgen_ions))
            logger.info(f"CMFGEN selected species: {cmfgen_str}.")

        if len(self.lanl_ads_ions) > 0:
            lanl_ads_str = ", ".join(to_string(self.lanl_ads_ions))
            logger.info(f"LANL ADS selected species: {lanl_ads_str}.")

        # Concatenate ground levels from NIST
        ground_levels = self.ionization_energies.get_ground_levels()
        ground_levels = ground_levels.rename(columns={"ion_charge": "ion_number"})
        ground_levels["ds_id"] = 1

        levels = pd.concat([ground_levels, levels], sort=True)
        levels["level_id"] = range(1, len(levels) + 1)
        levels = levels.set_index("level_id")

        # The following code should only remove the duplicated
        # ground levels. Other duplicated levels should be re-
        # moved at the reader stage.

        mask = (levels["energy"] == 0.0) & (
            levels[["atomic_number", "ion_number", "energy", "g"]].duplicated(
                keep="last"
            )
        )
        levels = levels[~mask]

        # Filter levels by priority
        for ion in self.chianti_ions:
            mask = (
                (levels["ds_id"] != 4)
                & (levels["atomic_number"] == ion[0])
                & (levels["ion_number"] == ion[1])
            )
            levels = levels.drop(levels[mask].index)

        
        for ion in self.cmfgen_ions:
            mask = (
                (levels["ds_id"] != 5)
                & (levels["atomic_number"] == ion[0])
                & (levels["ion_number"] == ion[1])
            )
            levels = levels.drop(levels[mask].index)

        for ion in self.lanl_ads_ions:
            mask = (
                (levels["ds_id"] != 6)
                & (levels["atomic_number"] == ion[0])
                & (levels["ion_number"] == ion[1])
            )
            levels = levels.drop(levels[mask].index)

        levels = levels[["atomic_number", "ion_number", "g", "energy", "ds_id"]]
        levels = levels.reset_index()

        return levels

    # replace with functools.cached_property with Python > 3.8
    @property
    @functools.lru_cache()
    def all_lines_data(self):
        """
        The resulting DataFrame contains stacked transition lines for
        GFALL, Chianti (optional) and CMFGEN (optional).

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData._get_all_lines_data()`.

        """

        logger.info("Ingesting transition lines.")
        lines = self.ingest_multiple_sources("lines")

        lines = lines.reset_index()
        lines = lines.rename(columns={"ion_charge": "ion_number"})
        lines["line_id"] = range(1, len(lines) + 1)

        # Filter lines by priority
        for ion in self.chianti_ions:
            mask = (
                (lines["ds_id"] != 4)
                & (lines["atomic_number"] == ion[0])
                & (lines["ion_number"] == ion[1])
            )
            lines = lines.drop(lines[mask].index)

        for ion in self.cmfgen_ions:
            mask = (
                (lines["ds_id"] != 5)
                & (lines["atomic_number"] == ion[0])
                & (lines["ion_number"] == ion[1])
            )
            lines = lines.drop(lines[mask].index)

        for ion in self.lanl_ads_ions:
            mask = (
                (lines["ds_id"] != 6)
                & (lines["atomic_number"] == ion[0])
                & (lines["ion_number"] == ion[1])
            )
            lines = lines.drop(lines[mask].index)

        lines = lines.set_index(["atomic_number", "ion_number"])
        lines = lines.sort_index()  # To supress warnings
        ions = (
            set(self.gfall_ions)
            .union(set(self.chianti_ions))
            .union((set(self.cmfgen_ions)))
            .union(set(self.lanl_ads_ions))
        )

        logger.info("Matching levels and lines.")
        lns_list = [
            get_lvl_index2id(lines.loc[ion], self.all_levels_data) for ion in ions
        ]
        lines = pd.concat(lns_list, sort=True)
        lines = lines.set_index("line_id").sort_index()

        lines["loggf"] = np.log10(lines["gf"])
        lines = lines.drop(
            columns=[
                "energy_upper",
                "j_upper",
                "energy_lower",
                "j_lower",
                "level_index_lower",
                "level_index_upper",
            ]
        )

        lines["wavelength"] = u.Quantity(lines["wavelength"], "nm").to("AA").value

        lines.loc[lines["wavelength"] <= GFALL_AIR_THRESHOLD, "medium"] = MEDIUM_VACUUM

        lines.loc[lines["wavelength"] > GFALL_AIR_THRESHOLD, "medium"] = MEDIUM_AIR

        # Chianti wavelengths are already given in vacuum
        gfall_mask = lines["ds_id"] == 2
        air_mask = lines["medium"] == MEDIUM_AIR
        lines.loc[air_mask & gfall_mask, "wavelength"] = convert_wavelength_air2vacuum(
            lines.loc[air_mask, "wavelength"]
        )

        lines = lines[
            ["lower_level_id", "upper_level_id", "wavelength", "gf", "loggf", "A_ul", "ds_id"]
        ]

        return lines
    
    @property
    def levels_prepared(self):
        """
        Prepare the DataFrame with levels for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """

        levels_prepared = self.levels.loc[
            :,
            [
                "atomic_number",
                "ion_number",
                "level_number",
                "energy",
                "g",
                "metastable",
            ],
        ].copy()

        levels_prepared = levels_prepared.set_index(
            ["atomic_number", "ion_number", "level_number"]
        )

        return levels_prepared

    @property
    def lines_prepared(self):
        """
        Prepare the DataFrame with lines for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """

        lines_prepared = self.lines.loc[
            :,
            [
                "line_id",
                "wavelength",
                "atomic_number",
                "ion_number",
                "f_ul",
                "f_lu",
                "level_number_lower",
                "level_number_upper",
                "nu",
                "B_lu",
                "B_ul",
                "A_ul",
            ],
        ].copy()

        # TODO: store units in metadata
        # wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1],
        # B_ul[cm^3 s^-2 erg^-1], B_lu[cm^3 s^-2 erg^-1],
        # A_ul[1/s].

        lines_prepared = lines_prepared.set_index(
            ["atomic_number", "ion_number", "level_number_lower", "level_number_upper"]
        )

        return lines_prepared

    def create_levels_lines(
        self, lines_loggf_threshold=-3, levels_metastable_loggf_threshold=-3
    ):
        """
        Generates the definitive `lines` and `levels` DataFrames by adding
        new columns and making some calculations.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData.create_levels_lines` method.

        """

        ionization_energies = self.ionization_energies.base.reset_index()
        ionization_energies = ionization_energies.rename(
            columns={"ion_charge": "ion_number"}
        )

        # Culling autoionization levels
        levels_w_ionization_energies = pd.merge(
            self.all_levels_data,
            ionization_energies,
            how="left",
            on=["atomic_number", "ion_number"],
        )

        mask = (
            levels_w_ionization_energies["energy"]
            < levels_w_ionization_energies["ionization_energy"]
        )

        levels = levels_w_ionization_energies[mask].copy()
        levels = levels.set_index("level_id").sort_values(
            by=["atomic_number", "ion_number"]
        )
        levels = levels.drop(columns="ionization_energy")

        # Clean lines
        lines = self.all_lines_data.join(
            pd.DataFrame(index=levels.index), on="lower_level_id", how="inner"
        ).join(pd.DataFrame(index=levels.index), on="upper_level_id", how="inner")

        # Culling lines with low gf values if needed
        if lines_loggf_threshold > -99:
            lines = lines.loc[lines["loggf"] < lines_loggf_threshold]

        # get a mask of the lines with loggf above the threshold
        high_gf_mask = lines["loggf"] > lines_loggf_threshold

        # Do not clean levels that don't exist in lines

        # Create the metastable flags for levels
        levels["metastable"] = self._create_metastable_flags(
            levels, self.all_lines_data, levels_metastable_loggf_threshold
        )

        # Create levels numbers
        levels = levels.sort_values(["atomic_number", "ion_number", "energy", "g"])

        levels["level_number"] = (
            levels.groupby(["atomic_number", "ion_number"])["energy"]
            .transform(lambda x: np.arange(len(x)))
            .values
        )

        levels["level_number"] = levels["level_number"].astype(np.int64)

        levels = levels[
            [
                "atomic_number",
                "ion_number",
                "g",
                "energy",
                "metastable",
                "level_number",
                "ds_id",
            ]
        ]

        # Join atomic_number, ion_number, level_number_lower,
        # level_number_upper on lines
        lower_levels = levels.rename(
            columns={"level_number": "level_number_lower", "g": "g_l"}
        ).loc[:, ["atomic_number", "ion_number", "level_number_lower", "g_l"]]

        upper_levels = levels.rename(
            columns={"level_number": "level_number_upper", "g": "g_u"}
        ).loc[:, ["level_number_upper", "g_u"]]

        lines = lines.join(lower_levels, on="lower_level_id").join(
            upper_levels, on="upper_level_id"
        )

        # Calculate absorption oscillator strength f_lu and emission
        # oscillator strength f_ul
        lines["f_lu"] = lines["gf"] / lines["g_l"]
        lines["f_ul"] = lines["gf"] / lines["g_u"]

        # Calculate frequency
        lines["nu"] = u.Quantity(lines["wavelength"], "AA").to("Hz", u.spectral()).value

        # Create Einstein coefficients
        create_einstein_coeff(lines, high_gf_mask)

        # Reset indexes because `level_id` cannot be an index once we
        # add artificial levels for fully ionized ions that don't have ids (-1)
        lines = lines.reset_index()
        levels = levels.reset_index()

        # Create and append artificial levels for fully ionized ions
        artificial_fully_ionized_levels = create_artificial_fully_ionized(levels)
        levels = pd.concat([levels, artificial_fully_ionized_levels], ignore_index=True)
        levels = levels.sort_values(["atomic_number", "ion_number", "level_number"])

        self.levels = levels
        self.lines = lines

def create_einstein_coeff(lines, high_gf_mask):
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

    lines.loc[high_gf_mask, "A_ul"] = (
        2
        * einstein_coeff
        * lines["nu"] ** 2
        / const.c.cgs.value**2
        * lines["f_ul"]
    )
