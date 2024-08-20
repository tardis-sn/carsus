import copy
import logging
import re

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd

from carsus.io.output.macro_atom import MacroAtomPreparer
from carsus.io.output.levels_lines import LevelsLinesPreparer
from carsus.util import (
    hash_pandas_object,
    serialize_pandas_object,
)
from carsus.calculations import calculate_collisional_strength



logger = logging.getLogger(__name__)


class TARDISAtomData:
    """
    Attributes
    ----------

    levels : pandas.DataFrame
    lines : pandas.DataFrame
    collisions : pandas.DataFrame
    macro_atom : pandas.DataFrame
    macro_atom_references : pandas.DataFrame
    levels_prepared : pandas.DataFrame
    lines_prepared : pandas.DataFrame
    collisions_prepared: pandas.DataFrame
    macro_atom_prepared : pandas.DataFrame
    macro_atom_references_prepared : pandas.DataFrame

    """

    def __init__(
        self,
        atomic_weights,
        ionization_energies,
        gfall_reader,
        zeta_data,
        chianti_reader=None,
        cmfgen_reader=None,
        nndc_reader=None,
        vald_reader=None,
        barklem_2016_data=None,
        levels_lines_param={
            "levels_metastable_loggf_threshold": -3,
            "lines_loggf_threshold": -3,
        },
        collisions_param={"temperatures": np.arange(2000, 50000, 2000)},
    ):
        self.atomic_weights = atomic_weights

        if (cmfgen_reader is not None) and hasattr(
            cmfgen_reader, "ionization_energies"
        ):
            combined_ionization_energies = copy.deepcopy(ionization_energies)
            combined_ionization_energies.base = (
                cmfgen_reader.ionization_energies.combine_first(
                    ionization_energies.base
                )
            )
            self.ionization_energies = combined_ionization_energies
        else:
            self.ionization_energies = ionization_energies

        self.gfall_reader = gfall_reader
        self.zeta_data = zeta_data
        self.chianti_reader = chianti_reader
        self.cmfgen_reader = cmfgen_reader
        self.nndc_reader = nndc_reader
        self.vald_reader = vald_reader
        self.barklem_2016_data = barklem_2016_data
        self.levels_lines_param = levels_lines_param
        self.collisions_param = collisions_param

        self.levels_lines_preparer = LevelsLinesPreparer(self.gfall_reader, self.chianti_reader, self.cmfgen_reader)
        self.levels_all = self.levels_lines_preparer.all_levels_data
        self.lines_all = self.levels_lines_preparer.all_lines_data
        self.levels_lines_preparer.create_levels_lines(**levels_lines_param)
        self.levels, self.lines = self.levels_lines_preparer.levels, self.levels_lines_preparer.lines
        self.macro_atom_preparer = MacroAtomPreparer(self.levels, self.lines)
        self.macro_atom_preparer.create_macro_atom()
        self.macro_atom_preparer.create_macro_atom_references()

        if ((cmfgen_reader is not None) and hasattr(cmfgen_reader, "collisions")) and (
            (chianti_reader is not None) and hasattr(chianti_reader, "collisions")
        ):
            raise ValueError(
                "Both Chianti and CMFGEN readers contain the collisions dataframe. "
                "Please set collisions=True in one or the other but not both."
            )
        else:
            if hasattr(cmfgen_reader, "collisions"):
                collisions = cmfgen_reader.collisions.copy()
                collisions.index = collisions.index.rename(
                    [
                        "atomic_number",
                        "ion_number",
                        "level_number_lower",
                        "level_number_upper",
                    ]
                )

                self.collisions = collisions
                self.collisions_metadata = cmfgen_reader.collisional_metadata

            elif hasattr(chianti_reader, "collisions"):
                self.collisions = self.create_collisions(**collisions_param)
                self.collisions_metadata = pd.Series(
                    {
                        "temperatures": collisions_param["temperatures"],
                        "dataset": ["chianti"],
                        "info": None,
                    }
                )
            else:
                logger.warning("No source of collisions was selected.")

        if (cmfgen_reader is not None) and hasattr(cmfgen_reader, "cross_sections"):
            self.cross_sections = self.create_cross_sections()

        logger.info("Finished.")

    @staticmethod
    def get_lvl_index2id(df, levels_all):
        """
        Matches level indexes with level IDs for a given DataFrame.

        """
        # TODO: re-write this method without a for loop
        ion = df.index.unique()
        lvl_index2id = levels_all.set_index(["atomic_number", "ion_number"]).loc[ion]
        lvl_index2id = lvl_index2id.reset_index()

        lower_level_id = []
        upper_level_id = []

        df = df.reset_index()
        for row in df.itertuples():
            llid = row.level_index_lower
            ulid = row.level_index_upper

            upper = lvl_index2id.at[ulid, "level_id"]
            lower = lvl_index2id.at[llid, "level_id"]

            lower_level_id.append(lower)
            upper_level_id.append(upper)

        df["lower_level_id"] = pd.Series(lower_level_id)
        df["upper_level_id"] = pd.Series(upper_level_id)

        return df

    @staticmethod
    def _create_artificial_fully_ionized(levels):
        """
        Returns a DataFrame with fully ionized levels.

        """
        fully_ionized_levels = []

        for atomic_number, _ in levels.groupby("atomic_number"):
            fully_ionized_levels.append(
                (-1, atomic_number, atomic_number, 0, 0.0, 1, True)
            )

        levels_columns = [
            "level_id",
            "atomic_number",
            "ion_number",
            "level_number",
            "energy",
            "g",
            "metastable",
        ]

        fully_ionized_levels_dtypes = [
            (key, levels.dtypes[key]) for key in levels_columns
        ]

        fully_ionized_levels = np.array(
            fully_ionized_levels, dtype=fully_ionized_levels_dtypes
        )

        return pd.DataFrame(data=fully_ionized_levels)

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
            self.get_lvl_index2id(collisions.loc[ion], self.levels_all) for ion in ions
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

    def create_cross_sections(self):
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
            self.get_lvl_index2id(cross_sections.loc[ion], self.levels_all)
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
    def ionization_energies_prepared(self):
        """
        Prepare the DataFrame with ionization energies for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """
        ionization_energies_prepared = self.ionization_energies.base.copy()
        ionization_energies_prepared = ionization_energies_prepared.reset_index()
        ionization_energies_prepared["ion_charge"] += 1
        ionization_energies_prepared = ionization_energies_prepared.rename(
            columns={"ion_charge": "ion_number"}
        )
        ionization_energies_prepared = ionization_energies_prepared.set_index(
            ["atomic_number", "ion_number"]
        )

        return ionization_energies_prepared.squeeze()

    @property
    def collisions_prepared(self):
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
        collisions_prepared = collisions_prepared.set_index(collisions_index)

        return collisions_prepared

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
    
    @property 
    def macro_atom(self):
        return self.macro_atom_preparer.macro_atom
    
    @property 
    def macro_atom_references(self):
        return self.macro_atom_preparer.macro_atom_references

    def to_hdf(self, fname):
        """
        Dump `prepared` attributes into an HDF5 file.

        Parameters
        ----------
        fname : path
           Path to the HDF5 output file.

        """
        import hashlib
        import platform
        import uuid
        from datetime import datetime

        import pytz

        from carsus import FORMAT_VERSION

        with pd.HDFStore(fname, "w") as f:
            f.put("/atom_data", self.atomic_weights.base)
            f.put("/ionization_data", self.ionization_energies_prepared)
            f.put("/zeta_data", self.zeta_data.base)
            f.put("/levels_data", self.levels_lines_preparer.levels_prepared)
            f.put("/lines_data", self.levels_lines_preparer.lines_prepared)
            f.put("/macro_atom_data", self.macro_atom_preparer.macro_atom_prepared)
            f.put("/macro_atom_references", self.macro_atom_preparer.macro_atom_references_prepared)

            if hasattr(self.nndc_reader, "decay_data"):
                f.put("/nuclear_decay_rad", self.nndc_reader.decay_data)

            if hasattr(self.vald_reader, "linelist_atoms"):
                f.put("/linelist_atoms", self.vald_reader.linelist_atoms)
            if hasattr(self.vald_reader, "linelist_molecules"):
                f.put("/linelist_molecules", self.vald_reader.linelist_molecules)

            if hasattr(self.barklem_2016_data, "equilibrium_constants"):
                f.put(
                    "/molecules/equilibrium_constants",
                    self.barklem_2016_data.equilibrium_constants,
                )
            if hasattr(self.barklem_2016_data, "ionization_energies"):
                f.put(
                    "/molecules/ionization_energies",
                    self.barklem_2016_data.ionization_energies,
                )
            if hasattr(self.barklem_2016_data, "dissociation_energies"):
                f.put(
                    "/molecules/dissociation_energies",
                    self.barklem_2016_data.dissociation_energies,
                )
            if hasattr(self.barklem_2016_data, "partition_functions"):
                f.put(
                    "/molecules/partition_functions",
                    self.barklem_2016_data.partition_functions,
                )

            if hasattr(self, "collisions"):
                f.put("/collisions_data", self.collisions_prepared)
                f.put("/collisions_metadata", self.collisions_metadata)

            if hasattr(self, "cross_sections"):
                f.put("/photoionization_data", self.cross_sections_prepared)

            lines_metadata = pd.DataFrame(
                data=[["format", "version", "1.0"]], columns=["field", "key", "value"]
            ).set_index(["field", "key"])
            f.put("/lines_metadata", lines_metadata)

            meta = []
            meta.append(("format", "version", FORMAT_VERSION))

            total_checksum = hashlib.md5()
            for key in f.keys():
                # update the total checksum to sign the file
                total_checksum.update(serialize_pandas_object(f[key]).to_buffer())

                # save individual DataFrame/Series checksum
                checksum = hash_pandas_object(f[key])
                meta.append(("md5sum", key.lstrip("/"), checksum))

            # data sources versions
            meta.append(("datasets", "nist_weights", self.atomic_weights.version))

            meta.append(("datasets", "nist_spectra", self.ionization_energies.version))

            meta.append(("datasets", "gfall", self.gfall_reader.version))

            meta.append(("datasets", "zeta", self.zeta_data.version))

            if self.chianti_reader is not None:
                meta.append(("datasets", "chianti", self.chianti_reader.version))

            if self.cmfgen_reader is not None:
                meta.append(("datasets", "cmfgen", self.cmfgen_reader.version))

            if self.vald_reader is not None:
                meta.append(("datasets", "vald", self.vald_reader.version))

            # relevant package versions
            meta.append(("software", "python", platform.python_version()))
            imports = [
                "carsus",
                "astropy",
                "numpy",
                "pandas",
                "pyarrow",
                "tables",
                "ChiantiPy",
            ]

            for package in imports:
                meta.append(("software", package, __import__(package).__version__))

            meta_df = pd.DataFrame.from_records(
                meta, columns=["field", "key", "value"], index=["field", "key"]
            )

            uuid1 = uuid.uuid1().hex

            logger.info(f"Signing TARDISAtomData.")
            logger.info(f"Format Version: {FORMAT_VERSION}")
            logger.info(f"MD5: {total_checksum.hexdigest()}")
            logger.info(f"UUID1: {uuid1}")

            f.root._v_attrs["MD5"] = total_checksum.hexdigest()
            f.root._v_attrs["UUID1"] = uuid1
            f.root._v_attrs["FORMAT_VERSION"] = FORMAT_VERSION

            tz = pytz.timezone("UTC")
            date = datetime.now(tz).isoformat()
            f.root._v_attrs["DATE"] = date

            self.meta = meta_df
            f.put("/metadata", meta_df)
