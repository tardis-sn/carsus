import logging

import numpy as np
import pandas as pd

from carsus.io.output.collisions import CollisionsPreparer
from carsus.io.output.ionization_energies import IonizationEnergiesPreparer
from carsus.io.output.levels_lines import LevelsLinesPreparer
from carsus.io.output.macro_atom import MacroAtomPreparer
from carsus.io.output.photo_ionization import PhotoIonizationPreparer

from carsus.util import (
    hash_pandas_object,
    serialize_pandas_object,
)

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
    
        self.gfall_reader = gfall_reader
        self.zeta_data = zeta_data
        self.chianti_reader = chianti_reader
        self.cmfgen_reader = cmfgen_reader
        self.nndc_reader = nndc_reader
        self.vald_reader = vald_reader
        self.barklem_2016_data = barklem_2016_data
        self.levels_lines_param = levels_lines_param
        self.collisions_param = collisions_param

        self.ionization_energies_preparer = IonizationEnergiesPreparer(self.cmfgen_reader, ionization_energies)

        self.levels_lines_preparer = LevelsLinesPreparer(self.ionization_energies, self.gfall_reader, self.chianti_reader, self.cmfgen_reader)
        self.levels_all = self.levels_lines_preparer.all_levels_data
        self.lines_all = self.levels_lines_preparer.all_lines_data
        self.levels_lines_preparer.create_levels_lines(**self.levels_lines_param)
        self.levels, self.lines = self.levels_lines_preparer.levels, self.levels_lines_preparer.lines

        self.macro_atom_preparer = MacroAtomPreparer(self.levels, self.lines)
        self.macro_atom_preparer.create_macro_atom()
        self.macro_atom_preparer.create_macro_atom_references()

        self.collisions_preparer = CollisionsPreparer(self.chianti_reader, self.levels, self.levels_all, self.lines_all, self.levels_lines_preparer.chianti_ions, self.cmfgen_reader, self.collisions_param)

        if (cmfgen_reader is not None) and hasattr(cmfgen_reader, "cross_sections"):
            self.cross_sections_preparer = PhotoIonizationPreparer(self.levels, self.levels_all, self.lines_all, self.cmfgen_reader,  self.levels_lines_preparer.cmfgen_ions)
        else:
            self.cross_sections_preparer = None
            

        logger.info("Finished.")
    
    @property
    def ionization_energies(self):
        return self.ionization_energies_preparer.ionization_energies

    @property
    def ionization_energies_prepared(self):
        return self.ionization_energies_preparer.ionization_energies_prepared

    @property
    def cross_sections(self):
        if self.cross_sections_preparer is not None:
            return self.cross_sections_preparer.cross_sections
        else:
            return None

    @property
    def cross_sections_prepared(self):
        if self.cross_sections_preparer is not None:
            return self.cross_sections_preparer.cross_sections_prepared
        else:
            return None
    
    @property
    def levels_prepared(self):
        return self.levels_lines_preparer.levels_prepared

    @property
    def lines_prepared(self):
        return self.levels_lines_preparer.lines_prepared
    
    @property 
    def macro_atom(self):
        return self.macro_atom_preparer.macro_atom
    
    @property 
    def macro_atom_references(self):
        return self.macro_atom_preparer.macro_atom_references

    @property
    def macro_atom_prepared(self):
        return self.macro_atom_preparer.macro_atom_prepared
    
    @property
    def macro_atom_references_prepared(self):
        return self.macro_atom_preparer.macro_atom_references_prepared
    
    @property
    def collisions_prepared(self):
        return self.collisions_preparer.prepare_collisions()
    
    @property
    def collisions_metadata(self):
        return self.collisions_preparer.collisions_metadata

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
            f.put("/levels_data", self.levels_prepared)
            f.put("/lines_data", self.lines_prepared)
            f.put("/macro_atom_data", self.macro_atom_prepared)
            f.put("/macro_atom_references", self.macro_atom_references_prepared)

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
