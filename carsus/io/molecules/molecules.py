import numpy as np
import pandas as pd
from carsus.io.base import BaseParser
from carsus.io.util import read_from_buffer
import logging
from pathlib import Path

BARKLEM_COLETT_DATA_URL = "https://raw.githubusercontent.com/tardis-sn/carsus-data-molecules-barklem2016/main/data/"


logger = logging.getLogger(__name__)


class BarklemColett2016Reader(object):

    def __init__(self, fpath=None):

        self.fpath = BARKLEM_COLETT_DATA_URL if fpath is None else fpath
        self._barklem_2016_raw = None
        self._barklem_2016 = None

        self.dissociation_energies = None
        self.ionization_energies = None
        self.partition_functions = None
        self.equilibrium_constants = None

    @property
    def barklem_2016_raw(self):
        """
        Reads the raw data and returns raw barklem2016 data as a series of dataframes
        """
        if self._barklem_2016_raw is None:
            self._barklem_2016_raw = self.read_barklem_2016_raw()
        return self._barklem_2016_raw

    def read_barklem_2016_raw(self, fpath=None):

        if fpath is None:
            fpath = self.fpath

        logger.info(f"Parsing Barklem & Collet 2016 from: {fpath}")

        dissociation_energies_data = f"{fpath}table1.dat"
        ionization_energies_data = f"{fpath}table4.dat"
        partition_functions_data = f"{fpath}table6.dat"
        equilibrium_constants_data = f"{fpath}table7.dat"

        diss_buffer, checksum = read_from_buffer(dissociation_energies_data)
        dissociation_energies_df = pd.read_csv(
            diss_buffer,
            delimiter=r"\s+",
            comment="#",
            header=None,
            names=[
                "Molecule",
                "Ion1",
                "Ion2",
                "H&H Energy [eV]",
                "H&H Sigma [eV]",
                "Luo Energy [eV]",
                "Luo Sigma [eV]",
                "G2 Energy [eV]",
                "G2 Sigma [eV]",
                "Adopted Energy [eV]",
                "Adopted Sigma [eV]",
            ],
            index_col=0,
        )

        ioniz_buffer, checksum = read_from_buffer(ionization_energies_data)
        ionization_energies_df = pd.read_csv(
            ioniz_buffer,
            delimiter=r"\s+",
            comment="#",
            header=None,
            names=["Atomic_Number", "Element", "IE1 [eV]", "IE2 [eV]", "IE3 [eV]"],
            index_col=0,
        )

        part_buffer, checksum = read_from_buffer(partition_functions_data)
        partition_functions_df = pd.read_csv(
            part_buffer,
            delimiter=r"\s{2,}",
            skiprows=[0, 1, 3],
            escapechar="#",
            index_col=0,
            engine="python",
        )
        partition_functions_df.index.name = "Molecule"
        partition_functions_df.columns = partition_functions_df.columns.astype(float)

        eq_buffer, checksum = read_from_buffer(equilibrium_constants_data)
        equilibrium_constants_df = pd.read_csv(
            eq_buffer,
            delimiter=r"\s{2,}",
            skiprows=[0, 1, 3],
            escapechar="#",
            index_col=0,
            engine="python",
        )
        equilibrium_constants_df.index.name = "Molecule"
        equilibrium_constants_df.columns = equilibrium_constants_df.columns.astype(
            float
        )

        return (
            dissociation_energies_df,
            ionization_energies_df,
            partition_functions_df,
            equilibrium_constants_df,
            checksum,
        )
