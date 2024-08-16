import pandas as pd
from carsus.io.util import read_from_buffer
import logging

logger = logging.getLogger(__name__)


BARKLEM_COLLET_DATA_URL = "https://raw.githubusercontent.com/tardis-sn/carsus-data-molecules-barklem2016/main/data/"


class BarklemCollet2016Reader(object):
    """
    A class for reading and parsing data from the Barklem & Collet 2016 dataset.

    This class initializes with a file path and provides methods to read raw data and parse it into structured dataframes for dissociation energies, ionization energies, partition functions, and equilibrium constants. It also manages versioning information for each dataset.

    Args:
        fpath (str, optional): The file path to the data source. If not provided, defaults to BARKLEM_COLLET_DATA_URL.

    Attributes:
        barklem_2016_raw (tuple): Raw data from the Barklem & Collet 2016 dataset as a tuple of dataframes and checksums.
        dissociation_energies (DataFrame): Dataframe containing dissociation energies.
        ionization_energies (DataFrame): Dataframe containing ionization energies.
        partition_functions (DataFrame): Dataframe containing partition functions.
        equilibrium_constants (DataFrame): Dataframe containing equilibrium constants.
        dissociation_version (str): Version information for dissociation energies.
        ionization_version (str): Version information for ionization energies.
        partition_version (str): Version information for partition functions.
        equilibrium_version (str): Version information for equilibrium constants.
    """

    def __init__(self, fpath=None):

        self.fpath = BARKLEM_COLLET_DATA_URL if fpath is None else fpath
        self._barklem_2016_raw = None

        self._dissociation_energies = None
        self._ionization_energies = None
        self._partition_functions = None
        self._equilibrium_constants = None

        self.dissociation_version = None
        self.ionization_version = None
        self.partition_version = None
        self.equilibrium_version = None

    @property
    def barklem_2016_raw(self):
        """
        Reads the raw data and returns raw barklem2016 data as a series of dataframes
        """
        if self._barklem_2016_raw is None:
            self._barklem_2016_raw = self.read_barklem_2016_raw()
        return self._barklem_2016_raw

    def read_barklem_2016_raw(self, fpath=None):
        """
        Reads raw data from the Barklem & Collet 2016 dataset and returns structured dataframes.

        This function retrieves data from specified files related to dissociation energies, ionization energies, partition functions, and equilibrium constants. It processes the data into pandas dataframes and returns them along with their respective checksums for verification.

        Args:
            fpath (str, optional): The file path to the data source. If not provided, defaults to the instance's fpath attribute.

        Returns:
            tuple: A tuple containing:
                - dissociation_energies_df (DataFrame): Dataframe of dissociation energies.
                - ionization_energies_df (DataFrame): Dataframe of ionization energies.
                - partition_functions_df (DataFrame): Dataframe of partition functions.
                - equilibrium_constants_df (DataFrame): Dataframe of equilibrium constants.
                - diss_checksum (str): Checksum for the dissociation energies data.
                - ioniz_checksum (str): Checksum for the ionization energies data.
                - part_checksum (str): Checksum for the partition functions data.
                - equil_checksum (str): Checksum for the equilibrium constants data.
        """
        if fpath is None:
            fpath = self.fpath

        logger.info(f"Parsing Barklem & Collet 2016 from: {fpath}")

        dissociation_energies_data = f"{fpath}table1.dat"
        ionization_energies_data = f"{fpath}table4.dat"
        partition_functions_data = f"{fpath}table6.dat"
        equilibrium_constants_data = f"{fpath}table7.dat"

        diss_buffer, diss_checksum = read_from_buffer(dissociation_energies_data)
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

        ioniz_buffer, ioniz_checksum = read_from_buffer(ionization_energies_data)
        ionization_energies_df = pd.read_csv(
            ioniz_buffer,
            delimiter=r"\s+",
            comment="#",
            header=None,
            names=["Atomic_Number", "Element", "IE1 [eV]", "IE2 [eV]", "IE3 [eV]"],
            index_col=0,
        )

        part_buffer, part_checksum = read_from_buffer(partition_functions_data)
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

        equil_buffer, equil_checksum = read_from_buffer(equilibrium_constants_data)
        equilibrium_constants_df = pd.read_csv(
            equil_buffer,
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
            diss_checksum,
            ioniz_checksum,
            part_checksum,
            equil_checksum,
        )

    def parse_barklem_2016(self):
        """
        Parses raw barklem2016 data
        """

        (
            dissociation_energies_df,
            ionization_energies_df,
            partition_functions_df,
            equilibrium_constants_df,
            dissociation_version,
            ionization_version,
            partition_version,
            equilibrium_version,
        ) = self.barklem_2016_raw

        self._dissociation_energies = dissociation_energies_df
        self._ionization_energies = ionization_energies_df
        self._partition_functions = partition_functions_df
        self._equilibrium_constants = equilibrium_constants_df

        self.dissociation_version = dissociation_version
        self.ionization_version = ionization_version
        self.partition_version = partition_version
        self.equilibrium_version = equilibrium_version

    @property
    def dissociation_energies(self):
        """
        Returns the dissociation energies dataframe
        """
        if self._dissociation_energies is None:
            self.parse_barklem_2016()
        return self._dissociation_energies

    @property
    def ionization_energies(self):
        """
        Returns the ionization energies dataframe
        """
        if self._ionization_energies is None:
            self.parse_barklem_2016()
        return self._ionization_energies

    @property
    def partition_functions(self):
        """
        Returns the partition functions dataframe
        """
        if self._partition_functions is None:
            self.parse_barklem_2016()
        return self._partition_functions

    @property
    def equilibrium_constants(self):
        """
        Returns the equilibrium constants dataframe
        """
        if self._equilibrium_constants is None:
            self.parse_barklem_2016()
        return self._equilibrium_constants

    def to_hdf(self, fname):
        """
        Write data to HDF5 file
        """
        with pd.HDFStore(fname, "w") as f:
            f.put("dissociation_energies", self.dissociation_energies)
            f.put("ionization_energies", self.ionization_energies)
            f.put("partition_functions", self.partition_functions)
            f.put("equilibrium_constants", self.equilibrium_constants)
