import re
import logging
import pandas as pd
import numpy as np
from io import StringIO
from carsus.io.util import read_from_buffer
from carsus.util.helpers import (
    convert_wavelength_air2vacuum,
    ATOMIC_SYMBOLS_DATA,
    convert_symbol2atomic_number,
)


VALD_URL = "https://media.githubusercontent.com/media/tardis-sn/carsus-db/master/vald/vald_sample.dat"

logger = logging.getLogger(__name__)


class VALDReader(object):
    """
    Class for extracting lines data from vald files

    Attributes
    ----------
    fname: str
        path to vald data file
    strip_molecules: bool
        Whether to remove molecules from the data.


    Methods
    --------
    vald_raw:
        Return pandas DataFrame representation of vald

    """

    vald_columns = [
        "elm_ion",
        "wl_air",
        "log_gf",
        "e_low",
        "j_lo",
        "e_up",
        "j_up",
        "lande_lower",
        "lande_upper",
        "lande_mean",
        "rad",
        "stark",
        "waals",
    ]

    def __init__(self, fname=None, strip_molecules=True):
        """
        Parameters
        ----------
        fname: str
            Path to the vald file (http or local file).
        strip_molecules: bool
            Whether to remove molecules from the data.

        """

        self.fname = VALD_URL if fname is None else fname

        self._vald_raw = None
        self._vald = None
        self._linelist = None

        self.strip_molecules = strip_molecules

    @property
    def vald_raw(self):
        """
        Reads the raw data and returns raw vald data as a pandas DataFrame
        """
        if self._vald_raw is None:
            self._vald_raw, self.version = self.read_vald_raw()
        return self._vald_raw

    @property
    def vald(self):
        """
        Processes the raw vald DataFrame
        """
        if self._vald is None:
            self._vald = self.parse_vald(strip_molecules=self.strip_molecules)
        return self._vald

    @property
    def linelist(self):
        """
        Prepares the linelist from the processed vald DataFrame
        """
        if self._linelist is None:
            self._linelist = self.extract_linelist(self.vald)
        return self._linelist

    def read_vald_raw(self, fname=None):
        """
        Reads in a vald data file

        Parameters
        ----------
        fname: ~str
            path to vald data file


        Returns
        -------
            pandas.DataFrame
                pandas Dataframe representation of vald


            str
                MD5 checksum
        """

        if fname is None:
            fname = self.fname

        logger.info(f"Parsing VALD from: {fname}")

        # FORMAT
        # Elm Ion       WL_air(A)  log gf* E_low(eV) J lo  E_up(eV) J up   lower   upper    mean   Rad.  Stark    Waals
        # 'TiO 1',     4100.00020, -11.472,  0.2011, 31.0,  3.2242, 32.0, 99.000, 99.000, 99.000, 6.962, 0.000, 0.000,

        data_match = re.compile("'[a-zA-Z]+ \d+',[\s*-?\d+[\.\d+]+,]*")

        buffer, checksum = read_from_buffer(self.fname)
        vald = pd.read_csv(
            StringIO("\n".join(data_match.findall(buffer.read().decode()))),
            names=self.vald_columns,
            index_col=False,
        )

        return vald, checksum

    def parse_vald(self, vald_raw=None, strip_molecules=True):
        """
        Parses raw vald DataFrame


        Parameters
        ----------
        vald_raw: pandas.DataFrame
        strip_molecules: bool
            If True, remove molecules from vald

        Returns
        -------
            pandas.DataFrame
        """

        vald = vald_raw if vald_raw is not None else self.vald_raw.copy()

        vald["elm_ion"] = vald["elm_ion"].str.replace("'", "")
        vald[["chemical", "ion_charge"]] = vald["elm_ion"].str.split(" ", expand=True)
        vald["ion_charge"] = vald["ion_charge"].astype(int) - 1
        vald["wavelength"] = convert_wavelength_air2vacuum(vald["wl_air"])

        del vald["elm_ion"]

        if strip_molecules:
            vald = self._strip_molecules(vald)
            vald.reset_index(drop=True, inplace=True)

            atom_nums = np.zeros(len(vald), dtype=int)
            atom_nums = [
                convert_symbol2atomic_number(symbol)
                for symbol in vald.chemical.to_list()
            ]
            vald["atomic_number"] = atom_nums

        return vald

    def _strip_molecules(self, vald):
        """
        Removes molecules from a vald dataframe
        """
        return vald[vald.chemical.isin(ATOMIC_SYMBOLS_DATA["symbol"])]

    def extract_linelist(self, vald):
        """
        Parameters
        ----------
        vald: pandas.DataFrame

        Returns
        -------
            pandas.DataFrame
                vald linelist containing only the following columns:
                atomic_number, ion_charge, wavelength, log_gf, rad, stark, waals
        """
        return vald[
            [
                "atomic_number",
                "ion_charge",
                "wavelength",
                "log_gf",
                "e_low",
                "e_up",
                "j_lo",
                "j_up",
                "rad",
                "stark",
                "waals",
            ]
        ].copy()

    def to_hdf(self, fname):
        """
        Parameters
        ----------
        fname : path
            Path to the HDF5 output file
        """
        with pd.HDFStore(fname, "w") as f:
            f.put("/vald_raw", self.vald_raw)
            f.put("/vald", self.vald)
