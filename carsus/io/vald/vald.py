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
from astropy import units as u


logger = logging.getLogger(__name__)


class VALDReader(object):
    """
    Class for extracting lines data from vald files

    Attributes
    ----------
    fname: str
        path to vald data file
    strip_molecules: bool
        Whether to remove molecules from the data. Defaults to True.
    shortlist: bool
        Whether the parsed file is a shortlist or not.


    Methods
    --------
    vald_raw:
        Return pandas DataFrame representation of vald
    linelist:
        Return pandas DataFrame representation of linelist properties necessary to compute line opacities

    """

    vald_columns = [
        "elm_ion",
        "wave_unprepared",  # This is the wavelength column header before it is ingested overwritten with appropriate units
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

    vald_shortlist_columns = [
        "elm_ion",
        "wave_unprepared",  # This is the wavelength column header before it is ingested overwritten with appropriate units
        "e_low",
        "log_gf",
        "rad",
        "stark",
        "waals",
        "lande_factor",
        "central_depth",
        "reference",
    ]

    def __init__(self, fname=None, strip_molecules=True, shortlist=False):
        """
        Parameters
        ----------
        fname: str
            Path to the vald file (http or local file).
        strip_molecules: bool
            Whether to remove molecules from the data.
        shortlist: bool
            Whether the parsed file is a shortlist or not.
        """

        assert fname is not None, "fname must be specified"
        self.fname = fname

        self._vald_raw = None
        self._vald = None
        self._linelist = None
        self._stellar_linelist = False

        self._vald_columns = (
            self.vald_shortlist_columns.copy()
            if shortlist
            else self.vald_columns.copy()
        )

        self.strip_molecules = strip_molecules
        self.shortlist = shortlist

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

        DATA_RE_PATTERN = re.compile("'[a-zA-Z]+ \d+',[\s*-?\d+[\.\d+]+,]*")

        buffer, checksum = read_from_buffer(self.fname)
        content = buffer.read().decode()

        # Need to identify the wavelength column header and overwrite the wavelength to obtain units and air or vacuum
        # Also need to identify if Vmic is in the columns for correct column construction
        for line in content.split("\n")[:10]:
            if "WL" in line:
                for column_header in line.split():
                    if "WL" in column_header:
                        self._vald_columns[1] = column_header
                        logger.info(f"Found wavelength column header: {column_header}")
            if "Vmicro" in line and self._stellar_linelist == False:
                logger.info("Found Vmic column - This is a stellar vald linelist")
                self._vald_columns.insert(3, "v_mic")

                self._stellar_linelist = True

        vald = pd.read_csv(
            StringIO("\n".join(DATA_RE_PATTERN.findall(content))),
            names=self._vald_columns,
            index_col=False,
        )

        if self.shortlist:
            del vald["reference"]

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

        wave = vald.columns[1]
        if "nm" in wave:
            if "air" in wave:
                vald["wavelength"] = convert_wavelength_air2vacuum(
                    (vald[wave].values * u.nm).to(u.AA)
                )
            else:
                vald["wavelength"] = (vald[wave].values * u.nm).to(u.AA)
        elif "air" in wave:
            vald["wavelength"] = convert_wavelength_air2vacuum(vald[wave])
        else:
            vald["wavelength"] = vald[wave]

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
                atomic_number or chemical, ion_charge, wavelength, e_low, log_gf, rad, stark, waals
                optionally: v_mic (if stellar linelist) and e_up, j_lo, j_up (if not shortlist)
        """
        if self.shortlist:
            linelist_mask = [
                "chemical",
                "ion_charge",
                "wavelength",
                "log_gf",
                "e_low",
                "rad",
                "stark",
                "waals",
            ]
            if self._stellar_linelist:
                linelist_mask.insert(5, "v_mic")

        else:
            linelist_mask = [
                "chemical",
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

        if self.strip_molecules:
            linelist_mask[0] = "atomic_number"

        return vald[linelist_mask].copy()

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
            f.put("/linelist", self.linelist)
