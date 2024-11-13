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
    shortlist: bool
        Whether the parsed file is a shortlist or not.


    Methods
    --------
    vald_raw:
        Return pandas DataFrame representation of vald
    linelist_atoms:
        Return pandas DataFrame representation of atomic linelist properties necessary to compute line opacities
    linelist_molecules:
        Return pandas DataFrame representation of molecular linelist properties necessary to compute line opacities

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

    def __init__(self, fname=None, shortlist=False):
        """
        Parameters
        ----------
        fname: str
            Path to the vald file (http or local file).
        shortlist: bool
            Whether the parsed file is a shortlist or not.
        """

        assert fname is not None, "fname must be specified"
        self.fname = fname

        self._vald_raw = None
        self._vald_atoms = None
        self._vald_molecules = None

        self._atom_linelist = None
        self._molecule_linelist = None

        self._stellar_linelist = False

        self._vald_columns = (
            self.vald_shortlist_columns.copy()
            if shortlist
            else self.vald_columns.copy()
        )

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
        Processes the raw vald DataFrame, separating atoms and molecules
        """
        if self._vald_atoms is None:
            self._vald_atoms, self._vald_molecules = self.parse_vald()
        return self._vald_atoms, self._vald_molecules

    @property
    def linelist_atoms(self):
        """
        Prepares the atomic linelist from the processed vald DataFrame
        """
        if self._atom_linelist is None:
            self._atom_linelist, self._molecule_linelist = self.extract_linelists(
                *self.vald
            )
        return self._atom_linelist

    @property
    def linelist_molecules(self):
        """
        Prepares the molecular linelist from the processed vald DataFrame
        """
        if self._molecule_linelist is None:
            self._atom_linelist, self._molecule_linelist = self.extract_linelists(
                *self.vald
            )
        return self._molecule_linelist

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

        DATA_RE_PATTERN = re.compile("'[a-zA-Z]+\d* \d+',[\s*-?\d+[\.\d+]+,]*")

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

    def check_wavelength_column_medium_and_units(self):
        """
        Checks the wavelength column's medium and units from the DataFrame's column header.

        This method examines the second column's header in the `vald_raw` DataFrame to determine the medium (air or vacuum) and the units of the wavelength values. It raises a ValueError if the medium or units cannot be determined from the column header.

        Returns
        -------
        wave_col_name : str
            The name of the wavelength column.
        wave_air : bool
            True if the wavelengths are in air, False if in vacuum.
        wave_units : astropy.units.core.PrefixUnit or astropy.units.composite.CompositeUnit
            The units of the wavelength, as an Astropy unit.

        Raises
        ------
        ValueError
            If the wavelength column header does not specify 'air' or 'vac' to indicate the medium, or if it does not contain recognizable units ('(A)', '(nm)', or '(cm-1)').

        """
        wave_col_name = self.vald_raw.columns[1]
        if "air" in wave_col_name:
            wave_air = True
        elif "vac" in wave_col_name:
            wave_air = False
        else:
            raise ValueError(
                "Wavelength column header does not contain air or vac - Not sure what medium the wavelengths are in"
            )
        if "(A)" in wave_col_name:
            wave_units = u.AA
        elif "(nm)" in wave_col_name:
            wave_units = u.nm
        elif "(cm-1)" in wave_col_name:
            wave_units = u.cm**-1
        else:
            raise ValueError(
                "Wavelength column header does not contain units - Not sure what units the wavelengths are in"
            )
        return wave_col_name, wave_air, wave_units

    def parse_vald(self, vald_raw=None):
        """
        Parses raw vald DataFrame


        Parameters
        ----------
        vald_raw: pandas.DataFrame


        Returns
        -------
            pandas.DataFrame
                vald atoms
            pandas.DataFrame
                vald molecules
        """

        vald = vald_raw if vald_raw is not None else self.vald_raw.copy()

        vald.loc[:, "elm_ion"] = vald["elm_ion"].str.replace("'", "")
        vald[["chemical", "ion_charge"]] = vald["elm_ion"].str.split(" ", expand=True)
        vald.loc[:, "ion_charge"] = vald["ion_charge"].astype(int) - 1

        # Check units and medium of wavelength column and create wavelength column in angstroms in vacuum
        (
            wave_col_name,
            wave_air,
            wave_units,
        ) = self.check_wavelength_column_medium_and_units()
        vald.loc[:, "wavelength"] = (
            (vald[wave_col_name].values * wave_units).to(u.AA).value
        )
        if wave_air:
            vald.loc[:, "wavelength"] = convert_wavelength_air2vacuum(
                (vald["wavelength"].values * wave_units).to(u.AA).value
            )

        del vald["elm_ion"]

        vald_atoms = vald[vald.chemical.isin(ATOMIC_SYMBOLS_DATA["symbol"])].copy()
        vald_molecules = vald[~vald.chemical.isin(ATOMIC_SYMBOLS_DATA["symbol"])].copy()

        # Generate atomic numbers and assign them
        vald_atoms.reset_index(drop=True, inplace=True)
        vald_atoms.loc[:, "atomic_number"] = [
            convert_symbol2atomic_number(symbol) for symbol in vald_atoms["chemical"]
        ]

        vald_molecules.rename(columns={"chemical": "molecule"}, inplace=True)

        return vald_atoms, vald_molecules

    def extract_linelists(self, vald_atoms, vald_molecules):
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
            pandas.DataFrame
                vald linelist containing only the following columns:
                molecule, ion_charge, wavelength, e_low, log_gf, rad, stark, waals
                optionally: e_up, j_lo, j_up (if not shortlist)

        """
        if self.shortlist:
            linelist_mask = [
                "atomic_number",
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

        linelist_atoms = vald_atoms[linelist_mask].copy()
        linelist_mask[0] = "molecule"
        linelist_molecules = vald_molecules[linelist_mask].copy()

        return linelist_atoms, linelist_molecules

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
            f.put("/linelist_atoms", self.linelist_atoms)
            f.put("/linelist_molecules", self.linelist_molecules)
