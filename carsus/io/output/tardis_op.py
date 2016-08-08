import numpy as np
import pandas as pd
import hashlib
import uuid
import pickle

from pandas import HDFStore
from carsus.model import Atom, Ion, Line, Level, DataSource, ECollision
from carsus.model.meta import yield_limit, Base, IonListMixin
from carsus.util import data_path, convert_camel2snake, convert_wavelength_air2vacuum
from sqlalchemy import and_, case
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import joinedload, aliased
from sqlalchemy.orm.exc import NoResultFound
from astropy import constants as const
from astropy import units as u
from astropy.units import Quantity
from scipy import interpolate
from tardis.util import species_string_to_tuple

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

MEDIUM_VACUUM = 0
MEDIUM_AIR = 1

LINES_MAXRQ = 10000  # for yield_limit

ZETA_DATAFILE = data_path("knox_long_recombination_zeta.dat")


class AtomDataUnrecognizedMediumError(Exception):
    pass


class AtomData(object):
    """
    Class for creating the atomic dataframes for TARDIS

    Parameters:
    ------------
    session: SQLAlchemy session
    ions: list of species str
        The ions to be taken from the database.
    chianti_ions: list of species str
        The levels data for these ions will be taken from the CHIANTI database.
        The list *must* be a subset of `ions`
        (default: None)
    kurucz_short_name: str
        The short name of the Kurucz datasource
        (default: "ku_latest")
    nist_short_name: str
        The short name of the NIST datasource
        (default: "nist-asd")
    chianti_short_name: str
        The short name of the CHIANTI datasource
        (default: "chianti_v8.0.2")
    atom_masses_max_atomic_number: int
        The maximum atomic number to be stored in atom_masses
        (default: 30)
    lines_loggf_threshold: int
        log(gf) threshold for lines
    levels_metastable_loggf_threshold: int
        log(gf) threshold for flagging metastable levels
        (default: -3)
    temperatures: np.array
        The temperatures for calculating collision strengths
    zeta_datafile: none

    Attributes:
    ------------
    session: SQLAlchemy session
    atom_masses_param: dict
        The parameters for creating the `atom_masses` DataFrame
    levels_lines_param: dict
        The parameters for creating the `levels` DataFrame and the `lines` DataFrame
    collisions_param: dict
        The parameters for creating the `collisions` DataFrame

    ions: list of tuples (atomic_number, ion_charge)
    chianti_ions: list of tuples (atomic_number, ion_charge)

    ions_table: sqlalchemy.sql.schema.Table
    chianti_ions_table: sqlalchemy.sql.schema.Table

    ku_ds: carsus.model.atomic.DataSource instance
        The Kurucz datasource
    cu_ds: carsus.model.atomic.DataSource instance
        The CHIANTI datasource
    nist_ds: carsus.model.atomic.DataSource instance
        The NIST datasource


    atom_masses
    ionization_energies
    levels
    lines
    collisions
    macro_atom
    macro_atom_references
    zeta_data

    atom_masses_prepared
    ionization_energies_prepared
    levels_prepared
    lines_prepared
    collisions_prepared
    macro_atom_prepared
    macro_atom_references_prepared

    Methods:
    ---------
    create_atom_masses
    create_ionization_energies
    create_levels_lines
    create_collisions
    create_macro_atom
    create_macro_atom_references
    create_zeta_data

    prepare_atom_masses
    prepare_ionization_energies
    prepare_levels
    prepare_lines
    prepare_collisions
    prepare_macro_atom
    prepare_macro_atom_references

    """

    def __init__(self, session, ions, chianti_ions=None,
                 kurucz_short_name="ku_latest", chianti_short_name="chianti_v8.0.2", nist_short_name="nist-asd",
                 atom_masses_max_atomic_number=30, lines_loggf_threshold=-3, levels_metastable_loggf_threshold=-3,
                 collisions_temperatures=None,
                 zeta_datafile=ZETA_DATAFILE):

        self.session = session

        # Set the parameters for the dataframes
        self.atom_masses_param = {
            "max_atomic_number": atom_masses_max_atomic_number
        }

        self.levels_lines_param = {
            "levels_metastable_loggf_threshold": levels_metastable_loggf_threshold,
            "lines_loggf_threshold": lines_loggf_threshold
        }

        if collisions_temperatures is None:
            collisions_temperatures = np.linspace(2000, 50000, 20)
        else:
            collisions_temperatures = np.array(collisions_temperatures)

        self.collisions_param = {
            "temperatures": collisions_temperatures
        }

        self.ions = [tuple(species_string_to_tuple(species_str)) for species_str in ions]

        if chianti_ions is not None:
            # Get a list of tuples (atomic_number, ion_charge) for the chianti ions
            self.chianti_ions = [tuple(species_string_to_tuple(species_str)) for species_str in chianti_ions]
            try:
                assert set(self.chianti_ions).issubset(set(self.ions))
            except AssertionError:
                raise ValueError("`chianti_ions` *must* be a subset of `ions`!")
        else:
            self.chianti_ions = list()

        self._ions_table = None
        self._chianti_ions_table = None

        # Query the data sources
        self.ku_ds = None
        self.ch_ds = None
        self.nist_ds = None

        try:
            self.ku_ds = session.query(DataSource).filter(DataSource.short_name == kurucz_short_name).one()
        except NoResultFound:
            raise NoResultFound("Kurucz data source is not found!")

        try:
            self.nist_ds = session.query(DataSource).filter(DataSource.short_name == nist_short_name).one()
        except NoResultFound:
            raise NoResultFound("NIST ASD data source is not found!")

        if self.chianti_ions:
            try:
                self.ch_ds = session.query(DataSource).filter(DataSource.short_name == chianti_short_name).one()
            except NoResultFound:
                raise NoResultFound("Chianti data source is not found!")

        self.zeta_datafile = zeta_datafile

        self._atom_masses = None
        self._ionization_energies = None
        self._levels = None
        self._lines = None
        self._collisions = None
        self._macro_atom = None
        self._macro_atom_references = None
        self._zeta_data = None

    @property
    def ions_table(self):
        if self._ions_table is None:

            ions_table_name = "MainIonList" + str(hash(frozenset(self.ions)))

            try:
                ions_table = Base.metadata.tables[convert_camel2snake(ions_table_name)]
            except KeyError:
                ions_table = type(ions_table_name,(Base, IonListMixin), dict()).__table__

            try:
                # To create the temporary table use the session's current transaction-bound connection
                ions_table.create(self.session.connection())
            except OperationalError:  # Raised if the table already exists
                pass
            else:
                # Insert values from `ions` into the table
                self.session.execute(ions_table.insert(),
                    [{"atomic_number": atomic_number, "ion_charge": ion_charge}
                     for atomic_number, ion_charge in self.ions])

            self._ions_table = ions_table
        return self._ions_table

    @property
    def chianti_ions_table(self):

        if self._chianti_ions_table is None:

            chianti_ions_table_name = "ChiantiIonList" + str(hash(frozenset(self.chianti_ions)))

            try:
                chianti_ions_table = Base.metadata.tables[convert_camel2snake(chianti_ions_table_name)]
            except KeyError:
                chianti_ions_table = type(chianti_ions_table_name, (Base, IonListMixin), dict()).__table__

            try:
                # To create the temporary table use the session's current transaction-bound connection
                chianti_ions_table.create(self.session.connection())
            except OperationalError:  # Raised if the table already exists
                pass
            else:
                # Insert values from `ions` into the table
                if self.chianti_ions:
                    self.session.execute(chianti_ions_table.insert(),
                                         [{"atomic_number": atomic_number, "ion_charge": ion_charge}
                                          for atomic_number, ion_charge in self.chianti_ions])

            self._chianti_ions_table = chianti_ions_table
        return self._chianti_ions_table

    @property
    def atom_masses(self):
        if self._atom_masses is None:
            self._atom_masses = self.create_atom_masses(**self.atom_masses_param)
        return self._atom_masses

    def create_atom_masses(self, max_atomic_number=30):
        """
        Create a DataFrame containing *atomic masses*.

        Parameters
        ----------
        max_atomic_number: int
            The maximum atomic number to be stored in `atom_masses`
            (default: 30)

        Returns
        -------
        atom_masses : pandas.DataFrame
            DataFrame with:
                index: none;
                columns: atom_masses, symbol, name, mass[u].
        """
        atom_masses_q = self.session.query(Atom). \
            filter(Atom.atomic_number <= max_atomic_number).\
            order_by(Atom.atomic_number)

        atom_masses = list()
        for atom in atom_masses_q.options(joinedload(Atom.weights)):
            weight = atom.weights[0].quantity.value if atom.weights else None  # Get the first weight from the collection
            atom_masses.append((atom.atomic_number, atom.symbol, atom.name, weight))

        atom_masses_dtype = [("atomic_number", np.int), ("symbol", "|S5"), ("name", "|S150"), ("mass", np.float)]
        atom_masses = np.array(atom_masses, dtype=atom_masses_dtype)
        atom_masses = pd.DataFrame.from_records(atom_masses)

        return atom_masses

    @property
    def atom_masses_prepared(self):
        return self.prepare_atom_masses()

    def prepare_atom_masses(self):
        """
        Prepare the DataFrame with atomic masses for TARDIS.

        Returns
        -------
        atom_masses_prepared : pandas.DataFrame
            DataFrame with:
                index: atomic_number;
                columns: symbol, name, mass[u].
        """
        atom_masses_prepared = self.atom_masses.loc[:, ["atomic_number", "symbol", "name", "mass"]].copy()
        atom_masses_prepared = atom_masses_prepared.set_index("atomic_number")

        return atom_masses_prepared

    @property
    def ionization_energies(self):
        if self._ionization_energies is None:
            self._ionization_energies = self.create_ionization_energies()
        return self._ionization_energies

    def create_ionization_energies(self):
        """
        Create a DataFrame containing *ionization energies*.

        Returns
        -------
        ionization_energies : pandas.DataFrame
            DataFrame with:
                index: none;
                columns: atomic_number, ion_number, ionization_energy[eV]
        """
        ionization_energies_q = self.session.query(Ion).\
            join(self.ions_table,
                 and_(Ion.atomic_number == self.ions_table.c.atomic_number,
                      Ion.ion_charge == self.ions_table.c.ion_charge)).\
            order_by(Ion.atomic_number, Ion.ion_charge)

        ionization_energies = list()
        for ion in ionization_energies_q.options(joinedload(Ion.ionization_energies)):
            ionization_energy = ion.ionization_energies[0].quantity.value if ion.ionization_energies else None
            ionization_energies.append((ion.atomic_number, ion.ion_charge, ionization_energy))

        ionization_dtype = [("atomic_number", np.int), ("ion_number", np.int), ("ionization_energy", np.float)]
        ionization_energies = np.array(ionization_energies, dtype=ionization_dtype)

        ionization_energies = pd.DataFrame.from_records(ionization_energies)

        return ionization_energies

    @property
    def ionization_energies_prepared(self):
        return self.prepare_ionization_energies()

    def prepare_ionization_energies(self):
        """
        Prepare the DataFrame with ionization energies for TARDIS

        Returns
        -------
        ionization_energies : pandas.DataFrame
            DataFrame with:
                index: atomic_number, ion_number;
                columns: ionization_energy[eV].

        Notes
        ------
        In TARDIS `ion_number` describes the final ion state,
        e.g. H I - H II is described with ion_number = 1
        On the other hand, in carsus `ion_number` describes the lower ion state,
        e.g. H I - H II is described with ion_number = 0
        For this reason we add 1 to `ion_number` in this prepare method.
        """
        ionization_energies_prepared = self.ionization_energies.loc[:, ["atomic_number", "ion_number",
                                                                        "ionization_energy"]].copy()
        ionization_energies_prepared.loc[:, "ion_number"] += 1

        ionization_energies_prepared = ionization_energies_prepared.set_index(["atomic_number", "ion_number"])

        return ionization_energies_prepared

    @property
    def levels(self):
        if self._levels is None:
            self._levels, self._lines = self.create_levels_lines(**self.levels_lines_param)
        return self._levels

    @property
    def lines(self):
        if self._lines is None:
            self._levels, self._lines = self.create_levels_lines(**self.levels_lines_param)
        return self._lines

    def _build_levels_q(self):

        lvl_alias = aliased(Level)

        whens = list()

        if self.chianti_ions:
            # 1. If ion is in `chianti_ions` and there exist levels for this ion from
            #    the chianti source in the database, then select from the chianti source
            whens.append(
                 (self.session.query(lvl_alias). \
                 join(self.chianti_ions_table,
                      and_(lvl_alias.atomic_number == self.chianti_ions_table.c.atomic_number,
                           lvl_alias.ion_charge == self.chianti_ions_table.c.ion_charge)). \
                 filter(and_(lvl_alias.atomic_number == Level.atomic_number,
                             lvl_alias.ion_charge == Level.ion_charge),
                        lvl_alias.data_source == self.ch_ds).exists(), self.ch_ds.data_source_id)
            )

        # 2. Else if there exist levels from kurucz for this ion then select from the kurucz source
        whens.append((self.session.query(lvl_alias). \
                     filter(and_(lvl_alias.atomic_number == Level.atomic_number,
                                 lvl_alias.ion_charge == Level.ion_charge),
                                 lvl_alias.data_source == self.ku_ds).exists(), self.ku_ds.data_source_id))

        # 3. Else select from the nist source (the ground levels)
        case_stmt = case(whens=whens, else_=self.nist_ds.data_source_id)

        levels_q = self.session.query(Level).\
            join(self.ions_table,
                 and_(Level.atomic_number == self.ions_table.c.atomic_number,
                      Level.ion_charge == self.ions_table.c.ion_charge)).\
            filter(Level.data_source_id == case_stmt)

        return levels_q

    def _build_lines_q(self, levels_ids):
        levels_subq = self.session.query(Level.level_id.label("level_id")). \
            filter(Level.level_id.in_(levels_ids)).subquery()

        lines_q = self.session.query(Line). \
            join(levels_subq, Line.lower_level_id == levels_subq.c.level_id)

        return lines_q

    @staticmethod
    def _get_all_levels_data(levels_q):
        levels = list()
        for lvl in levels_q.options(joinedload(Level.energies)):
            try:
                energy = None
                # Try to find the measured energy for this level
                for nrg in lvl.energies:
                    if nrg.method == "meas":
                        energy = nrg.quantity
                        break
                # If the measured energy is not available, try to get the first one
                if energy is None:
                    energy = lvl.energies[0].quantity
            except IndexError:
                print "No energy is available for level {0}".format(lvl.level_id)
                continue
            levels.append(
                (lvl.level_id, lvl.atomic_number, lvl.ion_charge, energy.value, lvl.g))

        # Create a dataframe with the levels data
        levels_dtype = [("level_id", np.int), ("atomic_number", np.int),
                        ("ion_number", np.int), ("energy", np.float), ("g", np.int)]
        levels = np.array(levels, dtype=levels_dtype)
        levels = pd.DataFrame.from_records(levels, index="level_id")

        return levels

    @staticmethod
    def _get_all_lines_data(lines_q):
        lines = list()
        for line in yield_limit(lines_q.options(joinedload(Line.wavelengths), joinedload(Line.gf_values)),
                                Line.line_id, maxrq=LINES_MAXRQ):
            try:
                # Try to get the first gf value
                gf = line.gf_values[0].quantity
            except IndexError:
                print "No gf value is available for line {0}".format(line.line_id)
                continue
            try:
                # Try to get the first wavelength
                wavelength = line.wavelengths[0]
            except IndexError:
                print "No wavelength is available for line {0}".format(line.line_id)
                continue

            if wavelength.medium == MEDIUM_VACUUM:
                wavelength_value = wavelength.quantity.value
            elif wavelength.medium == MEDIUM_AIR:
                wavelength_value = convert_wavelength_air2vacuum(wavelength.quantity.value)
            else:
                raise AtomDataUnrecognizedMediumError(
                    "The medium {} is not recognized (0 - vacuum, 1 - air)".format(wavelength.medium)
                )

            lines.append((line.line_id, line.lower_level_id, line.upper_level_id, wavelength_value, gf.value))

        lines_dtype = [("line_id", np.int), ("lower_level_id", np.int), ("upper_level_id", np.int),
                       ("wavelength", np.float), ("gf", np.float)]
        lines = np.array(lines, dtype=lines_dtype)
        lines = pd.DataFrame.from_records(lines, index="line_id")

        lines["loggf"] = np.log10(lines["gf"])

        return lines

    @staticmethod
    def _create_metastable_flags(levels, lines, levels_metastable_loggf_threshold=-3):

        # Filter lines on the loggf threshold value
        metastable_lines = lines.loc[lines["loggf"] > levels_metastable_loggf_threshold]

        # Count the remaining strong transitions
        metastable_lines_grouped = metastable_lines.groupby("upper_level_id")
        metastable_counts = metastable_lines_grouped["upper_level_id"].count()
        metastable_counts.name = "metastable_counts"

        # If there are no strong transitions for a level (the count is NaN) then the metastable flag is True
        # else (the count is a natural number) the metastable flag is False
        levels = levels.join(metastable_counts)
        metastable_flags = levels["metastable_counts"].isnull()
        metastable_flags.name = "metastable"

        return metastable_flags

    @staticmethod
    def _create_artificial_fully_ionized(levels):
        """ Create artificial levels for fully ionized ions. """
        fully_ionized_levels = list()

        for atomic_number, _ in levels.groupby("atomic_number"):
            fully_ionized_levels.append(
                (-1, atomic_number, atomic_number, 0, 0.0, 1, True)
            )

        levels_columns = ["level_id", "atomic_number", "ion_number", "level_number", "energy", "g", "metastable"]
        fully_ionized_levels_dtypes = [(key, levels.dtypes[key]) for key in levels_columns]

        fully_ionized_levels = np.array(fully_ionized_levels, dtype=fully_ionized_levels_dtypes)

        return pd.DataFrame(data=fully_ionized_levels)

    def create_levels_lines(self, levels_metastable_loggf_threshold=-3, lines_loggf_threshold=-3):
        """
        Create a DataFrame containing *levels data* and a DataFrame containing *lines data*.

        Parameters
        ----------
        levels_metastable_loggf_threshold: int
            log(gf) threshold for flagging metastable levels
        lines_loggf_threshold: int
            log(gf) threshold for lines

        Returns
        -------
        levels: pandas.DataFrame
            DataFrame with:
                index: level_id
                columns: atomic_number, ion_number, level_number, energy[eV], g[1]

        lines: pandas.DataFrame
            DataFrame with:
                index: line_id;
                columns: atomic_number, ion_number, level_number_lower, level_number_upper,
                         wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1], B_ul[?], B_ul[?], A_ul[1/s].
        """
        levels_q = self._build_levels_q()
        levels_all = self._get_all_levels_data(levels_q)

        lines_q = self._build_lines_q(levels_all.index.values)
        lines_all = self._get_all_lines_data(lines_q)

        # Culling autoionization levels
        ionization_energies = self.ionization_energies.set_index(["atomic_number", "ion_number"])
        levels_w_ionization_energies = levels_all.join(ionization_energies, on=["atomic_number", "ion_number"])
        levels = levels_all.loc[
            levels_w_ionization_energies["energy"] < levels_w_ionization_energies["ionization_energy"]
        ].copy()

        # Clean lines
        lines = lines_all.join(pd.DataFrame(index=levels.index), on="lower_level_id", how="inner").\
            join(pd.DataFrame(index=levels.index), on="upper_level_id", how="inner")

        # Culling lines with low gf values
        lines = lines.loc[lines["loggf"] > lines_loggf_threshold]

        # Do not clean levels that don't exist in lines

        # Create the metastable flags for levels
        levels["metastable"] = self._create_metastable_flags(levels, lines_all, levels_metastable_loggf_threshold)

        # Create level numbers
        levels.sort_values(["atomic_number", "ion_number", "energy", "g"], inplace=True)
        levels["level_number"] = levels.groupby(['atomic_number', 'ion_number'])['energy']. \
            transform(lambda x: np.arange(len(x))).values
        levels["level_number"] = levels["level_number"].astype(np.int)

        # Join atomic_number, ion_number, level_number_lower, level_number_upper on lines
        lower_levels = levels.rename(columns={"level_number": "level_number_lower", "g": "g_l"}). \
                              loc[:, ["atomic_number", "ion_number", "level_number_lower", "g_l"]]
        upper_levels = levels.rename(columns={"level_number": "level_number_upper", "g": "g_u"}). \
                              loc[:, ["level_number_upper", "g_u"]]
        lines = lines.join(lower_levels, on="lower_level_id").join(upper_levels, on="upper_level_id")

        # Calculate absorption oscillator strength f_lu and emission oscillator strength f_ul
        lines["f_lu"] = lines["gf"] / lines["g_l"]
        lines["f_ul"] = lines["gf"] / lines["g_u"]

        # Calculate frequency
        lines['nu'] = u.Unit('angstrom').to('Hz', lines['wavelength'], u.spectral())

        # Calculate Einstein coefficients
        einstein_coeff = (4 * np.pi ** 2 * const.e.gauss.value ** 2) / (const.m_e.cgs.value * const.c.cgs.value)
        lines['B_lu'] = einstein_coeff * lines['f_lu'] / (const.h.cgs.value * lines['nu'])
        lines['B_ul'] = einstein_coeff * lines['f_ul'] / (const.h.cgs.value * lines['nu'])
        lines['A_ul'] = 2 * einstein_coeff * lines['nu'] ** 2 / const.c.cgs.value ** 2 * lines['f_ul']

        # Reset indexes because `level_id` cannot be an index once we
        # add artificial levels for fully ionized ions that don't have ids (-1)
        lines = lines.reset_index()
        levels = levels.reset_index()

        # Create and append artificial levels for fully ionized ions
        artificial_fully_ionized_levels = self._create_artificial_fully_ionized(levels)
        levels = levels.append(artificial_fully_ionized_levels, ignore_index=True)
        levels = levels.sort_values(["atomic_number", "ion_number", "level_number"])

        return levels, lines

    @property
    def levels_prepared(self):
        return self.prepare_levels()

    def prepare_levels(self):
        """
        Prepare the DataFrame with levels for TARDIS

        Returns
        -------
        levels_prepared: pandas.DataFrame
            DataFrame with:
                index: none;
                columns: atomic_number, ion_number, level_number, energy[eV], g[1], metastable.
        """

        levels_prepared = self.levels.loc[:, ["atomic_number", "ion_number", "level_number",
                                              "energy", "g", "metastable"]].copy()

        # Set index
        # levels.set_index(["atomic_number", "ion_number", "level_number"], inplace=True)

        levels_prepared = levels_prepared.reset_index(drop=True)

        return levels_prepared

    @property
    def lines_prepared(self):
        return self.prepare_lines()

    def prepare_lines(self):
        """
            Prepare the DataFrame with lines for TARDIS

            Returns
            -------
            lines_prepared : pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: line_id, atomic_number, ion_number, level_number_lower, level_number_upper,
                             wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1], B_ul[?], B_ul[?], A_ul[1/s].
        """

        lines_prepared = self.lines.loc[:, ["line_id", "wavelength", "atomic_number", "ion_number",
                                            "f_ul", "f_lu", "level_number_lower", "level_number_upper",
                                            "nu", "B_lu", "B_ul", "A_ul"]].copy()

        # Set the index
        # lines.set_index(["atomic_number", "ion_number", "level_number_lower", "level_number_upper"], inplace=True)

        lines_prepared = lines_prepared.reset_index(drop=True)

        return lines_prepared

    @property
    def collisions(self):
        if self._collisions is None:
            self._collisions = self.create_collisions(**self.collisions_param)
        return self._collisions

    def _build_collisions_q(self, levels_ids):
        levels_subq = self.session.query(Level.level_id.label("level_id")). \
            filter(Level.level_id.in_(levels_ids)).subquery()

        collisions_q = self.session.query(ECollision). \
            join(levels_subq, ECollision.lower_level_id == levels_subq.c.level_id)

        return collisions_q

    def create_collisions(self, temperatures):
        """
            Create a DataFrame containing *collisions* data.

            Parameters
            -----------
            temperatures: np.array
                The temperatures for calculating collision strengths
                (default: None)

            Returns
            -------
            collisions: pandas.DataFrame
                DataFrame with:
        """

        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")

        levels_idx = levels.index.values
        collisions_q = self._build_collisions_q(levels_idx)

        collisions = list()
        for e_col in collisions_q.options(joinedload(ECollision.gf_values),
                                          joinedload(ECollision.temp_strengths)):

            # Try to get the first gf value
            try:
                gf = e_col.gf_values[0].quantity
            except IndexError:
                print "No gf is available for electron collision {0}".format(e_col.e_col_id)
                continue

            btemp, bscups = (list(ts) for ts in zip(*e_col.temp_strengths_tuple))

            collisions.append((e_col.e_col_id, e_col.lower_level_id, e_col.upper_level_id,
                e_col.data_source_id, btemp, bscups, e_col.bt92_ttype, e_col.bt92_cups, gf.value))

        collisions_dtype = [("e_col_id", np.int), ("lower_level_id", np.int), ("upper_level_id", np.int),
                            ("ds_id", np.int),  ("btemp", 'O'), ("bscups", 'O'), ("ttype", np.int),
                            ("cups", np.float), ("gf", np.float)]

        collisions = np.array(collisions, dtype=collisions_dtype)
        collisions = pd.DataFrame.from_records(collisions, index="e_col_id")

        # Join atomic_number, ion_number, level_number_lower, level_number_upper
        lower_levels = levels.rename(columns={"level_number": "level_number_lower", "g": "g_l", "energy": "energy_lower"}). \
                              loc[:, ["atomic_number", "ion_number", "level_number_lower", "g_l", "energy_lower"]]
        upper_levels = levels.rename(columns={"level_number": "level_number_upper", "g": "g_u", "energy": "energy_upper"}). \
                              loc[:, ["level_number_upper", "g_u", "energy_upper"]]

        collisions = collisions.join(lower_levels, on="lower_level_id").join(upper_levels, on="upper_level_id")

        # Calculate delta_e
        kb_ev = const.k_B.cgs.to('eV / K').value
        collisions["delta_e"] = (collisions["energy_upper"] - collisions["energy_lower"])/kb_ev

        def calculate_collisional_strength(row, temperatures):
            """
                Function to calculation upsilon from Burgess & Tully 1992 (TType 1 - 4; Eq. 23 - 38)
            """
            c = row["cups"]
            x_knots = np.linspace(0, 1, len(row["btemp"]))
            y_knots = row["bscups"]
            delta_e = row["delta_e"]
            g_u = row["g_u"]

            ttype = row["ttype"]
            if ttype > 5: ttype -= 5

            kt = kb_ev * temperatures

            spline_tck = interpolate.splrep(x_knots, y_knots)

            if ttype == 1:
                x = 1 - np.log(c) / np.log(kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func * np.log(kt / delta_e + np.exp(1))

            elif ttype == 2:
                x = (kt / delta_e) / (kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func

            elif ttype == 3:
                x = (kt / delta_e) / (kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func / (kt / delta_e + 1)

            elif ttype == 4:
                x = 1 - np.log(c) / np.log(kt / delta_e + c)
                y_func = interpolate.splev(x, spline_tck)
                upsilon = y_func * np.log(kt / delta_e + c)

            elif ttype == 5:
                raise ValueError('Not sure what to do with ttype=5')

            #### 1992A&A...254..436B Equation 20 & 22 #####

            c_ul = 8.63e-6 * upsilon / (g_u * temperatures**.5)
            return tuple(c_ul)

        collisions["c_ul"] = collisions.apply(calculate_collisional_strength, axis=1, args=(temperatures,))

        # Calculate g_ratio
        collisions["g_ratio"] = collisions["g_l"] / collisions["g_u"]

        return collisions

    @property
    def collisions_prepared(self):
        return self.prepare_collisions()

    def prepare_collisions(self):
        """
            Prepare the DataFrame with electron collisions for TARDIS
            Returns
            -------
            collisions_prepared : pandas.DataFrame
                DataFrame with:
                    index: atomic_number, ion_number, level_number_lower, level_number_upper;
                    columns: e_col_id, delta_e, g_ratio, c_ul.
        """

        collisions_prepared = self.collisions.loc[:, ["atomic_number", "ion_number",
                                                      "level_number_lower", "level_number_upper",
                                                      "delta_e", "g_ratio", "c_ul"]].copy()

        # Set multiindex
        collisions_prepared = collisions_prepared.reset_index()
        collisions_prepared = collisions_prepared.set_index(["atomic_number", "ion_number",
                                                             "level_number_lower", "level_number_upper"])

        return collisions_prepared

    @property
    def macro_atom(self):
        if self._macro_atom is None:
            self._macro_atom = self.create_macro_atom()
        return self._macro_atom

    def create_macro_atom(self):
        """
            Create a DataFrame containing *macro atom* data.

            Returns
            -------
            macro_atom: pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number, target_level_number,
                        transition_line_id, transition_type, transition_probability.

            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html

        """
        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")

        lvl_energy_lower = levels.rename(columns={"energy": "energy_lower"}).loc[:, ["energy_lower"]]
        lvl_energy_upper = levels.rename(columns={"energy": "energy_upper"}).loc[:, ["energy_upper"]]

        lines = self.lines.set_index("line_id")
        lines = lines.join(lvl_energy_lower, on="lower_level_id").join(lvl_energy_upper, on="upper_level_id")

        macro_atom = list()
        macro_atom_dtype = [("atomic_number", np.int), ("ion_number", np.int),
                            ("source_level_number", np.int), ("target_level_number", np.int),
                            ("transition_line_id", np.int), ("transition_type", np.int), ("transition_probability", np.float)]

        for line_id, row in lines.iterrows():
            atomic_number, ion_number = row["atomic_number"], row["ion_number"]
            level_number_lower, level_number_upper = row["level_number_lower"], row["level_number_upper"]
            nu = row["nu"]
            f_ul, f_lu = row["f_ul"], row["f_lu"]
            e_lower, e_upper = row["energy_lower"], row["energy_upper"]

            transition_probabilities_dict = dict()  # type : probability
            transition_probabilities_dict[P_EMISSION_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * (e_upper - e_lower)
            transition_probabilities_dict[P_INTERNAL_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * e_lower
            transition_probabilities_dict[P_INTERNAL_UP] = f_lu * e_lower / (const.h.cgs.value * nu)

            macro_atom.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_EMISSION_DOWN, transition_probabilities_dict[P_EMISSION_DOWN]))
            macro_atom.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_INTERNAL_DOWN, transition_probabilities_dict[P_INTERNAL_DOWN]))
            macro_atom.append((atomic_number, ion_number, level_number_lower, level_number_upper,
                                    line_id, P_INTERNAL_UP, transition_probabilities_dict[P_INTERNAL_UP]))

        macro_atom = np.array(macro_atom, dtype=macro_atom_dtype)
        macro_atom = pd.DataFrame(macro_atom)

        macro_atom = macro_atom.sort_values(["atomic_number", "ion_number", "source_level_number"])

        return macro_atom

    @property
    def macro_atom_prepared(self):
        return self.prepare_macro_atom()

    def prepare_macro_atom(self):
        """
            Prepare the DataFrame with macro atom data for TARDIS
            Returns
            -------
            macro_atom_prepared : pandas.DataFrame
                DataFrame with the *macro atom data* with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number, destination_level_number,
                        transition_line_id, transition_type, transition_probability.
            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html
        """

        macro_atom_prepared = self.macro_atom.loc[:, ["atomic_number", "ion_number",
                                                      "source_level_number", "target_level_number",
                                                      "transition_type", "transition_probability",
                                                      "transition_line_id"]].copy()

        # ToDo: choose between `target_level_number` and `destination_level_number`
        # Rename `target_level_number` to `destination_level_number` used in TARDIS
        # Personally, I think `target_level_number` is better so I use it in Carsus.
        macro_atom_prepared = macro_atom_prepared.rename(columns={"target_level_number": "destination_level_number"})

        macro_atom_prepared = macro_atom_prepared.reset_index(drop=True)

        return macro_atom_prepared

    @property
    def macro_atom_references(self):
        if self._macro_atom_references is None:
            self._macro_atom_references = self.create_macro_atom_references()
        return self._macro_atom_references

    def create_macro_atom_references(self):
        """
            Create a DataFrame containing *macro atom reference* data.

            Returns
            -------
            macro_atom_reference : pandas.DataFrame
                DataFrame with:
                index: no index;
                and columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total
        """
        macro_atom_references = self.levels.rename(columns={"level_number": "source_level_number"}).\
                                       loc[:, ["atomic_number", "ion_number", "source_level_number", "level_id"]]

        count_down = self.lines.groupby("upper_level_id").size()
        count_down.name = "count_down"

        count_up = self.lines.groupby("lower_level_id").size()
        count_up.name = "count_up"

        macro_atom_references = macro_atom_references.join(count_down, on="level_id").join(count_up, on="level_id")
        macro_atom_references = macro_atom_references.drop("level_id", axis=1)

        macro_atom_references = macro_atom_references.fillna(0)
        macro_atom_references["count_total"] = 2*macro_atom_references["count_down"] + macro_atom_references["count_up"]

        # Convert to int
        macro_atom_references["count_down"] = macro_atom_references["count_down"].astype(np.int)
        macro_atom_references["count_up"] = macro_atom_references["count_up"].astype(np.int)
        macro_atom_references["count_total"] = macro_atom_references["count_total"].astype(np.int)

        return macro_atom_references

    @property
    def macro_atom_references_prepared(self):
        return self.prepare_macro_atom_references()

    def prepare_macro_atom_references(self):
        """
            Prepare the DataFrame with macro atom references for TARDIS

            Returns
            -------
            macro_atom_references_prepared : pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total.
        """
        macro_atom_references_prepared = self.macro_atom_references.loc[:, ["atomic_number", "ion_number",
                                                                            "source_level_number", "count_down",
                                                                            "count_up", "count_total"]].copy()

        macro_atom_references_prepared = macro_atom_references_prepared.reset_index(drop=True)

        return macro_atom_references_prepared

    @property
    def zeta_data(self):
        if self._zeta_data is None:
            self._zeta_data = self.create_zeta_data()
        return self._zeta_data

    def create_zeta_data(self):
        zeta_data = np.loadtxt(self.zeta_datafile, usecols=xrange(1, 23), dtype=np.float64)
        t_rads = np.arange(2000, 42000, 2000)
        return pd.DataFrame(zeta_data[:, 2:],
                            index=pd.MultiIndex.from_arrays(zeta_data[:, :2].transpose().astype(int)),
                            columns=t_rads)

    def to_hdf(self, hdf5_path, store_atom_masses=True, store_ionization_energies=True,
               store_levels=True, store_lines=True, store_collisions=True, store_macro_atom=True,
               store_macro_atom_references=True, store_zeta_data=True):
        """
            Store the dataframes in an HDF5 file

            Parameters
            ------------
            hdf5_path: str
                The path of the HDF5 file
            store_atom_masses: bool
                Store the `atom_masses_prepared` DataFrame
                (default: True)
            store_ionization_energies: bool
                Store the `ionization_energies_prepared` DataFrame
                (default: True)
            store_levels: bool
                Store the `levels_prepared` DataFrame
                (default: True)
            store_lines: bool
                Store the `lines_prepared` DataFrame
                (default: True)
            store_collisions: bool
                Store the `collisions_prepared` DataFrame
                (default: True)
            store_macro_atom: bool
                Store the `macro_atom_prepared` DataFrame
                (default: True)
            store_macro_atom_references: bool
                Store the `macro_atom_references_prepared` DataFrame
                (default: True)
            store_zeta_data: bool
                Store the `zeta_data` DataFrame
                (default: True)
        """

        with HDFStore(hdf5_path) as store:

            if store_atom_masses:
                store.put("atom_data", self.atom_masses_prepared)

            if store_ionization_energies:
                store.put("ionization_data", self.ionization_energies_prepared)

            if store_levels:
                store.put("levels", self.levels_prepared)

            if store_lines:
                store.put("lines", self.lines_prepared)

            if store_collisions:
                store.put("collisions", self.collisions_prepared)

            if store_macro_atom:
                store.put("macro_atom_data", self.macro_atom_prepared)

            if store_macro_atom_references:
                store.put("macro_atom_references", self.macro_atom_references_prepared)

            if store_zeta_data:
                store.put("zeta_data", self.zeta_data)

            # Set the root attributes
            # It seems that the only way to set the root attributes is to use `_v_attrs`
            store.root._v_attrs["database_version"] = "v0.9"

            print "Signing AtomData with MD5 and UUID1"

            md5_hash = hashlib.md5()
            for key in store.keys():
                md5_hash.update(store[key].values.data)

            uuid1 = uuid.uuid1().hex

            store.root._v_attrs['md5'] = md5_hash.hexdigest()
            store.root._v_attrs['uuid1'] = uuid1