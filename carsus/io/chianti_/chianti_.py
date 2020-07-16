import os
import re
import pickle
import logging
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from astropy import units as u
from sqlalchemy import and_
from pyparsing import ParseException
from carsus.io.base import IngesterError
from carsus.io.util import convert_species_tuple2chianti_str
from carsus.util import convert_atomic_number2symbol, parse_selected_species
from carsus.model import DataSource, Ion, Level, LevelEnergy,\
    Line, LineGFValue, LineAValue, LineWavelength, MEDIUM_VACUUM, \
    ECollision, ECollisionEnergy, ECollisionGFValue, ECollisionTempStrength

# Compatibility with older versions and pip versions:
try:
    from ChiantiPy.tools.io import versionRead
    import ChiantiPy.core as ch 

except ImportError:
    # Shamefully copied from their GitHub source:
    def versionRead():
        """
        Read the version number of the CHIANTI database
        """
        xuvtop = os.environ['XUVTOP']
        vFileName = os.path.join(xuvtop, 'VERSION')
        vFile = open(vFileName)
        versionStr = vFile.readline()
        vFile.close()
        return versionStr.strip()
    import chianti.core as ch


logger = logging.getLogger(__name__)

masterlist_ions_path = os.path.join(
    os.getenv('XUVTOP'), "masterlist", "masterlist_ions.pkl"
)

masterlist_ions_file = open(masterlist_ions_path, 'rb')
masterlist_ions = pickle.load(masterlist_ions_file).keys()
# Exclude the "d" ions for now
masterlist_ions = [_ for _ in masterlist_ions
                   if re.match("^[a-z]+_\d+$", _)]

masterlist_version = versionRead()


class ChiantiIonReaderError(Exception):
    pass


class ChiantiIonReader(object):
    """
        Class for reading ion data from the CHIANTI database

        Attributes
        ----------
        ion: chianti.core.ion instance

        Methods
        -------
        levels
            Return a DataFrame with the data for ion's levels

        lines
            Return a DataFrame with the data for ion's lines

        collisions
            Return a DataFrame with the data for ion's electron collisions

        bound_levels
            Same as `levels`, but only for bound levels (with energy < ionization_potential)

        bound_lines
            Same as `lines`, but only for bound levels (with energy < ionization_potential)

        bound_collisions
            Same as `collisions`, but only for bound levels (with energy < ionization_potential)
    """

    elvlc_dict = {
        'lvl': 'level_index',
        'ecm': 'energy',  # cm-1
        'ecmth': 'energy_theoretical',  # cm-1
        'j': 'J',
        'spd': 'L',
        'spin': 'spin_multiplicity',
        'pretty': 'pretty',  # configuration + term
        'label': 'label'
    }

    wgfa_dict = {
        'avalue': 'a_value',
        'gf': 'gf_value',
        'lvl1': 'lower_level_index',
        'lvl2': 'upper_level_index',
        'wvl': 'wavelength'
    }

    scups_dict = {
        'btemp': 'temperatures',
        'bscups': 'collision_strengths',
        'gf': 'gf_value',
        'de': 'energy',  # Rydberg
        'lvl1': 'lower_level_index',
        'lvl2': 'upper_level_index',
        'ttype': 'ttype',  # BT92 Transition type
        'cups': 'cups'  # BT92 scaling parameter
    }

    def __init__(self, ion_name):

        self.ion = ch.ion(ion_name)
        self._levels = None
        self._lines = None
        self._collisions = None

    @property
    def levels(self):
        if self._levels is None:
            self._levels = self.read_levels()
        return self._levels

    @property
    def lines(self):
        if self._lines is None:
            self._lines = self.read_lines()
        return self._lines

    @property
    def collisions(self):
        if self._collisions is None:
            self._collisions = self.read_collisions()
        return self._collisions

    @property
    def last_bound_level(self):
        ionization_potential = u.eV.to(
            u.Unit("cm-1"), value=self.ion.Ip, equivalencies=u.spectral())
        last_row = self.levels.loc[self.levels['energy']
                                   < ionization_potential].tail(1)
        return last_row.index[0]

    @property
    def bound_levels(self):
        return self.levels.loc[:self.last_bound_level]

    def filter_bound_transitions(self, transitions):
        """ Filter transitions DataFrames on bound levels.

            The most succinct and accurate way to do this is to use slicing on multi index,
            but due to some bug in pandas out-of-range rows are included in the resulting DataFrame.
        """
        transitions = transitions.reset_index()
        transitions = transitions.loc[transitions["upper_level_index"]
                                      <= self.last_bound_level]
        transitions = transitions.set_index(
            ["lower_level_index", "upper_level_index"])
        transitions = transitions.sort_index()
        return transitions

    @property
    def bound_lines(self):
        bound_lines = self.filter_bound_transitions(self.lines)
        return bound_lines

    @property
    def bound_collisions(self):
        bound_collisions = self.filter_bound_transitions(self.collisions)
        return bound_collisions

    def read_levels(self):

        try:
            elvlc = self.ion.Elvlc
        except AttributeError:
            raise ChiantiIonReaderError(
                "No levels data is available for ion {}".format(self.ion.Spectroscopic))

        levels_dict = {}

        for key, col_name in self.elvlc_dict.items():
            levels_dict[col_name] = elvlc.get(key)

        # Check that ground level energy is 0
        try:
            for key in ['energy', 'energy_theoretical']:
                assert_almost_equal(levels_dict[key][0], 0)
        except AssertionError:
            raise ValueError('Level 0 energy is not 0.0')

        levels = pd.DataFrame(levels_dict)

        # Replace empty labels with NaN
        levels.loc[:, "label"] = levels["label"].replace(
            r'\s+', np.nan, regex=True)

        # Extract configuration and term from the "pretty" column
        levels[["term", "configuration"]] = levels["pretty"].str.rsplit(
            ' ', expand=True, n=1)
        levels = levels.drop("pretty", axis=1)

        levels = levels.set_index("level_index")
        levels = levels.sort_index()

        return levels

    def read_lines(self):

        try:
            wgfa = self.ion.Wgfa
        except AttributeError:
            raise ChiantiIonReaderError(
                "No lines data is available for ion {}".format(self.ion.Spectroscopic))

        lines_dict = {}

        for key, col_name in self.wgfa_dict.items():
            lines_dict[col_name] = wgfa.get(key)

        lines = pd.DataFrame(lines_dict)

        # two-photon transitions are given a zero wavelength and we ignore them for now
        lines = lines.loc[~(lines["wavelength"] == 0)]

        # theoretical wavelengths have negative values
        def parse_wavelength(row):
            if row["wavelength"] < 0:
                wvl = -row["wavelength"]
                method = "th"
            else:
                wvl = row["wavelength"]
                method = "m"
            return pd.Series([wvl, method])

        lines[["wavelength", "method"]] = lines.apply(parse_wavelength, axis=1)

        lines = lines.set_index(["lower_level_index", "upper_level_index"])
        lines = lines.sort_index()

        return lines

    def read_collisions(self):

        try:
            scups = self.ion.Scups
        except AttributeError:
            raise ChiantiIonReaderError(
                "No collision data is available for ion {}".format(self.ion.Spectroscopic))

        collisions_dict = {}

        for key, col_name in self.scups_dict.items():
            collisions_dict[col_name] = scups.get(key)

        collisions = pd.DataFrame(collisions_dict)

        collisions = collisions.set_index(
            ["lower_level_index", "upper_level_index"])
        collisions = collisions.sort_index()

        return collisions


class ChiantiIngester(object):
    """
        Class for ingesting data from the CHIANTI database

        Parameters
        -----------
        session:  SQLAlchemy session
        ions:  list of species str, if set to None then masterlist
            (default None)
        ds_short_name: str
            Short name of the datasource

        Attributes
        ----------
        session: SQLAlchemy session
        data_source: DataSource instance
        ion_readers : list of ChiantiIonReader instances

        Methods
        -------
        ingest(session)
            Persists data into the database
    """

    masterlist_ions = masterlist_ions
    ds_prefix = 'chianti'

    def __init__(self, session, ions=None, ds_short_name=None):
        if ds_short_name is None:
            ds_short_name = '{}_v{}'.format(
                self.ds_prefix,
                masterlist_version)

        self.session = session
        self.ion_readers = list()
        self.ions = list()

        if ions is not None:
            try:
                ions = parse_selected_species(ions)
            except ParseException:
                raise ValueError(
                    'Input is not a valid species string {}'.format(ions))
            self.ions = [convert_species_tuple2chianti_str(_) for _ in ions]
        else:
            self.ions = masterlist_ions

        for ion in self.ions:
            if ion in self.masterlist_ions:
                self.ion_readers.append(ChiantiIonReader(ion))
            else:
                print("Ion {0} is not available".format(ion))

        self.data_source = DataSource.as_unique(
            self.session, short_name=ds_short_name)
        # To get the id if a new data source was created
        if self.data_source.data_source_id is None:
            self.session.flush()

    def get_lvl_index2id(self, ion):
        """ Return a DataFrame that maps levels indexes to ids """

        q_ion_lvls = self.session.query(Level.level_id.label("id"),
                                        Level.level_index.label("index")). \
            filter(and_(Level.ion == ion,
                        Level.data_source == self.data_source))

        lvl_index2id = list()
        for id, index in q_ion_lvls:
            lvl_index2id.append((index, id))

        lvl_index2id_dtype = [("index", np.int), ("id", np.int)]
        lvl_index2id = np.array(lvl_index2id, dtype=lvl_index2id_dtype)
        lvl_index2id = pd.DataFrame.from_records(lvl_index2id, index="index")

        return lvl_index2id

    def ingest_levels(self):

        print("Ingesting levels from {}".format(self.data_source.short_name))

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion - 1

            ion = Ion.as_unique(
                self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            try:
                bound_levels = rdr.bound_levels
            except ChiantiIonReaderError:
                print("Levels not found for ion {} {}".format(
                    convert_atomic_number2symbol(atomic_number), ion_charge))
                continue

            print("Ingesting levels for {} {}".format(
                convert_atomic_number2symbol(atomic_number), ion_charge))

            # ToDo: Determine parity from configuration

            for index, row in bound_levels.iterrows():

                level = Level(
                    ion=ion, data_source=self.data_source, level_index=index,
                    configuration=row["configuration"], term=row["term"],
                    L=row["L"], J=row["J"], spin_multiplicity=row["spin_multiplicity"]
                )

                level.energies = []
                for column, method in [('energy', 'meas'), ('energy_theoretical', 'theor')]:
                    if row[column] != -1:  # check if the value exists
                        level.energies.append(
                            LevelEnergy(quantity=row[column]*u.Unit("cm-1"),
                                        data_source=self.data_source,
                                        method=method),
                        )
                self.session.add(level)

    def ingest_lines(self):

        print("Ingesting lines from {}".format(self.data_source.short_name))

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion - 1

            ion = Ion.as_unique(
                self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            try:
                bound_lines = rdr.bound_lines
            except ChiantiIonReaderError:
                print("Lines not found for ion {} {}".format(
                    convert_atomic_number2symbol(atomic_number), ion_charge))
                continue

            print("Ingesting lines for {} {}".format(
                convert_atomic_number2symbol(atomic_number), ion_charge))

            lvl_index2id = self.get_lvl_index2id(ion)

            for index, row in bound_lines.iterrows():

                # index: (lower_level_index, upper_level_index)
                lower_level_index, upper_level_index = index

                try:
                    lower_level_id = int(lvl_index2id.loc[lower_level_index])
                    upper_level_id = int(lvl_index2id.loc[upper_level_index])
                except KeyError:
                    raise IngesterError("Levels from this source have not been found."
                                        "You must ingest levels before transitions")

                # Create a new line
                line = Line(
                    lower_level_id=lower_level_id,
                    upper_level_id=upper_level_id,
                    data_source=self.data_source,
                    wavelengths=[
                        LineWavelength(quantity=row["wavelength"]*u.AA,
                                       data_source=self.data_source,
                                       medium=MEDIUM_VACUUM,
                                       method=row["method"])
                    ],
                    a_values=[
                        LineAValue(quantity=row["a_value"]*u.Unit("s**-1"),
                                   data_source=self.data_source)
                    ],
                    gf_values=[
                        LineGFValue(quantity=row["gf_value"],
                                    data_source=self.data_source)
                    ]
                )

                self.session.add(line)

    def ingest_collisions(self):

        print("Ingesting collisions from {}".format(
            self.data_source.short_name))

        for rdr in self.ion_readers:

            atomic_number = rdr.ion.Z
            ion_charge = rdr.ion.Ion - 1

            ion = Ion.as_unique(
                self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            try:
                bound_collisions = rdr.bound_collisions
            except ChiantiIonReaderError:
                print("Collisions not found for ion {} {}".format(
                    convert_atomic_number2symbol(atomic_number), ion_charge))
                continue

            print("Ingesting collisions for {} {}".format(
                convert_atomic_number2symbol(atomic_number), ion_charge))

            lvl_index2id = self.get_lvl_index2id(ion)

            for index, row in bound_collisions.iterrows():

                # index: (lower_level_index, upper_level_index)
                lower_level_index, upper_level_index = index

                try:
                    lower_level_id = int(lvl_index2id.loc[lower_level_index])
                    upper_level_id = int(lvl_index2id.loc[upper_level_index])
                except KeyError:
                    raise IngesterError("Levels from this source have not been found."
                                        "You must ingest levels before transitions")

                # Create a new electron collision
                e_col = ECollision(
                    lower_level_id=lower_level_id,
                    upper_level_id=upper_level_id,
                    data_source=self.data_source,
                    bt92_ttype=row["ttype"],
                    bt92_cups=row["cups"],
                    energies=[
                        ECollisionEnergy(quantity=row["energy"]*u.rydberg,
                                         data_source=self.data_source)
                    ],
                    gf_values=[
                        ECollisionGFValue(quantity=row["gf_value"],
                                          data_source=self.data_source)
                    ]
                )

                e_col.temp_strengths = [
                    ECollisionTempStrength(temp=temp, strength=strength)
                    for temp, strength in zip(row["temperatures"], row["collision_strengths"])
                ]

                self.session.add(e_col)

    def ingest(self, levels=True, lines=False, collisions=False):

        if levels:
            self.ingest_levels()
            self.session.flush()

        if lines:
            self.ingest_lines()
            self.session.flush()

        if collisions:
            self.ingest_collisions()
            self.session.flush()


class ChiantiReader:
    """
        Class for extracting lines and levels data from Chianti.
        Mimics the GFALLReader class.

        Attributes
        ----------
        levels : DataFrame
        lines : DataFrame
    """

    def __init__(self, ions, collisions=False, priority=10):
        """
        Parameters
        ----------
        ions : string
            Selected Chianti ions.

        collisions: bool, optional
            Grab collisional data (default is False).
        """
        self.ions = parse_selected_species(ions)
        self.priority = priority
        self._get_levels_lines(get_collisions=collisions)

    def _get_levels_lines(self, get_collisions=False):
        """Generates `levels`, `lines`  and `collisions` DataFrames.

        Parameters
        ----------
        get_collisions : bool, optional
            Grab collisional data, by default False.
        """
        lvl_list = []
        lns_list = []
        col_list = []
        for ion in self.ions:

            ch_ion = ch_ion = convert_species_tuple2chianti_str(ion)
            reader = ChiantiIonReader(ch_ion)

            # Do not keep levels if lines are not available.
            try:
                lvl = reader.levels
                lns = reader.lines

            except ChiantiIonReaderError:
                logger.info(f'Missing levels/lines data for `{ch_ion}`.')
                continue

            lvl['atomic_number'] = ion[0]
            lvl['ion_charge'] = ion[1]

            # Indexes must start from zero
            lvl.index = range(0, len(lvl))
            lvl.index.name = 'level_index'
            lvl_list.append(reader.levels)

            lns['atomic_number'] = ion[0]
            lns['ion_charge'] = ion[1]
            lns_list.append(lns)

            if get_collisions:
                try:
                    col = reader.collisions
                    col['atomic_number'] = ion[0]
                    col['ion_charge'] = ion[1]
                    col_list.append(col)

                except ChiantiIonReaderError:
                    logger.info(f'Missing collisional data for `{ch_ion}`.')

        levels = pd.concat(lvl_list, sort=True)
        levels = levels.rename(columns={'J': 'j'})
        levels['method'] = None
        levels['priority'] = self.priority
        levels = levels.reset_index()
        levels = levels.set_index(
            ['atomic_number', 'ion_charge', 'level_index'])
        levels = levels[['energy', 'j', 'label', 'method', 'priority']]

        lines = pd.concat(lns_list, sort=True)
        lines = lines.reset_index()
        lines = lines.rename(columns={'lower_level_index': 'level_index_lower',
                                      'upper_level_index': 'level_index_upper',
                                      'gf_value': 'gf'})

        # Kurucz levels starts from zero, Chianti from 1.
        lines['level_index_lower'] = lines['level_index_lower'] - 1
        lines['level_index_upper'] = lines['level_index_upper'] - 1

        lines = lines.set_index(['atomic_number', 'ion_charge',
                                 'level_index_lower', 'level_index_upper'])
        lines['energy_upper'] = None
        lines['energy_lower'] = None
        lines['j_upper'] = None
        lines['j_lower'] = None
        lines = lines[['energy_upper', 'j_upper', 'energy_lower', 'j_lower',
                       'wavelength', 'gf']]

        if get_collisions:
            collisions = pd.concat(col_list, sort=True)
            collisions = collisions.reset_index()
            collisions = collisions.rename(columns={'lower_level_index': 'level_index_lower',
                                                    'upper_level_index': 'level_index_upper',
                                                    'gf_value': 'gf',})
            # Do we need to fix level indexes here too ?
            collisions['level_index_lower'] -= 1
            collisions['level_index_upper'] -= 1
            collisions = collisions.set_index(['atomic_number', 'ion_charge',
                                               'level_index_lower', 'level_index_upper'])
            collisions = collisions[['temperatures', 'collision_strengths', 'gf', 'energy',
                                     'ttype', 'cups']]

        self.levels = levels
        self.lines = lines

        if get_collisions:
            self.collisions = collisions
        else:
            self.collisions = pd.DataFrame()

        self.version = versionRead()

    # TODO:
    def to_hdf(self, fname):
        raise NotImplementedError
