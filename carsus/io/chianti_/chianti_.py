import os
import re
import pickle
import logging
import numpy as np
import pandas as pd
from numpy.testing import assert_almost_equal
from astropy import units as u
from carsus.io.util import convert_species_tuple2chianti_str
from carsus.util import parse_selected_species

# Compatibility with older versions and pip versions:
try:
    from ChiantiPy.tools.io import versionRead
    import ChiantiPy.core as ch 

except ImportError:
    # Shamefully copied from their GitHub source:
    import chianti.core as ch
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


logger = logging.getLogger(__name__)

masterlist_ions_path = os.path.join(
    os.getenv('XUVTOP'), "masterlist", "masterlist_ions.pkl"
)

masterlist_ions_file = open(masterlist_ions_path, 'rb')
masterlist_ions = pickle.load(masterlist_ions_file).keys()
# Exclude the "d" ions for now
masterlist_ions = [_ for _ in masterlist_ions
                   if re.match(r"^[a-z]+_\d+$", _)]

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


class ChiantiReader:
    """ 
        Class for extracting levels, lines and collisional data 
        from Chianti.

        Mimics the GFALLReader class.

        Attributes
        ----------
        levels : DataFrame
        lines : DataFrame
        collisions: DataFrame
        version : str

    """

    def __init__(self, ions, collisions=False, priority=10):
        """
        Parameters
        ----------
        ions : string
            Selected Chianti ions.

        collisions: bool, optional
            Grab collisional data, by default False.

        priority: int
            Priority of the current data source.        
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

            ch_ion = convert_species_tuple2chianti_str(ion)
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

        lines['wavelength'] = u.Quantity(lines['wavelength'], u.AA).to('nm').value

        col_columns = ['temperatures', 'collision_strengths', 'gf', 'energy', 'ttype', 'cups']
        if get_collisions:
            collisions = pd.concat(col_list, sort=True)
            collisions = collisions.reset_index()
            collisions = collisions.rename(columns={'lower_level_index': 'level_index_lower',
                                                    'upper_level_index': 'level_index_upper',
                                                    'gf_value': 'gf',})
            collisions['level_index_lower'] -= 1
            collisions['level_index_upper'] -= 1
            collisions = collisions.set_index(['atomic_number', 'ion_charge',
                                               'level_index_lower', 'level_index_upper'])
            collisions = collisions[col_columns]
            self.collisions = collisions

        self.levels = levels
        self.lines = lines
        self.version = versionRead()

    def to_hdf(self, fname):
        """
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """

        with pd.HDFStore(fname, 'w') as f:
            f.put('/levels', self.levels)
            f.put('/lines', self.lines)
            f.put('/collisions', self.collisions)
