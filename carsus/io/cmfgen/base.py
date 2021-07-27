import gzip
import itertools
import logging
import pathlib

import astropy.units as u
import numpy as np
import pandas as pd
import roman
import yaml

from carsus import __path__ as CARSUS_PATH
from carsus.io.base import BaseParser
from carsus.util import convert_atomic_number2symbol, parse_selected_species

logger = logging.getLogger(__name__)


def open_cmfgen_file(fname, encoding='ISO-8859-1'):
    return gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname, encoding=encoding) 


def to_float(string):
    """
    String to float, also deals with Fortran 'D' type.

    Parameters
    ----------
    string : str

    Returns
    -------
    float
    """
    try:
        value = float(string.replace('D', 'E'))

    except ValueError:

        # Typo in `MG/VIII/23oct02/phot_sm_3000`, line 23340
        if string == '1-.00':
            value = 10.00

        # Typo in `SUL/V/08jul99/phot_op.big`, lines 9255-9257
        if string == '*********':
            value = np.nan

    return value


def find_row(fname, string1, string2=None, how='AND', row_number=False):
    """
    Search for strings in plain text files and returns the matching\
    line (or row number).

    Parameters
    ----------
    fname : str
        Path to plain text file.
    string1 : str
        String to search.
    string2 : str
        Secondary string to search (default is None).
    how : {'OR', 'AND', 'AND NOT'}
        Search method: `string1` <method> `string2`
            (default is 'AND').
    row_number : bool
        If true, returns row number (default is False).

    Returns
    -------
    str or int
        Returns matching line or match row number.
    """

    if string2 is None:
        string2 = ''

    with open_cmfgen_file(fname) as f:
        n = 0
        for line in f:

            n += 1
            if how == 'OR':
                if string1 in line or string2 in line:
                    break

            if how == 'AND':
                if string1 in line and string2 in line:
                    break

            if how == 'AND NOT':
                if string1 in line and string2 not in line:
                    break

        else:
            n, line = None, None

    if row_number is True:
        return n

    return line


def parse_header(fname, keys, start=0, stop=50):
    """
    Parse header information from CMFGEN files.

    Parameters
    ----------
    fname : str
        Path to plain text file.
    keys : list of str
        Entries to search.
    start : int
        First line to search in (default is 0).
    stop : int
        Last line to search in (default is 50).

    Returns
    -------
    dict
        Dictionary containing metadata.
    """
    meta = {k.strip('!'): None for k in keys}

    with open_cmfgen_file(fname) as f:
        for line in itertools.islice(f, start, stop):
            for k in keys:
                if k.lower() in line.lower():
                    meta[k.strip('!')] = line.split()[0]

    return meta


class CMFGENEnergyLevelsParser(BaseParser):
    """
    Description
    ----------
    base : pandas.DataFrame
    columns : list of str
    meta : dict
        Metadata extracted from file header.

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    keys = ['!Date',
            '!Format date',
            '!Number of energy levels',
            '!Ionization energy',
            '!Screened nuclear charge',
            '!Number of transitions',
            ]

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        skiprows = find_row(fname, "Number of transitions", row_number=True)
        nrows = int(meta['Number of energy levels'])
        kwargs = {'header': None,
                  'index_col': False,
                  'sep': '\s+',
                  'skiprows': skiprows,
                  'nrows': nrows}

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            logger.warning(f'Empty table: `{fname}`.')

        if df.shape[1] == 10:
            # Read column names and split them keeping just one space (e.g. '10^15 Hz')
            columns = find_row(fname, 'E(cm^-1)', "Lam").split('  ')
            columns = [c.rstrip().lstrip() for c in columns if c != '']
            columns = ['Configuration'] + columns
            df.columns = columns

        elif df.shape[1] == 7:
            df.columns = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'Hz 10^15', 'Lam(A)', 'ID']

        elif df.shape[1] == 6:
            df.columns = ['Configuration', 'g', 'E(cm^-1)', 'Hz 10^15', 'Lam(A)', 'ID']

        elif df.shape[1] == 5:
            df.columns = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'ID']

        else:
            logger.warning(f'Unknown column format: `{fname}`.')

        self.base = df
        self.columns = df.columns.tolist()
        self.fname = fname
        self.meta = meta

    def to_hdf(self, key='/energy_levels'):
        if not self.base.empty:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:
                f.put(key, self.base)
                f.get_storer(key).attrs.metadata = self.meta


class CMFGENOscillatorStrengthsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
        meta : dict
            Metadata extracted from file header.

        Methods
        -------
        load(fname)
            Parses the input file and stores the result in the `base` attribute.
    """

    keys = CMFGENEnergyLevelsParser.keys

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        skiprows = find_row(fname, "Transition", "Lam", row_number=True) +1
        # Parse only tables listed increasing lower level i, e.g. `FE/II/24may96/osc_nahar.dat`
        nrows = int(meta['Number of transitions'])
        kwargs = {'header': None,
                  'index_col': False,
                  'sep': '\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])',
                  'skiprows': skiprows,
                  'nrows': nrows}

        columns = ['label_lower', 'label_upper', 'f', 'A',
                    'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            logger.warning(f'Empty table: `{fname}`.')

        if df.shape[1] == 9:
            df.columns = columns

        # These files have 9-column, but the current regex produces 10 columns
        elif df.shape[1] == 10:
            df.columns = columns + ['?']
            df = df.drop(columns=['?'])

        elif df.shape[1] == 8:
            df.columns = columns[:7] + ['#']
            df = df.drop(columns=['#'])
            df['Lam(obs)'] = np.nan
            df['% Acc'] = np.nan

        else:
            logger.warning(f'Unknown column format: `{fname}`.')

        if df.shape[0] > 0 and 'D' in str(df['f'][0]):
            df['f'] = df['f'].map(to_float)
            df['A'] = df['A'].map(to_float)

        self.base = df
        self.columns = df.columns.tolist()
        self.fname = fname
        self.meta = meta

    def to_hdf(self, key='/oscillator_strengths'):
        if not self.base.empty:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:
                f.put(key, self.base)
                f.get_storer(key).attrs.metadata = self.meta


class CMFGENCollisionalStrengthsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
        meta : dict
            Metadata extracted from file header.

        Methods
        -------
        load(fname)
            Parses the input file and stores the result in the `base` attribute.
    """

    keys = ['!Number of transitions',
            '!Number of T values OMEGA tabulated at',
            '!Scaling factor for OMEGA (non-FILE values)',
            '!Value for OMEGA if f=0',
            ]

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        skiprows = find_row(fname, "ransition\T", row_number=True)
        kwargs = {'header': None,
                  'index_col': False,
                  'sep': '\s*-?\s+-?|(?<=[^edED])-|(?<=[FDP]e)-',
                  'skiprows': skiprows}

        # FIXME: expensive solution for two files containing more than one 
        # table: `ARG/III/19nov07/col_ariii` & `HE/II/5dec96/he2col.dat`
        end = find_row(fname, "Johnson values:", 
                        "dln_OMEGA_dlnT", how='OR', row_number=True)
        if end is not None:
            kwargs['nrows'] = end - kwargs['skiprows'] -2

        try:
            columns = find_row(fname, 'ransition\T').split()
            
            # NOTE: Comment next line when trying new regexes
            columns = [np.format_float_scientific(
                to_float(x)*1e+04, precision=4) for x in columns[1:]]
            kwargs['names'] = ['label_lower', 'label_upper'] + columns

        # FIXME: some files have no column names nor header
        except AttributeError:
            logger.warning(f'Unknown column format: `{fname}`.')

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')
            for c in df.columns[2:]:  # This is done column-wise on purpose
                try:
                    df[c] = df[c].astype('float64')

                except ValueError:
                    df[c] = df[c].map(to_float).astype('float64')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
            logger.warning(f'Empty table: `{fname}`.')

        self.base = df
        self.columns = df.columns.tolist()
        self.fname = fname
        self.meta = meta

    def to_hdf(self, key='/collisional_strengths'):
        if not self.base.empty:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:
                f.put(key, self.base)
                f.get_storer(key).attrs.metadata = self.meta


class CMFGENPhotoionizationCrossSectionParser(BaseParser):
    """
        Description
        ----------
        base : list of pandas.DataFrame
        columns : list of str
        meta : dict
            Metadata extracted from file header.

        Methods
        -------
        load(fname)
            Parses the input file and stores the result in the `base` attribute.
    """

    keys = ['!Date',
            '!Number of energy levels',
            '!Number of photoionization routes',
            '!Screened nuclear charge',
            '!Final state in ion',
            '!Excitation energy of final state',
            '!Statistical weight of ion',
            '!Cross-section unit',
            '!Split J levels',
            '!Total number of data pairs',
            ]

    def _table_gen(self, f):
        """Yields a cross section table for a single energy level target.

        Parameters
        ----------
        f : file buffer

        Yields
        -------
        pd.DataFrame
            DataFrame with attached metadata.
        """
        data = []
        meta = {}

        for line in f:
            
            try:
                value = line.split()[0]

            except IndexError:
                continue

            if '!Configuration name' in line:
                meta['Configuration name'] = value

            if '!Type of cross-section' in line:
                meta['Type of cross-section'] = int(value)

            if '!Number of cross-section points' in line:
                n_points = int(value)
                for i in range(n_points):

                    values = f.readline().split()
                    if len(values) == 8:  # Verner & Yakolev (1995) ground state fits

                        data.append(
                            list(map(int, values[:2])) + list(map(to_float, values[2:])))

                        if i == n_points/len(values) -1:
                            break

                    else:
                        data.append(map(to_float, values))

                meta['Number of cross-section points'] = n_points
                break

        arr = np.array(data)
        yield arr, meta

    def load(self, fname):

        data = []
        column_types = set()
        meta = parse_header(fname, self.keys)

        with open_cmfgen_file(fname) as f:

            while True:

                arr, meta_ = next(self._table_gen(f), None)
                df = pd.DataFrame.from_records(arr)
                df.attrs = meta_

                if df.empty:
                    break
                
                elif df.shape[1] == 2:
                    columns = ['energy', 'sigma']

                elif df.shape[1] == 1:
                    columns = ['fit_coeff']

                elif df.shape[1] == 8:  # Verner & Yakolev (1995) ground state fits
                    columns = ['n', 'l', 'E', 'E_0', 'sigma_0', 'y(a)', 'P', 'y(w)']

                else:
                    logger.warning(f'Unknown column format: `{fname}`.')

                column_types.add(tuple(columns))
                df.columns = columns
                data.append(df)

        self.base = data
        self.columns = sorted(column_types)
        self.fname = fname
        self.meta = meta

    def to_hdf(self, key='/photoionization_cross_sections'):
        if len(self.base) > 0:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:

                for i in range(0, len(self.base)-1):
                    subkey = '{0}/{1}'.format(key, i)
                    f.put(subkey, self.base[i])
                    f.get_storer(subkey).attrs.metadata = self.base[i].attrs

                f.root._v_attrs['metadata'] = self.meta


class CMFGENHydLParser(BaseParser):
    """
    Parser for the CMFGEN hydrogen photoionization cross sections.

    Attributes
    ----------
    base : pandas.DataFrame, dtype float
        Photoionization cross section table for hydrogen. Values are the
        common logarithm (i.e. base 10) of the cross section in units cm^2.
        Indexed by the principal quantum number n and orbital quantum
        number l.
    columns : list of float
        The frequencies for the cross sections. Given in units of the threshold
        frequency for photoionization.
    meta : dict
        Metadata extracted from file header.

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    keys = [
        '!Maximum principal quantum number',
        '!Number of values per cross-section',
        '!L_ST_U',
        '!L_DEL_U'
    ]
    nu_ratio_key = 'L_DEL_U'

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        self.meta = meta
        self.max_l = self.get_max_l()

        self.num_xsect_nus = int(meta['Number of values per cross-section'])
        nu_ratio = 10**float(meta[self.nu_ratio_key])
        nus = np.power(
            nu_ratio,
            np.arange(self.num_xsect_nus)
        )  # in units of the threshold frequency

        skiprows = find_row(fname, self.nu_ratio_key, row_number=True) + 1

        data = []
        indexes = []
        with open(fname, mode='r') as f:
            for i in range(skiprows):
                f.readline()
            while True:
                n, l, log10x_sect = next(self._table_gen(f), None)
                indexes.append((n, l))
                data.append(log10x_sect)
                if l == self.max_l:
                    break

        index = pd.MultiIndex.from_tuples(indexes, names=['n', 'l'])
        self.base = pd.DataFrame(data, index=index, columns=nus)
        self.base.columns.name = 'nu / nu_0'

        # self.base -= 10.  # Convert from cmfgen units to log10(cm^2)
        self.columns = self.base.columns.tolist()
        self.fname = fname

    def _table_gen(self, f):
        """Yields a logarithmic cross section table for a hydrogen level.

        Parameters
        ----------
        f : file buffer

        Yields
        -------
        int
            Principal quantum number n.
        int
            Principal quantum number l.
        numpy.ndarray, dtype float
            Photoionization cross section table. Values are the common
            logarithm (i.e. base 10) of the cross section in units cm^2.
        """
        line = f.readline()
        n, l, num_entries = self.parse_table_header_line(line)
        assert(num_entries == self.num_xsect_nus)

        log10_xsect = []
        while True:
            line = f.readline()
            if not line.strip():  # This is the end of the current table
                break
            log10_xsect += [float(entry) for entry in line.split()]

        log10_xsect = np.array(log10_xsect)
        assert(len(log10_xsect) == self.num_xsect_nus)

        yield n, l, log10_xsect

    @staticmethod
    def parse_table_header_line(line):
        return [int(entry) for entry in line.split()]

    def get_max_l(self):
        return int(self.meta['Maximum principal quantum number']) - 1

    def to_hdf(self, key='/hyd_l_data'):
        if not self.base.empty:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:
                f.put(key, self.base)
                f.get_storer(key).attrs.metadata = self.meta


class CMFGENHydGauntBfParser(CMFGENHydLParser):
    """
    Parser for the CMFGEN hydrogen bound-free gaunt factors.

    Attributes
    ----------
    base : pandas.DataFrame, dtype float
        Bound-free gaunt factors for hydrogen.
        Indexed by the principal quantum number n.
    columns : list of float
        The frequencies for the gaunt factors. Given in units of the threshold
        frequency for photoionization.
    meta : dict
        Metadata extracted from file header.

    Methods
    -------
    load(fname)
        Parses the input file and stores the result in the `base` attribute.
    """

    keys = [
        "!Maximum principal quantum number",
        "!Number of values per cross-section",
        "!N_ST_U",
        "!N_DEL_U",
    ]
    nu_ratio_key = "N_DEL_U"

    @staticmethod
    def parse_table_header_line(line):
        line_split = [int(entry) for entry in line.split()]
        n, l, num_entries = (
            line_split[0],
            line_split[0],
            line_split[1],
        )  # use n as mock l value
        return n, l, num_entries

    def load(self, fname):
        super().load(fname)
        self.base.index = self.base.index.droplevel("l")
        # self.base += 10.0  # undo unit conversion used in CMFGENHydLParser

    def get_max_l(self):
        return int(self.meta["Maximum principal quantum number"])

    def to_hdf(self, key="/gbf_n_data"):
        super().to_hdf(key)


class CMFGENReader:
    """
    Class for extracting levels and lines from CMFGEN.
    
    Mimics the GFALLReader class.

    Attributes
    ----------
    levels : DataFrame
    lines : DataFrame

    """

    CMFGEN_DICT = {
        'H': 'HYD', 'He': 'HE', 'C': 'CARB', 'N': 'NIT',
        'O': 'OXY', 'F': 'FLU', 'Ne': 'NEON', 'Na': 'NA',
        'Mg': 'MG', 'Al': 'AL', 'Si': 'SIL', 'P': 'PHOS',
        'S': 'SUL', 'Cl': 'CHL', 'Ar': 'ARG', 'K': 'POT',
        'Ca': 'CA', 'Sc': 'SCAN', 'Ti': 'TIT', 'V': 'VAN',
        'Cr': 'CHRO', 'Mn': 'MAN', 'Fe': 'FE', 'Co': 'COB',
        'Ni': 'NICK'
    }

    def __init__(self, data, priority=10):
        """
        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per species with 
            keys `levels` and `lines`.

        priority: int, optional
            Priority of the current data source, by default 10.
        """
        self.priority = priority
        self.ions = list(data.keys())
        self._get_levels_lines(data)

    @classmethod
    def from_config(cls, ions, atomic_path, phixs=False, config_yaml=None,):

        ATOMIC_PATH = pathlib.Path(atomic_path)
        if config_yaml is not None:
            YAML_PATH = pathlib.Path(config_yaml).as_posix()

        else:
            YAML_PATH = pathlib.Path(CARSUS_PATH[0]).joinpath('data', 'config.yml').as_posix()
            
        data = {}
        with open(YAML_PATH) as f:
            conf = yaml.load(f, Loader=yaml.FullLoader)
            ions = parse_selected_species(ions)

            for ion in ions:
                sym = convert_atomic_number2symbol(ion[0])

                try:
                    ion_keys = conf['atom'][sym]['ion_number'][ion[1]]
                    BASE_PATH = ATOMIC_PATH.joinpath(cls.CMFGEN_DICT[sym],
                                                     roman.toRoman(ion[1]+1),
                                                     ion_keys['date'])

                except KeyError:
                    logger.warning(f'No configuration found for {sym} {ion[1]}.')
                    continue

                osc_fname = BASE_PATH.joinpath(ion_keys['osc']
                                                ).as_posix()

                data[ion] = {}
                lvl_parser = CMFGENEnergyLevelsParser(osc_fname)
                lns_parser = CMFGENOscillatorStrengthsParser(osc_fname)
                data[ion]['levels'] = lvl_parser.base
                data[ion]['lines'] = lns_parser.base

                if phixs:
                    pho_flist = []
                    try:
                        for j, k in enumerate(ion_keys['pho']):
                            pho_fname = BASE_PATH.joinpath(ion_keys['pho'][j]).as_posix()
                            pho_flist.append(pho_fname)

                    except KeyError:
                        logger.warning(f'No `pho` data for {sym} {ion[1]}.')

                    data[ion]['phixs'] = []
                    for l in pho_flist:
                        pho_parser = CMFGENPhotoionizationCrossSectionParser(l)
                        data[ion]['phixs'].append(pho_parser.base)

                    if ion == (1,0):
                        hyd_fname = BASE_PATH.joinpath('hyd_l_data.dat').as_posix()
                        gbf_fname = BASE_PATH.joinpath('gbf_n_data.dat').as_posix()

                        hyd_parser = CMFGENHydLParser(hyd_fname)
                        gbf_parser = CMFGENHydGauntBfParser(gbf_fname)

                        data[ion]['hyd'] = hyd_parser.base
                        data[ion]['gbf'] =  gbf_parser.base

        return cls(data)

    def _get_levels_lines(self, data):
        """ Generates `levels` and `lines` DataFrames.

        Parameters
        ----------
        data : dict
            Dictionary containing one dictionary per specie with 
            keys `levels` and `lines`.
        """
        lvl_list = []
        lns_list = []
        for ion, parser_base in data.items():

            atomic_number = ion[0]
            ion_charge = ion[1]
            
            lvl = parser_base['levels']

            # some ID's have negative values (theoretical?)
            lvl.loc[ lvl['ID'] < 0, 'method'] = 'theor'
            lvl.loc[ lvl['ID'] > 0, 'method'] = 'meas'
            lvl['ID'] = np.abs(lvl['ID'])
            lvl_id = lvl.set_index('ID')
            lvl['atomic_number'] = atomic_number
            lvl['ion_charge'] =  ion_charge 
            lvl_list.append(lvl)

            lns = parser_base['lines']
            lns = lns.set_index(['i', 'j'])
            lns['energy_lower'] = lvl_id['E(cm^-1)'].reindex(lns.index, level=0).values
            lns['energy_upper'] = lvl_id['E(cm^-1)'].reindex(lns.index, level=1).values
            lns['g_lower'] = lvl_id['g'].reindex(lns.index, level=0).values
            lns['g_upper'] = lvl_id['g'].reindex(lns.index, level=1).values
            lns['j_lower'] = (lns['g_lower'] -1)/2
            lns['j_upper'] = (lns['g_upper'] -1)/2
            lns['atomic_number'] = atomic_number
            lns['ion_charge'] = ion_charge
            lns = lns.reset_index()
            lns_list.append(lns)

        levels = pd.concat(lvl_list)
        levels['priority'] = self.priority
        levels = levels.reset_index(drop=False)
        levels = levels.rename(columns={'Configuration': 'label', 
                                        'E(cm^-1)': 'energy', 
                                        'index': 'level_index'})
        levels['j'] = (levels['g'] -1) / 2
        levels = levels.set_index(['atomic_number', 'ion_charge', 'level_index'])
        levels = levels[['energy', 'j', 'label', 'method', 'priority']]
        
        lines = pd.concat(lns_list)
        lines = lines.rename(columns={'Lam(A)': 'wavelength'})
        lines['wavelength'] = u.Quantity(lines['wavelength'], u.AA).to('nm').value
        lines['level_index_lower'] = lines['i'] -1
        lines['level_index_upper'] = lines['j'] -1
        lines['gf'] = lines['f'] * lines['g_lower']
        lines = lines.set_index(['atomic_number', 'ion_charge', 
                                 'level_index_lower', 'level_index_upper'])
        lines = lines[['energy_lower', 'energy_upper', 
                       'gf', 'j_lower', 'j_upper', 'wavelength']]

        self.levels = levels
        self.lines = lines
        self.data = data

        return
