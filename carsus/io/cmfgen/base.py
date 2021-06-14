import logging
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
import itertools
import gzip
from carsus.io.base import BaseParser
from carsus.util import parse_selected_species

logger = logging.getLogger(__name__)


# TODO: add `skiprows` parameter
def find_row(fname, string1, string2='', how='both', num_row=False):
    """Search strings inside plain text files and returns matching\
    line or row number.

    Parameters
    ----------
    fname : str
        Path to plain text file.
    string1 : str
        String to search.
    string2 : str
        Extra string to search (default is '').
    how : {'one', 'both', 'first'}
        If 'both' search for string1 AND string2. If 'one' search for string1\
            OR string2. If 'first' searches for 'string1' AND NOT string2\
            (default is 'both').
    num_row : bool
        If true, returns row number instead (default is False).

    Returns
    -------
    str or int
        Returns matching line or match row number.
    """
    with open(fname, encoding='ISO-8859-1') as f:
        n = 0
        for line in f:
            n += 1
            if how == 'one':
                if string1 in line or string2 in line:
                    break

            if how == 'both':
                if string1 in line and string2 in line:
                    break

            if how == 'first':
                if string1 in line and string2 not in line:
                    break

        # In case there's no match
        else:
            n = None
            line = None

    if num_row is True:
        return n

    return line


def parse_header(fname, keys, start=0, stop=50):
    """Parse header from CMFGEN files.

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

    with gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname, encoding='ISO-8859-1') as f:
        for line in itertools.islice(f, start, stop):
            for k in keys:
                if k.lower() in line.lower():
                    meta[k.strip('!')] = line.split()[0]

    return meta


def to_float(string):
    """ String to float, useful to work with Fortran 'D' type.

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
        # Weird value in `MG/VIII/23oct02/phot_sm_3000`, line 23340
        if string == '1-.00':
            value = 10.00

        # Weird values in `SUL/V/08jul99/phot_op.big`, lines 9255-9257
        if string == '*********':
            value = np.nan

    return value


class CMFGENEnergyLevelsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
        meta : dict
            Metadata parsed from file header.

        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute.
    """

    # Metadata to parse from header. 
    # TODO: look for more keys
    keys = ['!Date',
            '!Format date',
            '!Number of energy levels',
            '!Ionization energy',
            '!Screened nuclear charge',
            '!Number of transitions',
            ]

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s+'
        kwargs['skiprows'] = find_row(
            fname, "Number of transitions", num_row=True)

        n = int(meta['Number of energy levels'])
        kwargs['nrows'] = n

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            logger.warn(f'Table is empty: `{fname}`.')

        # Assign column names by file content
        if df.shape[1] == 10:
            # Read column names and split them keeping one space (e.g. '10^15 Hz')
            columns = find_row(fname, 'E(cm^-1)', "Lam").split('  ')
            # Filter list elements containing empty strings
            columns = [c for c in columns if c != '']
            # Remove left spaces and newlines
            columns = [c.rstrip().lstrip() for c in columns]
            columns = ['Configuration'] + columns
            df.columns = columns

        elif df.shape[1] == 7:
            df.columns = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'Hz 10^15', 'Lam(A)', 'ID']

        elif df.shape[1] == 6:
            df.columns = ['Configuration', 'g', 'E(cm^-1)', 'Hz 10^15', 'Lam(A)', 'ID']

        elif df.shape[1] == 5:
            df.columns = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'ID']

        else:
            logger.warn(f'Inconsistent number of columns: `{fname}`.')

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
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
            Metadata parsed from file header.

        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute.
    """

    keys = CMFGENEnergyLevelsParser.keys

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])'
        kwargs['skiprows'] = find_row(
            fname, "Transition", "Lam", num_row=True) + 1

        # Parse only tables listed increasing lower level i, e.g. `FE/II/24may96/osc_nahar.dat`
        n = int(meta['Number of transitions'])
        kwargs['nrows'] = n

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=['State A', 'State B', 'f', 'A',
                                 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc'])
            logger.warn(f'Table is empty: `{fname}`.')

        # Assign column names by file content
        if df.shape[1] == 9:
            df.columns = ['State A', 'State B', 'f', 'A',
                            'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']

        # These files are 9-column, but for some reason the regex produces 10 columns
        elif df.shape[1] == 10:
            df.columns = ['State A', 'State B', 'f', 'A',
                            'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc', '?']
            df = df.drop(columns=['?'])

        elif df.shape[1] == 8:
            df.columns = ['State A', 'State B', 'f', 'A', 'Lam(A)', 
                            'i', 'j', '#']
            df['Lam(obs)'] = np.nan
            df['% Acc'] = np.nan
            df = df.drop(columns=['#'])

        else:
            logger.warn(f'Inconsistent number of columns `{fname}`.')

        # Fix Fortran float type 'D'
        if df.shape[0] > 0 and 'D' in str(df['f'][0]):
            df['f'] = df['f'].map(to_float)
            df['A'] = df['A'].map(to_float)

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
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
            Metadata parsed from file header.

        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute.
    """

    keys = ['!Number of transitions',
            '!Number of T values OMEGA tabulated at',
            '!Scaling factor for OMEGA (non-FILE values)',
            '!Value for OMEGA if f=0',
            ]

    def load(self, fname):
        meta = parse_header(fname, self.keys)
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*-?\s+-?|(?<=[^edED])-|(?<=[FDP]e)-'
        kwargs['skiprows'] = find_row(fname, "ransition\T", num_row=True)

        # FIXME: expensive solution for two files with more than one table
        # `ARG/III/19nov07/col_ariii` & `HE/II/5dec96/he2col.dat`
        footer = find_row(fname, "Johnson values:",
                          "dln_OMEGA_dlnT", how='one', num_row=True)

        if footer is not None:
            kwargs['nrows'] = footer - kwargs['skiprows'] - 2

        try:
            names = find_row(fname, 'ransition\T').split()  # Not a typo!
            
            # Comment next line when trying new regexes!
            names = [np.format_float_scientific(
                to_float(x)*1e+04, precision=4) for x in names[1:]]
            kwargs['names'] = ['State A', 'State B'] + names

        except AttributeError:
            # TODO: some files have no column names nor header
            logger.warn(f'Column names not found: `{fname}`.')

        try:
            df = pd.read_csv(fname, **kwargs, engine='python')
            for c in df.columns[2:]:  # This is done column-wise on purpose
                try:
                    df[c] = df[c].astype('float64')

                except ValueError:
                    df[c] = df[c].map(to_float).astype('float64')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
            logger.warn(f'Table is empty: `{fname}`.')

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
        self.meta = meta

    def to_hdf(self, key='/collisional_strengths'):
        if not self.base.empty:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:
                f.put(key, self.base)
                f.get_storer(key).attrs.metadata = self.meta


# TODO: inherit from `BaseParser` class seems a bit forced
class CMFGENPhotoionizationCrossSectionParser(BaseParser):
    """
        Description
        ----------
        base : list of pandas.DataFrame
        columns : list of str
        meta : dict
            Metadata parsed from file header.

        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute.
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
        """Yields a cross section table for an energy level.

        Parameters
        ----------
        f : file buffer

        Yields
        -------
        pd.DataFrame
            DataFrame with metadata.
        """        
        meta = {}
        data = []

        for line in f:

            if '!Configuration name' in line:
                meta['Configuration'] = line.split()[0]

            if '!Type of cross-section' in line:
                meta['Type of cross-section'] = int(line.split()[0])

            if '!Number of cross-section points' in line:
                meta['Points'] = int(line.split()[0])

                p = meta['Points']
                for i in range(p):

                    values = f.readline().split()
                    # Verner ground state fits
                    if len(values) == 8:

                        data.append(
                            list(map(int, values[:2])) + list(map(float, values[2:])))

                        if i == p/len(values) - 1:
                            break

                    else:
                        data.append(map(to_float, values))

                break

        df = pd.DataFrame.from_records(data)
        df._meta = meta

        yield df

    def load(self, fname):

        meta = parse_header(fname, self.keys)
        tables = []
        with gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname) as f:

            while True:

                df = next(self._table_gen(f), None)

                if df.empty:
                    break

                if df.shape[1] == 2:
                    df.columns = ['Energy', 'Sigma']

                elif df.shape[1] == 1:
                    df.columns = ['Fit coefficients']

                elif df.shape[1] == 8:  # Verner ground state fits. TODO: add units
                    df.columns = ['n', 'l', 'E', 'E_0',
                                  'sigma_0', 'y(a)', 'P', 'y(w)']

                else:
                    logger.warn(f'Inconsistent number of columns: `{fname}`.')

                tables.append(df)

        self.fname = fname
        self.base = tables
        self.columns = []
        self.meta = meta

    def to_hdf(self, key='/photoionization_cross_sections'):
        if len(self.base) > 0:
            with pd.HDFStore('{}.h5'.format(self.fname), 'w') as f:

                for i in range(0, len(self.base)-1):
                    subkey = '{0}/{1}'.format(key, i)
                    f.put(subkey, self.base[i])
                    f.get_storer(subkey).attrs.metadata = self.base[i]._meta

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
        Metadata parsed from file header.

    Methods
    -------
    load(fname)
        Parses the input data and stores the results in the `base` attribute.
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

        skiprows = find_row(fname, self.nu_ratio_key, num_row=True) + 1

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

        self.base -= 10.  # Convert from cmfgen units to log10(cm^2)
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


class CMFGENReader:
    """
    Class for extracting levels and lines from CMFGEN.
    
    Mimics the GFALLReader class.

    Attributes
    ----------
    levels : DataFrame
    lines : DataFrame

    """

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
        self._get_levels_lines(data)
    
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
        for ion, parser in data.items():

            atomic_number = parse_selected_species(ion)[0][0]
            ion_charge = parse_selected_species(ion)[0][1]
            
            lvl = parser['levels'].base
            # some ID's have negative values (theoretical?)
            lvl.loc[ lvl['ID'] < 0, 'method'] = 'theor'
            lvl.loc[ lvl['ID'] > 0, 'method'] = 'meas'
            lvl['ID'] = np.abs(lvl['ID'])
            lvl_id = lvl.set_index('ID')
            lvl['atomic_number'] = atomic_number
            lvl['ion_charge'] =  ion_charge  # i.e. Si I = (14,0) then `ion_charge` = 0 
            lvl_list.append(lvl)

            lns = parser['lines'].base
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

        return
