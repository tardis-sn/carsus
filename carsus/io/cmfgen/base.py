import numpy as np
import pandas as pd
import itertools
import gzip
import warnings
from carsus.io.base import BaseParser


def find_row(fname, string1, string2='', how='both', num_row=False):  # TODO: add `skiprows` parameter
    """Search strings inside plain text files and returns matching line or row number.
    
    Parameters
    ----------
    fname : str
        Path to plain text file.
    string1 : str
        String to search.    
    string2 : str
        Extra string to search (default is '').
    how : {'one', 'both', 'first'}
        If 'both' search for string1 AND string2. If 'one' search for string1 OR string2.\
            If 'first' searches for 'string1' AND NOT string2 (default is 'both').
    num_row : bool
        If true, returns row number instead (default is False).

    Returns
    -------
    str or int
        Returns matching line or row number.

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
    
    if num_row == True:
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
    with gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname, encoding='ISO-8859-1') as f :
        for line in itertools.islice(f, start, stop):  # start=17, stop=None
            for k in keys:
                if k.lower() in line.lower():
                    meta[k.strip('!')] = line.split()[0]

    return meta


def to_float(string):
    """ String to float, taking care of Fortran 'D' values """
    try:
        value = float(string.replace('D', 'E'))
    
    except:
        
        if string == '1-.00':      # Bad value at MG/VIII/23oct02/phot_sm_3000 line 23340
            value = 10.00
        
        if string == '*********':  # Bad values at SUL/V/08jul99/phot_op.big lines 9255-9257
            value = np.nan
            
    return value


class CMFGENEnergyLevelsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
            (default value = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'Hz 10^15', 'Lam(A)'])
        meta : dict 
            Metadata parsed from file header.

        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
    """

    keys = ['!Date',  # Metadata to parse from header. TODO: look for more keys
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
        kwargs['skiprows'] = find_row(fname, "Number of transitions", num_row=True)
        
        n = int(meta['Number of energy levels'])
        kwargs['nrows'] = n
        
        columns = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'Hz 10^15', 'Lam(A)']
        
        try:
            df = pd.read_csv(fname, **kwargs, engine='python')
    
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            warnings.warn('Empty table')

        # Assign column names by file content
        if df.shape[1] == 10:
            columns = find_row(fname, 'E(cm^-1)', "Lam").split('  ')
            columns = list(filter(lambda x: x != '', columns))
            columns = ['Configuration'] + columns
            df.columns = columns

        elif df.shape[1] == 7:
            df.columns = columns + ['#']
            df = df.drop(columns=['#'])

        elif df.shape[1] == 6:
            df.columns = range(6)  # FIXME: These files don't have column names

        elif df.shape[1] == 5:
            df.columns = columns[:-2] + ['#']
            df = df.drop(columns=['#'])
    
        else:
            warnings.warn('Inconsistent number of columns')  # TODO: raise exception here (discuss)

        assert df.shape[0] == n

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
        self.meta = meta


class CMFGENOscillatorStrengthsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
            (default value = ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc'])
        meta : dict 
            Metadata parsed from file header.
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
    """

    keys = CMFGENEnergyLevelsParser.keys

    def load(self, fname):
        meta = parse_header(fname, self.keys)    
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*\|\s*|-?\s+-?|(?<=[^ED\s])-(?=[^\s])'
        kwargs['skiprows'] = find_row(fname, "Transition", "Lam", num_row=True) +1

        # Will only parse tables listed increasing lower level i, e.g. FE/II/24may96/osc_nahar.dat
        n = int(meta['Number of transitions'])  
        kwargs['nrows'] = n

        columns = ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    
        try:
            df = pd.read_csv(fname, **kwargs, engine='python')
    
        except pd.errors.EmptyDataError:
            df = pd.DataFrame(columns=columns)
            warnings.warn('Empty table')
    
        # Assign column names by file content
        if df.shape[1] == 9:
            df.columns = columns
        
        elif df.shape[1] == 10:
            df.columns = columns + ['?']
            df = df.drop(columns=['?'])
    
        elif df.shape[1] == 8:
            df.columns = columns[:-2] + ['#']
            df = df.drop(columns=['#'])
            df['Lam(obs)'] = np.nan
            df['% Acc'] = np.nan
    
        else:
            warnings.warn('Inconsistent number of columns')       
    
        # Fix for Fortran float type 'D'
        if df.shape[0] > 0 and 'D' in str(df['f'][0]):
            df['f'] = df['f'].map(to_float)
            df['A'] = df['A'].map(to_float)

        assert df.shape[0] == n

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
        self.meta = meta


class CMFGENCollisionalDataParser(BaseParser):
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
            Parses the input data and stores the results in the `base` attribute
    """

    keys = ['!Number of transitions',  # Metadata to parse from header. TODO: look for more keys
            '!Number of T values OMEGA tabulated at', 
            '!Scaling factor for OMEGA (non-FILE values)', 
            '!Value for OMEGA if f=0',
            ]

    def load(self, fname):
        meta = parse_header(fname, self.keys)       
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*-?\s+-?|(?<=[^edED])-|(?<=Pe)-'  # TODO: this regex needs some review
        kwargs['skiprows'] = find_row(fname, "ransition\T", num_row=True)

        # FIXME: expensive solution for two files with more than one table 
        # ARG/III/19nov07/col_ariii  &  HE/II/5dec96/he2col.dat
        footer = find_row(fname, "Johnson values:", "dln_OMEGA_dlnT", how='one', num_row=True)
        if footer is not None:
            kwargs['nrows'] = footer -kwargs['skiprows'] -2

        try:
            names = find_row(fname, 'ransition\T').split()  # Not a typo
            # Comment next line when trying new regexes!
            names = [ np.format_float_scientific(to_float(x)*1e+04, precision=4) for x in names[1:] ]
            kwargs['names'] = ['State A', 'State B'] + names[1:]

        except AttributeError:
            warnings.warn('No column names')  # TODO: some files have no column names nor header
    
        try:
            df = pd.read_csv(fname, **kwargs, engine='python')
            for c in df.columns[2:]:          # This is done column-wise on purpose
                try:
                    df[c] = df[c].astype('float64')

                except ValueError:
                    df[c] = df[c].map(to_float).astype('float64')

        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
            warnings.warn('Empty table')

        try:
            n = int(meta['Number of transitions'])
            if df.shape[0] != n:
                # Sometimes header values can't be trusted, e.g: NEON/III/19nov07/col_neiii 
                # so we only display a warn instead of making an assertion
                warnings.warn('`Number of transitions` and DataFrame dimension don\'t match')
        
        except TypeError:  # int(None) triggers a TypeError
            warnings.warn('File without header')

        self.fname = fname
        self.base = df
        self.columns = df.columns.tolist()
        self.meta = meta

class CMFGENPhotoionizationCrossSectionParser(BaseParser):
    """
        Description
        ----------
        base : list of pandas.DataFrame 's
        columns : list of str
        meta : dict 
            Metadata parsed from file header.
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
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
        """ Generator. Yields a cross section table for an energy level """
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
                    if len(values) == 8:  # Verner ground state fits

                        data.append(list(map(int, values[:2])) + list(map(float, values[2:])))

                        if i == p/len(values) -1:
                            break

                    else:
                        data.append(map(to_float, values))

                break

        df = pd.DataFrame.from_records(data)  # TODO: make assertions for the yielded DataFrame
        df._meta = meta    

        yield df


    def load(self, fname):
    
        meta = parse_header(fname, self.keys)
        tables = []
        with gzip.open(fname, 'rt') if fname.endswith('.gz') else open(fname) as f :

            while True:

                df = next(self._table_gen(f), None)

                if df.empty:
                    break

                if df.shape[1] == 2:
                    df.columns = ['Energy', 'Sigma']

                elif df.shape[1] == 1:
                    df.columns = ['Fit coefficients']

                elif df.shape[1] == 8:  # Verner ground state fits. TODO: add units
                    df.columns = ['n', 'l', 'E', 'E_0', 'sigma_0', 'y(a)', 'P', 'y(w)']

                else:
                    warnings.warn('Inconsistent number of columns')

                tables.append(df)

        tables.insert(0, meta)  # TODO: assert the `tables` list length

        self.fname = fname
        self.base = tables[1:]
        self.columns = []
        self.meta = tables[0]