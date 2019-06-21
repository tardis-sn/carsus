import numpy as np
import pandas as pd
import gzip
import warnings
from carsus.io.base import BaseParser


def find_row(fname, string1, string2='', how='both', num_row=False):
    """ Search strings inside plain text files and return values or matching row number. """
    with open(fname, encoding='ISO-8859-1') as f:
        n = 0
        for line in f:        
            n += 1
            
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
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
    """

    def load(self, fname):
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s+'
        kwargs['skiprows'] = find_row(fname, "Number of transitions", num_row=True)
    
        n = find_row(fname, "Number of energy levels").split()[0]
        kwargs['nrows'] = int(n)

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

        self.base = df
        self.columns = df.columns.tolist()


class CMFGENOscillatorStrengthsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
            (default value = ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc'])
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
    """

    def load(self, fname):
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*\|\s*|-?\s+-?|(?<=[^ED\s])-(?=[^\s])'
        kwargs['skiprows'] = find_row(fname, "Transition", "Lam", num_row=True) +1
    
        n = find_row(fname, "Number of transitions").split()[0]
        kwargs['nrows'] = int(n)

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

        self.base = df
        self.columns = df.columns.tolist()


class CMFGENCollisionalDataParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
    """

    def load(self, fname):
    
        kwargs = {}
        kwargs['header'] = None
        kwargs['index_col'] = False
        kwargs['sep'] = '\s*-?\s+-?|(?<=[^edED])-|(?<=Pe)-'  # TODO: this regex needs some review
        kwargs['skiprows'] = find_row(fname, "ransition\T", num_row=True)

        try:
            n = find_row(fname, "Number of transitions").split()[0]
            kwargs['nrows'] = int(n)
    
        except AttributeError:
            pass
    
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

        self.base = df
        self.columns = df.columns.tolist()


class CMFGENPhotoionizationCrossSectionParser(BaseParser):
    """
        Description
        ----------
        base : list of pandas.DataFrame 's
        columns : dictionary with photoionization cross-section file metadata
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
    """
    keys = ['!Date',  # Metadata to parse from header
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

                for i in range(meta['Points']):

                    values = f.readline().split()
                    if len(values) > 2:  # Verner ground state fits case

                        data.append(list(map(int, values[:2])) + list(map(float, values[2:])))

                        if i == meta['Points']/len(values) -1:
                            break

                    else:
                        data.append(map(to_float, values))              

                break

        df = pd.DataFrame.from_records(data)
        df._meta = meta    

        yield df


    def load(self, fname):
    
        meta = {}
        tables = []

        # There's only one .gz file: POT/I/4mar12/phot.tar.gz
        with gzip.open(fname, 'rt') if fname[-2:] == 'gz' else open(fname) as f :

            for line in f:
                for k in self.keys:

                    if k in line:
                        meta[k.strip('!')] = line.split()[0]

                        if '!Total number of data pairs' in line:
                            break


        with gzip.open(fname, 'rt') if fname[-2:] == 'gz' else open(fname) as f :

            while True:

                df = next(self._table_gen(f), None)

                if df.empty:
                    break

                if df.shape[1] == 2:
                    df.columns = ['Energy', 'Sigma']

                elif df.shape[1] == 1:
                    df.columns = ['Fit coefficients']

                elif df.shape[1] == 8:
                    df.columns = ['n', 'l', 'E', 'E_0', 'sigma_0', 'y(a)', 'P', 'y(w)']

                else:
                    warnings.warn('Inconsistent number of columns')

                tables.append(df)

        tables.insert(0, meta)
        self.base = tables[1:]
        self.columns = tables[0]