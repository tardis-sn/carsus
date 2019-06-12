import numpy as np
import pandas as pd
import warnings
from carsus.io.base import BaseParser


def find_row(fname, string1, string2='', how='both', num_row=False):
    """  """
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


to_float = lambda x: float(x.replace('D', 'E'))  # string to float, taking care of Fortran 'D' values


class CMFGENEnergyLevelsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
            (default value = COLUMNS)
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
        to_hdf(fname, key)
            Saves `base` attribute to HDF5 file.
    """

    def load(self, fname):
        args = {}
        args['header'] = None
        args['index_col'] = False
        args['sep'] = '\s+'
        args['skiprows'] = find_row(fname, "Number of transitions", num_row=True)
    
        n = find_row(fname, "Number of energy levels").split()[0]
        args['nrows'] = int(n)

        columns = ['Configuration', 'g', 'E(cm^-1)', 'eV', 'Hz 10^15', 'Lam(A)']
        
        try:
            df = pd.read_csv(fname, **args, engine='python')
    
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
            warnings.warn('Inconsistent number of columns')

        self.base = df


    def to_hdf(self, fname, key):
        self.base.to_hdf(fname, key)


class CMFGENOscillatorStrengthsParser(BaseParser):
    """
        Description
        ----------
        base : pandas.DataFrame
        columns : list of str
            (default value = COLUMNS)
            
        Methods
        -------
        load(fname)
            Parses the input data and stores the results in the `base` attribute
        to_hdf(fname, key)
            Saves `base` attribute to HDF5 file.
    """

    def load(self, fname):
        args = {}
        args['header'] = None
        args['index_col'] = False
        args['sep'] = '\s*\|\s*|-?\s+-?|(?<=[^ED\s])-(?=[^\s])'
        args['skiprows'] = find_row(fname, "Transition", "Lam", num_row=True) +1
    
        n = find_row(fname, "Number of transitions").split()[0]
        args['nrows'] = int(n)

        columns = ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i', 'j', 'Lam(obs)', '% Acc']
    
        try:
            df = pd.read_csv(fname, **args, engine='python')
    
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


    def to_hdf(self, fname, key):
        self.base.to_hdf(fname, key)