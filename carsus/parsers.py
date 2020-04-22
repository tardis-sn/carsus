"""
=====================
TARDIS parsers module
=====================
created on April 12, 2020
"""

import pandas as pd
import os
import numpy as np
import urllib.request, shutil
import tarfile
import re
import h5py


def OSC_Parser(path):
        """
        Reads and process the OSC file. It stores the output information in a HDF5
        file format.
        
        Parameters
        ----------
        path: String
            path where the file is located
            
        Returns
        ----------
        Dataframes: Dictionary
            dataframe as value and number as key
        
        Metadata: Dictionary
            MetaData as value and dataframe number as key
        """
        metadata = {}
        META_PATTERN = r"^([\w\-\.]+)\s+!.*"
        Dataframe_PATTERN = r"\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])"
        
        with open(path, encoding='ISO-8859-1') as f:
            n = 1
            for line in f:
                if re.match(META_PATTERN,line):
                    val = line.split("!")
                    val = [word.strip() for word in val]
                    metadata[val[1]] = val[0]
                    
                if "Number of transitions" in line:
                    break
                n += 1
            
            else:
                n = None
                line = None
        
        Energy_Table = pd.read_csv(path, header = None, index_col = False, 
                                   sep = Dataframe_PATTERN ,skiprows = n, 
                                   nrows = int(metadata['Number of energy levels']),
                                   engine='python')
        
        if Energy_Table.shape[1] == 10:
            columns = ['Configuration', 'g', 'E(cm^-1)', '10^15 Hz', 'eV', 
                       'Lam(A)', 'ID', 'ARAD', 'C4', 'C6']
            Energy_Table.columns = columns
    
        elif Energy_Table.shape[1] == 7:
            Energy_Table.columns = ['Configuration', 'g', 'E(cm^-1)', 'eV',
                                    'Hz 10^15', 'Lam(A)', '?']
            Energy_Table = Energy_Table.drop(columns=['?'])
    
        elif Energy_Table.shape[1] == 6:
            Energy_Table.columns = ['Configuration', 'g',
                                    'E(cm^-1)', 'Hz 10^15', 'Lam(A)', '?']
            Energy_Table = Energy_Table.drop(columns=['?'])
    
        elif Energy_Table.shape[1] == 5:
            Energy_Table.columns = ['Configuration', 'g', 'E(cm^-1)', 'eV','?']
            Energy_Table = Energy_Table.drop(columns=['?'])
            
        with open(path, encoding='ISO-8859-1') as f:
            n = 1
            for line in f:
                if "i-j" in line:
                    break
                n += 1
            
            else:
                n = None
                line = None
                
        Transition_Table = pd.read_csv(path, header = None, index_col = False,
                                       skiprows = n, sep = Dataframe_PATTERN, 
                                       nrows = int(metadata["Number of transitions"]),
                                       engine='python')

        if Transition_Table.shape[1] == 9:
            Transition_Table.columns = ['State A', 'State B', 'f', 'A','Lam(A)',
                                        'i', 'j', 'Lam(obs)', '% Acc']

        elif Transition_Table.shape[1] == 10:
            Transition_Table.columns = ['State A', 'State B', 'f', 'A', 'Lam(A)',
                                        'i', 'j', 'Lam(obs)', '% Acc','?']
            Transition_Table = Transition_Table.drop(columns=['?'])

        elif Transition_Table.shape[1] == 8:
            Transition_Table.columns = ['State A', 'State B', 'f', 'A',
                                        'Lam(A)', 'i', 'j','?']
            Transition_Table = Transition_Table.drop(columns=['?'])
            Transition_Table['Lam(obs)'] = np.nan
            Transition_Table['% Acc'] = np.nan
        
        Dataframes = {}
        Dataframes[0] = Energy_Table
        Dataframes[1] = Transition_Table
        
        Metadata = {}
        Metadata[0] = metadata
        Metadata[1] = metadata
        
        return Dataframes, Metadata
    
    
def COL_Parser(path):
    """
    Reads and process the OSC file. It stores the output information in a HDF5
    file format.
    
    Parameters
    ----------
    path: String
        path where the file is located
        
    Returns
    ----------
    Dataframes: Dictionary
        dataframe as value and number as key
    
    Metadata: Dictionary
        MetaData as value and dataframe number as key
    """
    metadata = {}
    META_PATTERN = r"^([\w\-\.]+)\s+!.*"
    Dataframe_PATTERN = r"\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])"
    
    with open(path, encoding='ISO-8859-1') as f:
        n = 1
        flag = 1
        col_name = []
        for line in f:
            if re.match(META_PATTERN,line):
                val = line.split("!")
                val = [word.strip() for word in val]
                metadata[val[1]] = val[0]
                
            if "Transition\T" in line:
                flag = 0
                col_name = line.split()
                
            if flag==1:
                n += 1
                
    Collisional_Table = pd.read_csv(path, header = None, index_col = False, 
                               sep = Dataframe_PATTERN ,skiprows = n, 
                               names = ['State A', 'State B'] + col_name[1:],
                               nrows = int(metadata['Number of transitions']),
                               engine='python')
    
    Dataframes = {}
    Dataframes[0] = Collisional_Table
    
    Metadata = {}
    Metadata[0] = metadata
    
    return Dataframes, Metadata
    

def PHOT_Parser(path):
    """
    Reads and process the OSC file. It stores the output information in a HDF5
    file format.
    
    Parameters
    ----------
    path: String
        path where the file is located
        
    Returns
    ----------
    Dataframes: Dictionary
        dataframe as value and number as key
    
    Metadata: Dictionary
        MetaData as value and dataframe number as key
    """
    metadata = {}
    META_PATTERN = r"^([\w\-\.]+)\s+!.*"
    Dataframe_PATTERN = r"\s*\|\s*|-?\s+-?\s*|(?<=[^ED\s])-(?=[^\s])"
    
    Dataframes = {}
    Metadata = {}
    
    with open(path, encoding='ISO-8859-1') as f:
        n = 1
        tables = 0
        for line in f:
            if re.match(META_PATTERN,line):
                val = line.split("!")
                val = [word.strip() for word in val]
                metadata[val[1]] = val[0]
                
            if "Number of cross-section points" in line:
        
                Phot_Table = pd.read_csv(path, header = None, index_col = False, 
                           sep = Dataframe_PATTERN ,skiprows = n,
                           nrows = int(metadata['Number of cross-section points']),
                           engine='python')
                
                if Phot_Table.shape[1] == 2:
                    Phot_Table.columns = ['Energy', 'Sigma']
            
                elif Phot_Table.shape[1] == 1:
                    Phot_Table.columns = ['Fit coefficients']
            
                elif Phot_Table.shape[1] == 8:  # Verner ground state fits. TODO: add units
                    Phot_Table.columns = ['n', 'l', 'E', 'E_0','sigma_0', 'y(a)', 'P', 'y(w)']
                    
                Dataframes[tables] = Phot_Table
                Metadata[tables] = metadata
                
                metadata = {}
                    
                tables += 1
                
            n += 1
            
    return Dataframes, Metadata
