import os
import numpy as np
import pandas as pd
import carsus
from carsus.util import convert_symbol2atomic_number
from carsus.util.helpers import ATOMIC_SYMBOLS_DATA
from carsus.io.nist import (download_ionization_energies,
                            NISTIonizationEnergiesParser
                            )
from carsus.io.nist.weightscomp_grammar import isotope, COLUMNS, ATOM_NUM_COL, MASS_NUM_COL,\
    AM_VAL_COL, AM_SD_COL, INTERVAL, STABLE_MASS_NUM, ATOM_WEIGHT_COLS, AW_STABLE_MASS_NUM_COL,\
    AW_TYPE_COL, AW_VAL_COL, AW_SD_COL, AW_LWR_BND_COL, AW_UPR_BND_COL
from carsus.io.nist import (download_weightscomp,
                            NISTWeightsCompPyparser)


basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data',
                                       'basic_atomic_data.csv')

class NISTIonizationEnergies:
    def __init__(self, spectra):
        input_data = download_ionization_energies(spectra)
        self.nist_parser = NISTIonizationEnergiesParser(input_data)
        self._prepare_data()
    
    def _prepare_data(self):
        self.ioniz_energies = pd.DataFrame()
        self.ioniz_energies['atomic_number'] = self.nist_parser.base['atomic_number']
        self.ioniz_energies['ion_number'] = self.nist_parser.base['ion_charge'] + 1
        self.ioniz_energies['ionization_energy'] = self.nist_parser.base['ionization_energy_str'].str.strip('[]()').astype(np.float64)
        self.ioniz_energies.set_index(['atomic_number', 'ion_number'], inplace=True)

        # Convert DataFrame to Series
        self.ioniz_energies = self.ioniz_energies['ionization_energy']

    def to_hdf(self, fname):
        with pd.HDFStore(fname, 'a') as f:
            f.append('/ionization_data', self.ioniz_energies)


class NISTWeightsComp:
    def __init__(self, atom):
        input_data = download_weightscomp()
        self.nist_parser = NISTWeightsCompPyparser(input_data=input_data)
        self._prepare_data(atom)

    def _prepare_data(self, atom):
        atomic_number = convert_symbol2atomic_number(atom)
        basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
        basic_atomic_data = basic_atomic_data.loc[atomic_number-1]

        atom_masses = self.nist_parser.prepare_atomic_dataframe()
        atom_masses = atom_masses.drop(columns='atomic_weight_std_dev')
        atom_masses = atom_masses.rename(columns={'atomic_weight_nominal_value': 'mass'})
        
        atom_data = atom_masses.loc[[(atomic_number)]]
        atom_data['symbol'] = basic_atomic_data['symbol']
        atom_data['name'] = basic_atomic_data['name']
        self.atom_data = atom_data[['symbol', 'name', 'mass']]


    def to_hdf(self, fname):
        with pd.HDFStore(fname, 'a') as f:
            f.append('/atom_data', self.atom_data, min_itemsize={'symbol': 2, 'name': 15})


class KnoxLongZetaData:  
    def __init__(self, fname):
        self.fname = fname
        self._prepare_data()

    def _prepare_data(self):
        t_values = np.arange(2000, 42000, 2000)

        names = ['atomic_number', 'ion_charge']
        names += [str(i) for i in t_values]

        zeta_raw = np.recfromtxt(
                self.fname,
                usecols=range(1, 23),
                names=names)

        self.zeta_data = (
                pd.DataFrame(zeta_raw).set_index(
                    ['atomic_number', 'ion_charge'])
                )

        columns = [float(c) for c in self.zeta_data.columns]

        # To match exactly the `old` format
        self.zeta_data.columns = pd.Float64Index(columns, name='temp')

    def to_hdf(self, fname):
        with pd.HDFStore(fname, 'a') as f:
            f.append('/zeta_data', self.zeta_data)