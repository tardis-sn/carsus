import numpy as np
import pandas as pd
from carsus.util import parse_selected_atoms
from carsus.io.nist import (download_ionization_energies,
                            NISTIonizationEnergiesParser,
                            )

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

        
