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
