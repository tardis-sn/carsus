import chianti.core as ch
import pandas as pd
import numpy as np
import pickle
import os
import re
from numpy.testing import assert_almost_equal
from astropy import units as u
from carsus.model import DataSource, Ion, Level, LevelChiantiData, LevelEnergy

if os.getenv('XUVTOP'):
    masterlist_ions_path = os.path.join(
        os.getenv('XUVTOP'), "masterlist", "masterlist_ions.pkl"
    )

    masterlist_ions_file = open(masterlist_ions_path, 'rb')
    masterlist_ions = pickle.load(masterlist_ions_file).keys()
    # Exclude the "d" ions for now
    masterlist_ions = [_ for _ in masterlist_ions
                       if re.match("[a-z]+_\d+", _)]

else:
    print "Chianti database is not installed!"
    masterlist_ions = list()


class ChiantiReader(object):

    def __init__(self, ions_list):
        # ToDo write a parser for Spectral Notation
        self.ions_list = list()
        for ion in ions_list:
            if ion in self.masterlist_ions:
                self.ions_list.append(ion)
            else:
                print("Ion {0} is not available".format(ion))

    @property
    def ions(self):
        return [ch.ion(_) for _ in self.ions_list]

    masterlist_ions = masterlist_ions

    elvlc_dict = {'lvl': 'level_index',
                  'ecm': 'energy',
                  'ecmth': 'energy_theoretical',
                  'j': 'J',
                  'spd': 'L',
                  'spin': 'spin_mult',
                  'term': 'configuration',
                  'label': 'label'}

    def _read_ion_levels(self, ion):

        if not hasattr(ion, 'Elvlc'):
            return None

        levels_dict = {}

        for key, col_name in self.elvlc_dict.iteritems():
            levels_dict[col_name] = ion.Elvlc.get(key)

        # Convert energies from cm**-1 to eV
        for key in ['energy', 'energy_theoretical']:
            levels_dict[key] = u.Unit('cm-1').to('eV', levels_dict[key], equivalencies=u.spectral())

        # Check that ground level energy is 0
        try:
            for key in ['energy', 'energy_theoretical']:
                assert_almost_equal(levels_dict[key][0], 0)
        except AssertionError:
            raise ValueError('Level 0 energy is not 0.0')

        levels_df = pd.DataFrame(levels_dict)
        levels_df.replace(r'\s+', np.nan, regex=True, inplace=True)  # Replace space with NaN
        levels_df["atomic_number"] = ion.Z
        levels_df["ionization_stage"] = ion.Ion
        levels_df.set_index(["atomic_number", "ionization_stage", "level_index"], inplace=True)

        # Keep only bound levels
        levels_df = levels_df[levels_df['energy'] < ion.Ip]

        return levels_df

    def read_levels(self):
        levels_df = pd.DataFrame()
        for ion in self.ions:
            df = self._read_ion_levels(ion)
            levels_df = levels_df.append(df)  # None is treated as an empty dataframe
        return levels_df


def get_level_term(spin_mult, L, conf):
    # ToDo: determine parity from configuration
    term = str(spin_mult) + L;
    return term


class ChiantiIngestor(object):

    def __init__(self, session, ds_short_name="ch_v8.0"):
        self.session = session
        self.ds = DataSource.as_unique(self.session, short_name=ds_short_name)

    def ingest_levels(self, levels_df):

        for index, row in levels_df.iterrows():
            atomic_number, ioniz_stage, level_index = index
            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ioniz_stage-1)
            self.session.commit()
            term = get_level_term(row["spin_mult"], row["L"], row["configuration"])
            level = Level.as_unique(self.session, ion=ion, configuration=row["configuration"],
                                        term=term, J=row["J"])
            self.session.commit()
            # ToDo: check if data from this source already exists; update it in this case
            level_data = LevelChiantiData(
                level=level, data_source=self.ds, ch_index=level_index, ch_label=row["label"],
                energies=[
                    LevelEnergy(quantity=row["energy"]*u.eV, method="m"),
                    LevelEnergy(quantity=row["energy_theoretical"]*u.eV, method="th")
                ]
            )
            self.session.add(level_data)
