import chianti.core as ch
import pandas as pd
import numpy as np
import pickle
import os
import re
from numpy.testing import assert_almost_equal
from astropy import units as u
from carsus.model import DataSource, Ion, ChiantiLevel, LevelEnergy

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
                  'ecm': 'energy',  # cm-1
                  'ecmth': 'energy_theoretical',  # cm-1
                  'j': 'J',
                  'spd': 'L',
                  'spin': 'spin_multiplicity',
                  'pretty': 'pretty',  # configuration + term
                  'label': 'label'}

    def _read_ion_levels(self, ion):

        if not hasattr(ion, 'Elvlc'):
            print("No energy data is available for ion {}".format(ion))
            return None

        levels_dict = {}

        for key, col_name in self.elvlc_dict.iteritems():
            levels_dict[col_name] = ion.Elvlc.get(key)

        # Check that ground level energy is 0
        try:
            for key in ['energy', 'energy_theoretical']:
                assert_almost_equal(levels_dict[key][0], 0)
        except AssertionError:
            raise ValueError('Level 0 energy is not 0.0')

        levels_df = pd.DataFrame(levels_dict)

        # Replace empty labels with NaN
        levels_df["label"].replace(r'\s+', np.nan, regex=True, inplace=True)

        # Extract configuration and term from the "pretty" column
        term_conf_df = levels_df["pretty"].str.\
            extract("(?P<configuration>(\d[a-z]\d?[\s,.]?)+)\s(?P<term>\d+[A-Z][\d,.]+)")
        levels_df[["term", "configuration"]] = term_conf_df[["term", "configuration"]]
        levels_df.drop("pretty", axis=1, inplace=True)

        levels_df["atomic_number"] = ion.Z
        levels_df["ion_charge"] = ion.Ion - 1
        levels_df.set_index(["atomic_number", "ion_charge", "level_index"], inplace=True)

        # Keep only bound levels ?
        # ip = u.eV.to(u.Unit("cm-1", value=ion.Ip, equivalencies=u.spectral())
        # levels_df = levels_df[levels_df['energy'] < ion.Ip]

        return levels_df

    def read_levels(self):
        levels_df = pd.DataFrame()
        for ion in self.ions:
            df = self._read_ion_levels(ion)
            levels_df = levels_df.append(df)  # None is treated as an empty dataframe
        return levels_df


class ChiantiIngester(object):

    def __init__(self, session, ions_list=masterlist_ions, ds_short_name="chianti_v8.0.2"):
        self.session = session
        self.reader = ChiantiReader(ions_list=ions_list)
        self.ds = DataSource.as_unique(self.session, short_name=ds_short_name)

    def _ingest_levels(self, levels_df):

        for index, ion_df in levels_df.groupby(level=["atomic_number", "ion_charge"]):

            atomic_number, ion_charge = index
            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            # ToDo: Determine parity from configuration
            # ToDo: Check if the level from this source already exists and update it then

            for index, row in ion_df.iterrows():

                ch_index = index[2]  # (atomic_number, ion_charge, chianti_index)
                level = ChiantiLevel(ion=ion, data_source=self.ds, ch_index=ch_index, ch_label=row["label"],
                                     configuration=row["configuration"], term=row["term"],
                                     L=row["L"], J=row["J"], spin_multiplicity=row["spin_multiplicity"])

                level.energies = []
                for column, method in [('energy', 'm'), ('energy_theoretical', 'th')]:
                    if row[column] != -1:  # check if the value exists
                        level.energies.append(
                            LevelEnergy(quantity=row[column] * u.Unit("cm-1"), method=method),
                        )
                self.session.add(level)

    def ingest(self, levels=True):

        if levels:
            levels_df = self.reader.read_levels()
            self._ingest_levels(levels_df)
