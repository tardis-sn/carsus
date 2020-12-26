import os
import carsus
import numpy as np
import pandas as pd
from carsus.io.base import BaseParser
from carsus.model import (
    Zeta,
    Temperature,
    DataSource
)

ZETA_DATA_URL = "https://media.githubusercontent.com/media/tardis-sn/carsus-db/master/zeta/knox_long_recombination_zeta.dat"

class KnoxLongZetaIngester(object):

    def __init__(self, session, fname=None, ds_name='knox_long'):
        self.session = session

        if fname is None:
            self.fname = ZETA_DATA_URL
        else:
            self.fname = fname

        self.data_source = DataSource.as_unique(
            self.session,
            short_name=ds_name
        )
        
        if self.data_source.data_source_id is None:
            self.session.flush()

    def ingest_zeta_values(self):
        t_values = np.arange(2000, 42000, 2000)

        names = ['atomic_number', 'ion_charge']
        names += [str(i) for i in t_values]

        zeta = np.recfromtxt(
            self.fname,
            usecols=range(1, 23),
            names=names)

        zeta_df = (
            pd.DataFrame.from_records(zeta).set_index(
                ['atomic_number', 'ion_charge']).T
        )

        data = list()
        for i, s in zeta_df.iterrows():
            T = Temperature.as_unique(self.session, value=int(i))
            if T.id is None:
                self.session.flush()

            for (atomic_number, ion_charge), rate in s.items():
                data.append(
                    Zeta(
                        atomic_number=atomic_number,
                        ion_charge=ion_charge,
                        data_source=self.data_source,
                        temp=T,
                        zeta=rate
                    )
                )

    def ingest(self):
        self.ingest_zeta_values()
        self.session.commit()


class KnoxLongZeta(BaseParser):
    """
        Attributes
        ----------
        base : pandas.DataFrame
        columns : list of str

        Methods
        -------
        to_hdf(fname)
            Dump the `base` attribute into an HDF5 file
    """

    def __init__(self, fname=None):

        if fname is None:
            self.fname = ZETA_DATA_URL
        else:
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

        self.base = (
            pd.DataFrame(zeta_raw).set_index(
                ['atomic_number', 'ion_charge'])
        )

        columns = [float(c) for c in self.base.columns]

        # To match exactly the `old` format
        self.base.columns = pd.Float64Index(columns, name='temp')
        self.columns = columns

    def to_hdf(self, fname):
        """Dump the `base` attribute into an HDF5 file

        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """
        with pd.HDFStore(fname, 'w') as f:
            f.put('/zeta_data', self.base)
