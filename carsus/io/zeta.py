import numpy as np
import pandas as pd
from carsus.model import (
        Zeta,
        Temperature,
        DataSource
        )


class KnoxLongZetaIngester(object):

    def __init__(self, session, data_fn, ds_name='knox_long'):
        self.session = session
        self.data_fn = data_fn
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
                self.data_fn,
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
