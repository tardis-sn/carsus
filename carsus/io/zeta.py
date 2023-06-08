import numpy as np
import pandas as pd
from carsus.io.base import BaseParser
from carsus.io.util import read_from_buffer
from carsus.model import (
    Zeta,
    Temperature,
    DataSource
)

ZETA_DATA_URL = "https://media.githubusercontent.com/media/tardis-sn/carsus-db/master/zeta/knox_long_recombination_zeta.dat"

class KnoxLongZetaIngester(object):

    def __init__(self, session, fname=None, ds_name='knox_long'):
        self.session = session

        self.fname = ZETA_DATA_URL if fname is None else fname
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

        data = []
        for i, s in zeta_df.iterrows():
            T = Temperature.as_unique(self.session, value=int(i))
            if T.id is None:
                self.session.flush()

            data.extend(
                Zeta(
                    atomic_number=atomic_number,
                    ion_charge=ion_charge,
                    data_source=self.data_source,
                    temp=T,
                    zeta=rate,
                )
                for (atomic_number, ion_charge), rate in s.items()
            )

    def ingest(self):
        self.ingest_zeta_values()
        self.session.commit()


class KnoxLongZeta(BaseParser):
    """
    Attributes
    ----------
    base : pandas.DataFrame
    """

    def __init__(self, fname=None):

        self.fname = ZETA_DATA_URL if fname is None else fname
        self._prepare_data()

    def _prepare_data(self):
        t_values = np.arange(2000, 42000, 2000)
        names = ["atomic_number", "ion_charge"]
        names += [str(i) for i in t_values]

        buffer, checksum = read_from_buffer(self.fname)
        self.version = checksum

        zeta_df = pd.read_csv(
            buffer,
            usecols=range(1, 23),
            names=names,
            comment="#",
            delim_whitespace=True)

        self.base = (
            pd.DataFrame(zeta_df).set_index(
                ["atomic_number", "ion_charge"])
        )

        columns = [float(c) for c in self.base.columns]
        self.base.columns = pd.Float64Index(columns, name="temp")
