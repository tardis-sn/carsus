import numpy as np
import pandas as pd
from carsus.io.base import BaseParser
from carsus.io.util import read_from_buffer

ZETA_DATA_URL = "https://media.githubusercontent.com/media/tardis-sn/carsus-db/master/zeta/knox_long_recombination_zeta.dat"

class KnoxLongZeta(BaseParser):
    """
    Attributes
    ----------
    base : pandas.DataFrame
    """

    def __init__(self, fname=None):

        if fname is None:
            self.fname = ZETA_DATA_URL

        else:
            self.fname = fname

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
