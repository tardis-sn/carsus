import re
import logging
import pandas as pd
from io import StringIO
from carsus.io.util import read_from_buffer

VALD_URL = "https://media.githubusercontent.com/media/tardis-sn/carsus-db/master/vald/vald_sample.dat"

logger = logging.getLogger(__name__)


class VALDReader(object):
    """
    Class for extracting lines and levels data from vald files

    Attributes
    ----------
    fname: path to vald.dat

    Methods
    --------
    vald_raw:
        Return pandas DataFrame representation of vald

    """

    vald_columns = [
        "elm_ion",
        "wl_air",
        "log_gf",
        "e_low",
        "j_lo",
        "e_up",
        "j_up",
        "lande_lower",
        "lande_upper",
        "lande_mean",
        "rad",
        "stark",
        "waals",
    ]

    def __init__(self, fname=None):
        """
        Parameters
        ----------
        fname: str
            Path to the vald file (http or local file).

        """

        if fname is None:
            self.fname = VALD_URL
        else:
            self.fname = fname

        self._vald_raw = None
        self._vald = None

    @property
    def vald_raw(self):
        if self._vald_raw is None:
            self._vald_raw, self.version = self.read_vald_raw()
        return self._vald_raw

    @property
    def vald(self):
        if self._vald is None:
            self._vald = self.parse_vald()
        return self._vald

    def read_vald_raw(self, fname=None):
        """
        Reading in a normal vald.dat

        Parameters
        ----------
        fname: ~str
            path to vald.dat

        Returns
        -------
            pandas.DataFrame
                pandas Dataframe represenation of vald

            str
                MD5 checksum
        """

        if fname is None:
            fname = self.fname

        logger.info(f"Parsing VALD from: {fname}")

        # FORMAT
        # Elm Ion       WL_air(A)  log gf* E_low(eV) J lo  E_up(eV) J up   lower   upper    mean   Rad.  Stark    Waals
        # 'TiO 1',     4100.00020, -11.472,  0.2011, 31.0,  3.2242, 32.0, 99.000, 99.000, 99.000, 6.962, 0.000, 0.000,

        data_match = re.compile("'[a-zA-Z]+ \d+',[\s*-?\d+[\.\d+]+,]*")

        buffer, checksum = read_from_buffer(self.fname)
        vald = pd.read_csv(
            StringIO("\n".join(data_match.findall(buffer.read().decode()))),
            names=self.vald_columns,
            index_col=False,
        )

        return vald, checksum

    def parse_vald(self, vald_raw=None):
        """
        Parse raw vald DataFrame

        Parameters
        ----------
        vald_raw: pandas.DataFrame

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """

        vald = vald_raw if vald_raw is not None else self.vald_raw.copy()

        vald["elm_ion"] = vald["elm_ion"].str.replace("'", "")
        vald[["chemical", "ion_charge"]] = vald["elm_ion"].str.split(" ", expand=True)
        vald["ion_charge"] = vald["ion_charge"].astype(int) - 1

        del vald["elm_ion"]

        return vald

    def to_hdf(self, fname):
        """
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """
        with pd.HDFStore(fname, "w") as f:
            f.put("/vald_raw", self.vald_raw)
            f.put("/vald", self.vald)
