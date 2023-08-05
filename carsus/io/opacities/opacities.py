import logging
import pandas as pd

from carsus.io.util import read_from_buffer


h_minus_URL = "https://raw.githubusercontent.com/tardis-sn/carsus-data-negative-ions/main/h_minus_cross_section_wbr.dat"

logger = logging.getLogger(__name__)


class HMINUSOPACITIESReader(object):
    """
    Class for extracting wavelengths and cross-section from h-minus opacities data files

    Attributes
    ----------
    fname: path to h_minus_cross_section_wbr.dat

    Methods
    --------
    h_minus_raw:
        Return pandas DataFrame representation of h_minus_cross_section_wbr

    """

    h_minus_columns = ["wavelength", "cross_section"]

    def __init__(self, fname=None):
        """
        Parameters
        ----------
        fname: str
            Path to the h_minus file (http or local file).
        """

        if fname is None:
            self.fname = h_minus_URL
        else:
            self.fname = fname

        self._h_minus = None

    @property
    def h_minus(self):
        if self._h_minus is None:
            self._h_minus, self.version = self.read_h_minus()
        return self._h_minus

    def read_h_minus(self, fname=None):
        """
        Reading in a normal h_minus_cross_section_wbr.dat

        Parameters
        ----------
        fname: ~str
            path to h_minus_cross_section_wbr.dat

        Returns
        -------
            pandas.DataFrame
                pandas Dataframe represenation of h_minus

            str
                MD5 checksum
        """

        if fname is None:
            fname = self.fname

        logger.info(f"Parsing h_minus from: {fname}")

        # Format
        # wavelength,cross_section

        buffer, checksum = read_from_buffer(self.fname)
        h_minus = pd.read_csv(
            buffer,
            skip_blank_lines=True,
            names=self.h_minus_columns,
        )

        # remove empty lines
        h_minus = h_minus[~h_minus.isnull().all(axis=1)].reset_index(drop=True)
        return h_minus, checksum

    def to_hdf(self, fname):
        """
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """
        with pd.HDFStore(fname, "w") as f:
            f.put("/h_minus", self.h_minus)
