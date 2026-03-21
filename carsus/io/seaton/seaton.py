import pandas as pd
import logging

logger = logging.getLogger(__name__)

SEATON_1992_URL = "https://cdsarc.cds.unistra.fr/ftp/VI/80/s92.201.gz"


class Seaton1992Reader(object):
    """
    Reader for the Seaton 1992 continuum opacity data (Opacity Project).

    Reads the s92.201.gz file from the CDS archive and parses it
    into two pandas DataFrames:
    - abundances: element abundances used in opacity calculations
    - opacities: frequency-temperature opacity grid

    Parameters
    ----------
    fpath : str, optional
        Path or URL to the s92.201.gz file.
        Defaults to the CDS archive URL.

    Attributes
    ----------
    abundances : DataFrame
        DataFrame with columns: atomic_number, abundance.
    opacities : DataFrame
        DataFrame with columns: log_frequency, log_temperature, opacity.

    Notes
    -----
    If the CDS archive is unavailable, you can pass a local file path:
        reader = Seaton1992Reader(fpath="path/to/s92.201.gz")
    """

    def __init__(self, fpath=None):
        self.fpath = SEATON_1992_URL if fpath is None else fpath
        self._abundances = None
        self._opacities = None

    def read_raw(self):
        """
        Downloads and reads the raw Seaton 1992 data file.

        Returns
        -------
        tuple
            (abundances_df, opacities_df)
        """
        logger.info(f"Reading Seaton 1992 opacity data from: {self.fpath}")

        try:
            df = pd.read_csv(
                self.fpath,
                compression="gzip",
                sep=r"\s+",
                header=None,
                on_bad_lines="skip",
            )
        except Exception as e:
            logger.error(
                f"Failed to read data from {self.fpath}: {e}\n"
                f"Try passing a local file path explicitly:\n"
                f"Seaton1992Reader("
                f"fpath='path/to/s92.201.gz')"
            )
            raise

        # --- Section 1: Abundances (rows 2 to 18) ---
        abundances_df = df.iloc[2:19][[0, 1]].copy()
        abundances_df.columns = ["atomic_number", "abundance"]
        abundances_df["atomic_number"] = pd.to_numeric(
            abundances_df["atomic_number"], errors="coerce"
        )
        abundances_df["abundance"] = pd.to_numeric(
            abundances_df["abundance"], errors="coerce"
        )
        abundances_df = abundances_df.dropna()
        col = abundances_df["atomic_number"]
        abundances_df["atomic_number"] = col.astype(int)
        abundances_df = abundances_df.reset_index(drop=True)

        # --- Section 2: Opacity grid (rows 20 onwards) ---
        opacity_rows = []
        for _, row in df.iloc[20:].iterrows():
            col0 = pd.to_numeric(row[0], errors="coerce")
            col1 = pd.to_numeric(row[1], errors="coerce")
            col2 = pd.to_numeric(row[2], errors="coerce")

            if pd.isna(col0) or pd.isna(col1) or pd.isna(col2):
                continue

            # Skip block header rows (large integers like 140, 142)
            if col0 > 100 and col2 > 50:
                continue

            opacity_rows.append({
                "log_frequency": col0,
                "log_temperature": col1,
                "opacity": col2
            })

        opacities_df = pd.DataFrame(opacity_rows)
        opacities_df = opacities_df.reset_index(drop=True)

        return abundances_df, opacities_df

    @property
    def abundances(self):
        """
        Returns the element abundances DataFrame.
        Loads data on first access.
        """
        if self._abundances is None:
            self._abundances, self._opacities = self.read_raw()
        return self._abundances

    @property
    def opacities(self):
        """
        Returns the opacity grid DataFrame.
        Loads data on first access.
        """
        if self._opacities is None:
            self._abundances, self._opacities = self.read_raw()
        return self._opacities
