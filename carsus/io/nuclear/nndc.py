import logging

import pandas as pd
from pathlib import Path
import subprocess

DECAY_DATA_SOURCE_DIR = Path.home() / "Downloads" / "carsus-data-nndc"

DECAY_DATA_FINAL_DIR = Path.home() / "Downloads" / "tardis-data" / "decay-data"

NNDC_SOURCE_URL = "https://github.com/tardis-sn/carsus-data-nndc"

logger = logging.getLogger(__name__)


class NNDCReader:
    """
    Class for extracting nuclear decay data from NNDC archives

    Attributes
    ----------
    dirname: path to directory containing the decay data in CSV format

    Methods
    --------
    decay_data:
        Return pandas DataFrame representation of the decay data
    """

    def __init__(self, dirname=None, remote=False):
        """
        Parameters
        ----------
        dirname : path
            Path to the directory containing the source CSV data (local file).

        """
        if dirname is None:
            if remote:
                subprocess.run(['git', 'clone', NNDC_SOURCE_URL, DECAY_DATA_SOURCE_DIR])
                logger.info(f"Downloading NNDC decay data from {NNDC_SOURCE_URL}")
            self.dirname = Path().joinpath(DECAY_DATA_SOURCE_DIR, "csv")
        else:
            self.dirname = dirname

        logger.info(f"Parsing decay data from: {DECAY_DATA_SOURCE_DIR}/csv")

        self._decay_data = None

    @property
    def decay_data(self):
        if self._decay_data is None:
            self._decay_data = self._prepare_nuclear_dataframes()
        return self._decay_data

    def _get_nuclear_decay_dataframe(self):
        """
        Convert the CSV files from the source directory into dataframes

        Returns
        -------
            pandas.DataFrame
                pandas Dataframe representation of the decay data
        """

        all_data = []
        dirpath = Path(self.dirname)
        for file in dirpath.iterdir():
            # convert every csv file to Dataframe and append it to all_data
            if file.suffix == ".csv" and file.stat().st_size != 0:
                data = pd.read_csv(
                    file,
                )
                all_data.append(data)

        decay_data = pd.concat(all_data)
        return decay_data

    def _set_group_true(self, group):
        """
        Sets the entire 'Metastable' column to True if any of the values in the group is True.

        Parameters
        ----------
        group: pandas.DataFrameGroupBy object
            A groupby object that contains information about the groups.
        """

        if group['Metastable'].any():
            group['Metastable'] = True
        return group

    def _add_metastable_column(self, decay_data=None):
        """
        Adds a 'Metastable' column to decay_data indicating the metastable isotopes (e.g: Mn52).

        Returns
        -------
            pandas.Dataframe
                Decay dataframe after the 'Metastable' column has been added.
        """
        metastable_df = decay_data if decay_data is not None else self.decay_data.copy()

        # Create a boolean metastable state column before the 'Decay Mode' column
        metastable_df.insert(7, "Metastable", False)

        metastable_filters = (metastable_df["Decay Mode"] == "IT") & (metastable_df["Decay Mode Value"] != 0.0) & (
                metastable_df["Parent E(level)"] != 0.0)

        metastable_df.loc[metastable_filters, 'Metastable'] = True

        # avoid duplicate indices since metastable_df is a result of pd.concat operation
        metastable_df = metastable_df.reset_index()

        # Group by the combination of these columns
        group_criteria = ['Parent E(level)', 'T1/2 (sec)', 'Isotope']
        metastable_df = metastable_df.groupby(group_criteria).apply(self._set_group_true)

        return metastable_df

    def _prepare_nuclear_dataframes(self):
        """
        Creates the decay dataframe to be stored in HDF Format and formats it by adding
        the 'Metastable' and 'Isotope' columns, setting the latter as the index.
        """
        decay_data_raw = self._get_nuclear_decay_dataframe()
        decay_data_raw["Isotope"] = decay_data_raw.Element.map(str) + decay_data_raw.A.map(str)

        decay_data = self._add_metastable_column(decay_data_raw)

        decay_data.set_index(['Isotope'], inplace=True)
        decay_data.drop(['index'], axis=1, inplace=True)
        decay_data=decay_data.sort(['A', 'Z'])

        return decay_data

    def to_hdf(self, fpath=None):
        """
        Parameters
        ----------
        fpath: path
            Path to the HDF5 output file
        """
        if fpath is None:
            fpath = DECAY_DATA_FINAL_DIR

        if not Path(fpath).exists():
            Path(fpath).mkdir()

        target_fname = Path().joinpath(fpath, "compiled_ensdf_csv.h5")

        with pd.HDFStore(target_fname, 'w') as f:
            f.put('/decay_data', self.decay_data)
