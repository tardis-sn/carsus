import pandas as pd
import os
import datetime

DECAY_DATA_SOURCE_DIR = os.path.join(
    os.path.expanduser("~"), "Downloads", "carsus-data-nndc-main", "csv"
)

DECAY_DATA_FINAL_DIR = os.path.join(
    os.path.expanduser("~"), "Downloads", "tardis-data", "decay-data"
)


class NNDCReader:

    def __init__(self, dirname=None):
        if dirname is None:
            self.dirname = DECAY_DATA_SOURCE_DIR
        else:
            self.dirname = dirname

        self._decay_data = None

        # TODO: create an 'nndc_columns' list present in the final dataset.

    @property
    def decay_data(self):
        if self._decay_data is None:
            self._decay_data = self._prepare_nuclear_dataframes()
        return self._decay_data

    def _get_nuclear_decay_dataframe(self):
        all_data = []
        for fileName in os.listdir(self.dirname):
            file = os.path.join(self.dirname, fileName)

            # convert every csv file to Dataframe and appends it to all_data
            if os.path.splitext(file)[1] == ".csv" and os.path.getsize(file) != 0:
                data = pd.read_csv(
                    file,
                )
                all_data.append(data)

        decay_data = pd.concat(all_data)
        return decay_data

    def _set_group_true(self, group):
        # set the entire Metastable column to True if any of the values in the group is True
        if group['Metastable'].any():
            group['Metastable'] = True
        return group

    def _add_metastable_column(self, decay_data=None):
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
        decay_data_raw = self._get_nuclear_decay_dataframe()
        decay_data_raw["Isotope"] = decay_data_raw.Element.map(str) + decay_data_raw.A.map(str)

        decay_data = self._add_metastable_column(decay_data_raw)

        decay_data.set_index(['Isotope'], inplace=True)
        decay_data.drop(['index'], axis=1, inplace=True)

        return decay_data

    def to_hdf(self, fpath=None):
        if fpath is None:
            fpath = DECAY_DATA_FINAL_DIR

        if not os.path.exists(fpath):
            os.mkdir(fpath)

        target_fname = os.path.join(fpath, "compiled_ensdf_csv.h5")

        with pd.HDFStore(target_fname, 'w') as f:
            f.put('/decay_data', self.decay_data)
