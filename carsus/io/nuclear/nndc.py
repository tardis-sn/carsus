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

    def _prepare_nuclear_dataframes(self):
        decay_data = self._get_nuclear_decay_dataframe()
        decay_data["Isotope"] = decay_data.Element.map(str) + decay_data.A.map(str)

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
