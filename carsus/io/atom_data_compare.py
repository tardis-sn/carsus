import pandas as pd
import numpy as np
import logging
from carsus.util import parse_selected_species, convert_atomic_number2symbol
from collections import defaultdict

from collections import defaultdict
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def highlight_values(val):
    if val == True:
        return "background-color: #BCF5A9"
    else:
        return "background-color: #F5A9A9"


def highlight_diff(val):
    if val == 0:
        return "background-color: #BCF5A9"
    else:
        return "background-color: #F5A9A9"


def highlight_values(val):
    if val == True:
        return "background-color: #BCF5A9"
    else:
        return "background-color: #F5A9A9"


def highlight_diff(val):
    if val == 0:
        return "background-color: #BCF5A9"
    else:
        return "background-color: #F5A9A9"


class AtomDataCompare(object):
    def __init__(self, d1_path=None, d2_path=None):
        self.d1_path = d1_path
        self.d2_path = d2_path
        # TODO: for all dataframes
        self.level_columns = ["energy", "g", "metastable"]
        self.setup()

        self.alt_keys_default = {
            "lines": ["lines_data", "lines"],
            "levels": ["levels_data", "levels"],
            "collisions": ["collisions_data", "collision_data"],
            "photoionization_data": ["photoionization_data"],
        }
        self.alt_keys_default = defaultdict(list, self.alt_keys_default)

    def set_keys_as_attributes(self, alt_keys={}):
        # alt keys should be a subset of this self.alt_keys_default
        # other keys would be ignored

        for key, val in self.alt_keys_default.items():
            if alt_keys.get(key, None):
                self.alt_keys_default[key].extend(alt_keys[key])

            for item in val:
                if self.d1.get_node(item):
                    setattr(self, f"{key}1", self.d1[item])
                if self.d2.get_node(item):
                    setattr(self, f"{key}2", self.d2[item])

    def setup(self):
        self.d1 = pd.HDFStore(self.d1_path)
        self.d2 = pd.HDFStore(self.d2_path)

    def teardown(self):
        self.d1.close()
        self.d2.close()

    def generate_comparison_table(
        self, exclude_correct_matches=False, drop_file_keys=True
    ):
        for index, file in enumerate((self.d1, self.d2)):
            file_keys = {item[1:]: item[1:] for item in file.keys()}

            for original_keyname in self.alt_keys_default.keys():
                for file_key in file_keys.keys():
                    alt_key_names = self.alt_keys_default.get(original_keyname, [])

                    if file_key in alt_key_names:
                        # replace value with key name in self.alt_keys_default
                        file_keys[file_key] = original_keyname

            # flip the dict to create the dataframe
            file_keys = {v: k for k, v in file_keys.items()}
            df = pd.DataFrame(file_keys, index=["file_keys"]).T
            df["exists"] = True
            setattr(self, f"d{index+1}_df", df)

        joined_df = self.d1_df.join(self.d2_df, how="outer", lsuffix="_1", rsuffix="_2")
        joined_df[["exists_1", "exists_2"]] = joined_df[
            ["exists_1", "exists_2"]
        ].fillna(False)
        self.comparison_table = joined_df
        self.comparison_table["match"] = None

        for index, row in self.comparison_table.iterrows():
            if row[["exists_1", "exists_2"]].all():
                row1_df = self.d1[row["file_keys_1"]]
                row2_df = self.d2[row["file_keys_2"]]
                if row1_df.equals(row2_df):
                    self.comparison_table.at[index, "match"] = True
                else:
                    self.comparison_table.at[index, "match"] = False
            else:
                self.comparison_table.at[index, "match"] = False

        if exclude_correct_matches:
            self.comparison_table = self.comparison_table[
                self.comparison_table.match == False
            ]
        if drop_file_keys:
            self.comparison_table = self.comparison_table.drop(
                columns=["file_keys_1", "file_keys_2"]
            )

    @property
    def comparison_table_stylized(self):
        return self.comparison_table.style.applymap(
            highlight_values, subset=["exists_1", "exists_2", "match"]
        )

    def diff(self, key_name, ion, rtol=1e-07, simplify_output=False):
        # Ion differences

        try:
            df1 = getattr(self, f"{key_name}1")
            df2 = getattr(self, f"{key_name}2")
        except AttributeError as exc:
            raise Exception(
                f"Either key_name: {key_name} is invalid or keys are not set."
                "Please use the set_keys_as_attributes method to set keys as attributes for comparison."
            )

        # TODO: should this be moved somewhere else?
        species1 = df1.index.get_level_values("atomic_number")
        species1 = set([convert_atomic_number2symbol(item) for item in species1])

        species2 = df2.index.get_level_values("atomic_number")
        species2 = set([convert_atomic_number2symbol(item) for item in species2])

        species_diff = species1.symmetric_difference(species2)
        if len(species_diff):
            print(f"Elements not in common in both dataframes: {species_diff}")

        common_columns = df2.columns.intersection(df1.columns)
        if common_columns.empty:
            raise ValueError("There are no common columns for comparison. Exiting.")

        mismatched_cols = df2.columns.symmetric_difference(df1.columns)
        if not mismatched_cols.empty:
            logger.warning("Columns do not match.")
            logger.warning(f"Mismatched columns: {mismatched_cols}")
            logger.info(f"Using common columns for comparison:{common_columns}")

        if df1.index.names != df2.index.names:
            raise ValueError("Index names do not match.")

        setattr(self, f"{key_name}_columns", common_columns)
        parsed_ion = parse_selected_species(ion)[0]

        try:
            df1 = df1.loc[parsed_ion]
            df2 = df2.loc[parsed_ion]
        except KeyError as exc:
            raise Exception(
                "The element does not exist in one of the dataframes."
            ) from exc

        merged_df = pd.merge(
            df1,
            df2,
            left_index=True,
            right_index=True,
            suffixes=["_1", "_2"],
        )

        non_numeric_cols = ["line_id", "metastable"]
        common_cols_rearranged = []

        for item in common_columns:
            if item in non_numeric_cols:
                merged_df[f"matches_{item}"] = (
                    merged_df[f"{item}_1"] == merged_df[f"{item}_2"]
                )
                common_cols_rearranged.extend(
                    [
                        f"{item}_1",
                        f"{item}_2",
                        f"matches_{item}",
                    ]
                )
            else:
                merged_df[f"matches_{item}"] = np.isclose(
                    merged_df[f"{item}_1"], merged_df[f"{item}_2"], rtol=rtol
                )
                merged_df[f"pct_change_{item}"] = merged_df[
                    [f"{item}_1", f"{item}_2"]
                ].pct_change(axis=1)[f"{item}_2"]

                merged_df[f"pct_change_{item}"] = merged_df[
                    f"pct_change_{item}"
                ].fillna(0)

                common_cols_rearranged.extend(
                    [f"{item}_1", f"{item}_2", f"matches_{item}", f"pct_change_{item}"]
                )

        merged_df = merged_df[common_cols_rearranged]
        merged_df = merged_df.sort_values(by=merged_df.index.names, axis=0)
        setattr(self, f"merged_{key_name}_{ion}", merged_df)

        if simplify_output:
            return self.simplified_df(merged_df.copy())

        return merged_df

    def simplified_df(self, df):
        df_simplified = df.drop(df.filter(regex="_1$|_2$").columns, axis=1)
        return df_simplified

    def plot(self, key_name, ion, column):
        df_name_format = "merged_{}_".format(key_name)
        if hasattr(self, df_name_format + ion):
            merged_df = getattr(self, df_name_format + ion)
        else:
            merged_df = self.diff(key_name=key_name, ion=ion)
        return plt.scatter(
            merged_df[f"{column}_1"] / merged_df[f"{column}_2"],
            merged_df[f"{column}_2"],
        )

    def style_df(self, key_name, ion, simplify_df=True):
        df_name = f"merged_{key_name}_{ion}"

        if hasattr(self, df_name):
            merged_df = getattr(self, df_name)
        else:
            merged_df = self.diff(key_name=key_name, ion=ion)

        columns = getattr(self, f"{key_name}_columns")
        subset = [f"matches_{item}" for item in columns]
        if simplify_df:
            merged_df = self.simplified_df(merged_df)

        return merged_df.style.applymap(highlight_values, subset=[*subset])