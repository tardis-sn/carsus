import pandas as pd
import numpy as np
import functools
import logging
from carsus.util import parse_selected_species, convert_atomic_number2symbol
from collections import defaultdict

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

LIGHT_GREEN = "#BCF5A9"
LIGHT_RED = "#F5A9A9"


def highlight_values(val):
    """
    Return hex string of background color.

    Parameters
    ----------
    val : bool

    Returns
    -------
    string
    """
    if val == True:
        return f"background-color: {LIGHT_GREEN}"
    else:
        return f"background-color: {LIGHT_RED}"


class AtomDataCompare(object):
    """
    Differentiate between two Carsus atomic files.

    Parameters
    ----------
    d1_path : string
        Path to the first file.
    d2_path : string
        Path to the second file.
    alt_keys : dict, optional
        Alternate names to dataframes inside the atomic files.
        For example, the `lines` dataframe was used to be called `lines_data` in earlier carsus versions.
    """

    def __init__(self, d1_path=None, d2_path=None, alt_keys={}):
        self.d1_path = d1_path
        self.d2_path = d2_path
        self.alt_keys_default = {
            "lines": ["lines_data", "lines"],
            "levels": ["levels_data", "levels"],
            "collisions": ["collisions_data", "collision_data"],
            "photoionization_data": ["photoionization_data"],
        }
        self.alt_keys_default = defaultdict(list, self.alt_keys_default)
        self.setup(alt_keys=alt_keys)

    def set_keys_as_attributes(self, alt_keys={}):
        """
        Set dataframes as attributes.

        Parameters
        ----------
        alt_keys : dict, optional
            Alternate names to dataframes inside the atomic files. Defaults to {}.
        """
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

    def setup(self, alt_keys={}):
        """
        Opeb HDF files using Pandas HDFStore.

        Parameters
        ----------
        alt_keys : dict, optional
            Alternate names to dataframes inside the atomic files. Defaults to {}.
        """
        self.d1 = pd.HDFStore(self.d1_path)
        self.d2 = pd.HDFStore(self.d2_path)
        self.set_keys_as_attributes(alt_keys=alt_keys)

    def teardown(self):
        """
        Close open HDF files.
        """
        self.d1.close()
        self.d2.close()

    def verify_key_diff(self, key_name):
        """
        Check if dataframes can be compared.

        Parameters
        ----------
        key_name : string
        """
        try:
            df1 = getattr(self, f"{key_name}1")
            df2 = getattr(self, f"{key_name}2")
        except AttributeError:
            raise Exception(
                f"Either key_name: {key_name} is invalid or keys are not set. "
                "Please use the set_keys_as_attributes method to set keys as attributes for comparison."
            )

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

    def ion_diff(
        self,
        key_name,
        ion,
        rtol=1e-07,
        simplify_output=True,
        return_summary=False,
        style=True,
        style_axis=0,
    ):
        """
        Compare two dataframes- ion wise.

        Parameters
        ----------
        key_name : string
        ion: string or tuple
        rtol: int
        simplify_output: bool
        return_summary: bool
        style: bool
        style_axis: int or None
        """
        try:
            df1 = getattr(self, f"{key_name}1")
            df2 = getattr(self, f"{key_name}2")
        except AttributeError:
            raise Exception(
                f"Either key_name: {key_name} is invalid or keys are not set."
                "Please use the set_keys_as_attributes method to set keys as attributes for comparison."
            )

        if not hasattr(self, f"{key_name}_columns"):
            self.verify_key_diff(key_name)

        common_columns = getattr(self, f"{key_name}_columns")

        if not isinstance(ion, tuple):
            parsed_ion = parse_selected_species(ion)[0]
        else:
            parsed_ion = ion

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

        non_numeric_cols = ["line_id", "metastable"]  # TODO
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
        merged_df.apply(
            lambda column: column.abs() if column.dtype.kind in "iufc" else column
        )

        if return_summary:
            summary_dict = {}
            summary_dict["total_rows"] = len(merged_df)

            for column in merged_df.copy().columns:
                if column.startswith("matches_"):
                    summary_dict[column] = (
                        merged_df[column].copy().value_counts().get(True, 0)
                    )
            summary_df = pd.DataFrame(summary_dict, index=["values"])
            return summary_df

        if simplify_output:
            matches_cols = [
                column for column in merged_df.columns if column.startswith("matches")
            ]
            conditions = [merged_df[column] != True for column in matches_cols]

            merged_df = self.simplified_df(merged_df.copy())  # TODO
            merged_df = merged_df[functools.reduce(np.logical_or, conditions)]

            if merged_df.empty:
                print("All the values in both the dataframes match.")
                return None

            merged_df = merged_df.drop(
                columns=[
                    column
                    for column in merged_df.columns
                    if column.startswith("matches")
                ]
            )

        if style:
            pct_change_subset = [
                column
                for column in merged_df.columns
                if column.startswith("pct_change")
            ]
            return merged_df.style.background_gradient(
                cmap="Reds", subset=pct_change_subset, axis=style_axis
            )

        return merged_df

    def key_diff(
        self, key_name, rtol=1e-07, simplify_output=True, style=True, style_axis=0
    ):
        """
        Compare two dataframes.

        Parameters
        ----------
        key_name : string
        simplify_output: bool
        style: bool
        style_axis: int or None
        """
        if not hasattr(self, f"{key_name}_columns"):
            self.verify_key_diff(key_name)

        df1 = getattr(self, f"{key_name}1")
        df2 = getattr(self, f"{key_name}2")

        ions1 = set(
            [(atomic_number, ion_number) for atomic_number, ion_number, *_ in df1.index]
        )
        ions2 = set(
            [(atomic_number, ion_number) for atomic_number, ion_number, *_ in df2.index]
        )

        ions = set(ions1).intersection(ions2)
        ion_diffs = []
        for ion in ions:
            ion_diff = self.ion_diff(
                key_name=key_name, ion=ion, rtol=rtol, return_summary=True
            )
            ion_diff["atomic_number"], ion_diff["ion_number"] = ion
            ion_diff = ion_diff.set_index(["atomic_number", "ion_number"])
            ion_diffs.append(ion_diff)
        key_diff = pd.concat(ion_diffs)

        columns = key_diff.columns
        for column in columns:
            if column.startswith("matches"):
                key_diff[column] = key_diff["total_rows"] - key_diff[column]
                key_diff = key_diff.rename(columns={column: f"not_{column}"})
        key_diff = key_diff.sort_values(["atomic_number", "ion_number"])

        subset = [
            column for column in key_diff.columns if column.startswith("not_matches")
        ]
        conditions = [key_diff[column] != 0 for column in subset]

        if simplify_output:
            key_diff = key_diff[functools.reduce(np.logical_or, conditions)]

        if style:
            return key_diff.style.background_gradient(
                cmap="Reds", subset=subset, axis=style_axis
            )

        return key_diff

    def generate_comparison_table(self):
        """
        Generate empty comparison table.
        """
        for index, file in enumerate((self.d1, self.d2)):
            # create a dict to contain names of keys in the file
            # and their alternate(more recent) names
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

    def compare(self, exclude_correct_matches=False, drop_file_keys=True, style=True):
        """
        Compare the two HDF files.

        Parameters
        ----------
        exclude_correct_matches : bool
        drop_file_keys: bool
        style: bool
        """
        if not hasattr(self, "comparison_table"):
            self.generate_comparison_table()

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
        if style:
            return self.comparison_table.style.applymap(
                highlight_values, subset=["exists_1", "exists_2", "match"]
            )
        return self.comparison_table

    def simplified_df(self, df):
        """
        Drop additional columns belonging to the original dataframes but were used for comparison.

        Parameters
        ----------
        df : pd.DataFrame
        """
        df_simplified = df.drop(df.filter(regex="_1$|_2$").columns, axis=1)
        return df_simplified

    def plot_ion_diff(self, key_name, ion, column):
        """
        Plot fractional difference between properties of ions.

        Parameters
        ----------
        key_name : string
        ion: string or tuple
        column: string
        """
        df = self.ion_diff(
            key_name=key_name, ion=ion, style=False, simplify_output=False
        )
        plt.scatter(
            df[f"{column}_1"] / df[f"{column}_2"],
            df[f"{column}_2"],
        )

        plt.xlabel(f"{column}$_1$/{column}$_2$")
        plt.ylabel(f"{column}$_2$")
        plt.show()
