import numpy as np
import pandas as pd

import astropy.constants as const

from carsus.io.util import exclude_artificial_levels

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

class MacroAtomPreparer():
    def __init__(self, levels, lines):
        self.levels = levels
        self.lines = lines

    def create_macro_atom(self):
        """
        Create a DataFrame containing macro atom data.

        Notes
        -----
        Refer to the docs: https://tardis-sn.github.io/tardis/physics/setup/plasma/macroatom.html

        """

        # Exclude artificially created levels from levels
        levels = exclude_artificial_levels(self.levels)

        lvl_energy_lower = levels.rename(columns={"energy": "energy_lower"}).loc[
            :, ["energy_lower"]
        ]

        lvl_energy_upper = levels.rename(columns={"energy": "energy_upper"}).loc[
            :, ["energy_upper"]
        ]

        lines = self.lines.set_index("line_id")
        lines = lines.join(lvl_energy_lower, on="lower_level_id").join(
            lvl_energy_upper, on="upper_level_id"
        )

        macro_atom = list()
        macro_atom_dtype = [
            ("atomic_number", np.int64),
            ("ion_number", np.int64),
            ("source_level_number", np.int64),
            ("target_level_number", np.int64),
            ("transition_line_id", np.int64),
            ("transition_type", np.int64),
            ("transition_probability", np.float),
        ]

        for line_id, row in lines.iterrows():
            atomic_number, ion_number = row["atomic_number"], row["ion_number"]
            level_number_lower, level_number_upper = (
                row["level_number_lower"],
                row["level_number_upper"],
            )
            nu = row["nu"]
            f_ul, f_lu = row["f_ul"], row["f_lu"]
            e_lower, e_upper = row["energy_lower"], row["energy_upper"]

            transition_probabilities_dict = dict()

            transition_probabilities_dict[P_EMISSION_DOWN] = (
                2 * nu**2 * f_ul / const.c.cgs.value**2 * (e_upper - e_lower)
            )

            transition_probabilities_dict[P_INTERNAL_DOWN] = (
                2 * nu**2 * f_ul / const.c.cgs.value**2 * e_lower
            )

            transition_probabilities_dict[P_INTERNAL_UP] = (
                f_lu * e_lower / (const.h.cgs.value * nu)
            )

            macro_atom.append(
                (
                    atomic_number,
                    ion_number,
                    level_number_upper,
                    level_number_lower,
                    line_id,
                    P_EMISSION_DOWN,
                    transition_probabilities_dict[P_EMISSION_DOWN],
                )
            )

            macro_atom.append(
                (
                    atomic_number,
                    ion_number,
                    level_number_upper,
                    level_number_lower,
                    line_id,
                    P_INTERNAL_DOWN,
                    transition_probabilities_dict[P_INTERNAL_DOWN],
                )
            )

            macro_atom.append(
                (
                    atomic_number,
                    ion_number,
                    level_number_lower,
                    level_number_upper,
                    line_id,
                    P_INTERNAL_UP,
                    transition_probabilities_dict[P_INTERNAL_UP],
                )
            )

        macro_atom = np.array(macro_atom, dtype=macro_atom_dtype)
        macro_atom = pd.DataFrame(macro_atom)

        macro_atom = macro_atom.sort_values(
            ["atomic_number", "ion_number", "source_level_number"]
        )

        self.macro_atom = macro_atom

    def create_macro_atom_references(self):
        """
        Create a DataFrame containing macro atom reference data.
        """
        macro_atom_references = self.levels.rename(
            columns={"level_number": "source_level_number"}
        ).loc[:, ["atomic_number", "ion_number", "source_level_number", "level_id"]]

        count_down = self.lines.groupby("upper_level_id").size()
        count_down.name = "count_down"

        count_up = self.lines.groupby("lower_level_id").size()
        count_up.name = "count_up"

        macro_atom_references = macro_atom_references.join(
            count_down, on="level_id"
        ).join(count_up, on="level_id")
        macro_atom_references = macro_atom_references.drop("level_id", axis=1)

        macro_atom_references = macro_atom_references.fillna(0)
        macro_atom_references["count_total"] = (
            2 * macro_atom_references["count_down"] + macro_atom_references["count_up"]
        )

        macro_atom_references["count_down"] = macro_atom_references[
            "count_down"
        ].astype(np.int64)

        macro_atom_references["count_up"] = macro_atom_references["count_up"].astype(
            np.int64
        )

        macro_atom_references["count_total"] = macro_atom_references[
            "count_total"
        ].astype(np.int64)

        self.macro_atom_references = macro_atom_references


    @property
    def macro_atom_prepared(self):
        """
        Prepare the DataFrame with macro atom data for TARDIS

        Returns
        -------
        macro_atom_prepared : pandas.DataFrame

        Notes
        -----
        Refer to the docs: https://tardis-sn.github.io/tardis/physics/setup/plasma/macroatom.html

        """

        macro_atom_prepared = self.macro_atom.loc[
            :,
            [
                "atomic_number",
                "ion_number",
                "source_level_number",
                "target_level_number",
                "transition_type",
                "transition_probability",
                "transition_line_id",
            ],
        ].copy()

        macro_atom_prepared = macro_atom_prepared.rename(
            columns={"target_level_number": "destination_level_number"}
        )

        macro_atom_prepared = macro_atom_prepared.reset_index(drop=True)

        return macro_atom_prepared

    @property
    def macro_atom_references_prepared(self):
        """
        Prepare the DataFrame with macro atom references for TARDIS

        Returns
        -------
        pandas.DataFrame

        """
        macro_atom_references_prepared = self.macro_atom_references.loc[
            :,
            [
                "atomic_number",
                "ion_number",
                "source_level_number",
                "count_down",
                "count_up",
                "count_total",
            ],
        ].copy()

        macro_atom_references_prepared = macro_atom_references_prepared.set_index(
            ["atomic_number", "ion_number", "source_level_number"]
        )

        return macro_atom_references_prepared