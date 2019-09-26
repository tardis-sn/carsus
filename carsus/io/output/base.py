import numpy as np
import pandas as pd
import hashlib
import uuid
from carsus.util import parse_selected_species, convert_wavelength_air2vacuum
from carsus.model import MEDIUM_VACUUM, MEDIUM_AIR
from astropy import units as u
from astropy import constants as const

# [nm], wavelengths above this value are given in air
# TODO: pass GFALL_AIR_THRESHOLD as parameter
GFALL_AIR_THRESHOLD = 200

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1


class TARDISAtomData:

    """
    Attributes
    ----------
    levels_prepared : pandas.DataFrame
    lines_prepared : pandas.DataFrame
    macro_atom_prepared : pandas.DataFrame
    macro_atom_references_prepared : pandas.DataFrame


    Methods
    -------
    to_hdf(fname)
        Dump all attributes into an HDF5 file
    """

    def __init__(self, gfall_reader, ionization_energies, ions,
                 lines_loggf_threshold=-3,
                 levels_metastable_loggf_threshold=-3):

        self.levels_lines_param = {
            "levels_metastable_loggf_threshold":
            levels_metastable_loggf_threshold,
            "lines_loggf_threshold": lines_loggf_threshold
        }

        self.ions = parse_selected_species(ions)
        self.gfall_reader = gfall_reader

        self.ionization_energies = ionization_energies.base
        self.ground_levels = ionization_energies.get_ground_levels()

        self.levels_all = self._get_all_levels_data().reset_index()
        self.lines_all = self._get_all_lines_data(self.levels_all)

        self._create_levels_lines(**self.levels_lines_param)
        self._create_macro_atom()
        self._create_macro_atom_references()

    @staticmethod
    def _create_artificial_fully_ionized(levels):
        """ Create artificial levels for fully ionized ions """
        fully_ionized_levels = list()

        for atomic_number, _ in levels.groupby("atomic_number"):
            fully_ionized_levels.append(
                (-1, atomic_number, atomic_number, 0, 0.0, 1, True)
            )

        levels_columns = ["level_id", "atomic_number",
                          "ion_number", "level_number",
                          "energy", "g", "metastable"]
        fully_ionized_levels_dtypes = [
            (key, levels.dtypes[key]) for key in levels_columns]

        fully_ionized_levels = np.array(
            fully_ionized_levels, dtype=fully_ionized_levels_dtypes)

        return pd.DataFrame(data=fully_ionized_levels)

    @staticmethod
    def _create_metastable_flags(levels, lines,
                                 levels_metastable_loggf_threshold=-3):
        # Filter lines on the loggf threshold value
        metastable_lines = lines.loc[lines["loggf"]
                                     > levels_metastable_loggf_threshold]

        # Count the remaining strong transitions
        metastable_lines_grouped = metastable_lines.groupby("upper_level_id")
        metastable_counts = metastable_lines_grouped["upper_level_id"].count()
        metastable_counts.name = "metastable_counts"

        # If there are no strong transitions for a level (the count is NaN)
        # then the metastable flag is True else (the count is a natural number)
        # the metastable flag is False
        levels = levels.join(metastable_counts)
        metastable_flags = levels["metastable_counts"].isnull()
        metastable_flags.name = "metastable"

        return metastable_flags

    @staticmethod
    def _create_einstein_coeff(lines):
        einstein_coeff = (4 * np.pi ** 2 * const.e.gauss.value **
                          2) / (const.m_e.cgs.value * const.c.cgs.value)
        lines['B_lu'] = einstein_coeff * lines['f_lu'] / \
            (const.h.cgs.value * lines['nu'])
        lines['B_ul'] = einstein_coeff * lines['f_ul'] / \
            (const.h.cgs.value * lines['nu'])
        lines['A_ul'] = 2 * einstein_coeff * lines['nu'] ** 2 / \
            const.c.cgs.value ** 2 * lines['f_ul']

    def _get_all_levels_data(self):
        """ Returns the same output than `AtomData._get_all_levels_data()` """
        gf = self.gfall_reader
        df_list = []

        for ion in self.ions:
            try:
                df = gf.levels.loc[ion].copy()

            except (KeyError, TypeError):
                continue

            df['atomic_number'] = ion[0]
            df['ion_number'] = ion[1]
            df_list.append(df)

        levels = pd.concat(df_list, sort=True)
        levels['g'] = 2*levels['j'] + 1
        levels['g'] = levels['g'].astype(np.int)
        levels = levels.drop(columns=['j', 'label', 'method'])
        levels = levels.reset_index(drop=True)
        levels = levels[['atomic_number', 'ion_number', 'g', 'energy']]

        levels['energy'] = levels['energy'].apply(lambda x: x*u.Unit('cm-1'))
        levels['energy'] = levels['energy'].apply(
            lambda x: x.to(u.eV, equivalencies=u.spectral()))
        levels['energy'] = levels['energy'].apply(lambda x: x.value)

        ground_levels = self.ground_levels
        ground_levels.rename(
            columns={'ion_charge': 'ion_number'}, inplace=True)

        # Fixes Ar II duplicated ground level. For Kurucz, ground state
        # has g=2, for NIST has g=4. We keep Kurucz.

        mask = (ground_levels['atomic_number'] == 18) & (
            ground_levels['ion_number'] == 1)
        ground_levels.loc[mask, 'g'] = 2

        levels = pd.concat([ground_levels, levels], sort=True)
        levels['level_id'] = range(1, len(levels)+1)
        levels = levels.set_index('level_id')
        levels = levels.drop_duplicates(keep='last')

        return levels

    def _get_all_lines_data(self, levels):
        """ Returns the same output than `AtomData._get_all_lines_data()` """
        gf = self.gfall_reader
        df_list = []

        for ion in self.ions:

            try:
                df = gf.lines.loc[ion]

            except (KeyError, TypeError):
                continue

            df = df.reset_index()
            lvl_index2id = levels.set_index(
                ['atomic_number', 'ion_number']).loc[ion]
            lvl_index2id = lvl_index2id.reset_index()
            lvl_index2id = lvl_index2id[['level_id']]

            lower_level_id = []
            upper_level_id = []
            for i, row in df.iterrows():

                llid = int(row['level_index_lower'])
                ulid = int(row['level_index_upper'])

                upper = int(lvl_index2id.loc[ulid])
                lower = int(lvl_index2id.loc[llid])

                lower_level_id.append(lower)
                upper_level_id.append(upper)

            df['lower_level_id'] = pd.Series(lower_level_id)
            df['upper_level_id'] = pd.Series(upper_level_id)
            df_list.append(df)

        lines = pd.concat(df_list, sort=True)
        lines['line_id'] = range(1, len(lines)+1)
        lines['loggf'] = lines['gf'].apply(np.log10)

        lines.set_index('line_id', inplace=True)
        lines.drop(columns=['energy_upper', 'j_upper', 'energy_lower',
                            'j_lower', 'level_index_lower',
                            'level_index_upper'], inplace=True)

        lines.loc[lines['wavelength'] <=
                  GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_VACUUM
        lines.loc[lines['wavelength'] >
                  GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_AIR
        lines['wavelength'] = lines['wavelength'].apply(lambda x: x*u.nm)
        lines['wavelength'] = lines['wavelength'].apply(
            lambda x: x.to('angstrom'))
        lines['wavelength'] = lines['wavelength'].apply(lambda x: x.value)

        air_mask = lines['medium'] == MEDIUM_AIR
        lines.loc[air_mask, 'wavelength'] = convert_wavelength_air2vacuum(
            lines.loc[air_mask, 'wavelength'])
        lines.drop(columns=['medium'], inplace=True)
        lines = lines[['lower_level_id', 'upper_level_id',
                       'wavelength', 'gf', 'loggf']]

        return lines

    def _create_levels_lines(self, lines_loggf_threshold=-3,
                             levels_metastable_loggf_threshold=-3):
        """ Returns almost the same output than
        `AtomData.create_levels_lines` method """
        levels_all = self.levels_all
        lines_all = self.lines_all
        ionization_energies = self.ionization_energies.reset_index()
        ionization_energies['ion_number'] -= 1

        # Culling autoionization levels
        levels_w_ionization_energies = pd.merge(levels_all,
                                                ionization_energies,
                                                how='left',
                                                on=["atomic_number",
                                                    "ion_number"])
        mask = levels_w_ionization_energies["energy"] < \
            levels_w_ionization_energies["ionization_energy"]
        levels = levels_w_ionization_energies[mask].copy()
        levels = levels.set_index('level_id').sort_values(
            by=['atomic_number', 'ion_number'])
        levels = levels.drop(columns='ionization_energy')

        # Clean lines
        lines = lines_all.join(pd.DataFrame(index=levels.index),
                               on="lower_level_id", how="inner").\
            join(pd.DataFrame(index=levels.index),
                 on="upper_level_id", how="inner")

        # Culling lines with low gf values
        lines = lines.loc[lines["loggf"] > lines_loggf_threshold]

        # Do not clean levels that don't exist in lines

        # Create the metastable flags for levels
        levels["metastable"] = \
            self._create_metastable_flags(levels, lines_all,
                                          levels_metastable_loggf_threshold)

        # Create levels numbers
        levels = levels.sort_values(
            ["atomic_number", "ion_number", "energy", "g"])
        levels["level_number"] = levels.groupby(['atomic_number',
                                                 'ion_number'])['energy']. \
            transform(lambda x: np.arange(len(x))).values
        levels["level_number"] = levels["level_number"].astype(np.int)

        levels = levels[['atomic_number', 'energy', 'g', 'ion_number',
                         'level_number', 'metastable']]

        # Join atomic_number, ion_number, level_number_lower,
        # level_number_upper on lines
        lower_levels = levels.rename(
            columns={
                "level_number": "level_number_lower",
                "g": "g_l"}
        ).loc[:, ["atomic_number", "ion_number", "level_number_lower", "g_l"]]
        upper_levels = levels.rename(
            columns={
                "level_number": "level_number_upper",
                "g": "g_u"}
        ).loc[:, ["level_number_upper", "g_u"]]
        lines = lines.join(lower_levels, on="lower_level_id").join(
            upper_levels, on="upper_level_id")

        # Calculate absorption oscillator strength f_lu and emission
        # oscillator strength f_ul
        lines["f_lu"] = lines["gf"] / lines["g_l"]
        lines["f_ul"] = lines["gf"] / lines["g_u"]

        # Calculate frequency
        lines['nu'] = u.Quantity(
            lines['wavelength'], 'angstrom').to('Hz', u.spectral())

        # Create Einstein coefficients
        self._create_einstein_coeff(lines)

        # Reset indexes because `level_id` cannot be an index once we
        # add artificial levels for fully ionized ions that don't have ids (-1)
        lines = lines.reset_index()
        levels = levels.reset_index()

        # Create and append artificial levels for fully ionized ions
        artificial_fully_ionized_levels = \
            self._create_artificial_fully_ionized(levels)
        levels = levels.append(
            artificial_fully_ionized_levels, ignore_index=True)
        levels = levels.sort_values(
            ["atomic_number", "ion_number", "level_number"])

        self.lines = lines
        self.levels = levels

    @property
    def levels_prepared(self):
        """
        Prepare the DataFrame with levels for TARDIS
        Returns
        -------
        levels_prepared: pandas.DataFrame
            DataFrame with:
                index: none;
                columns: atomic_number, ion_number, level_number,
                            energy[eV], g[1], metastable.
        """

        levels_prepared = self.levels.loc[:, [
            "atomic_number", "ion_number", "level_number",
            "energy", "g", "metastable"]].copy()

        # Set index
        levels_prepared.set_index(
            ["atomic_number", "ion_number", "level_number"], inplace=True)

        return levels_prepared

    @property
    def lines_prepared(self):
        """
            Prepare the DataFrame with lines for TARDIS
            Returns
            -------
            lines_prepared : pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: line_id, atomic_number, ion_number,
                             level_number_lower, level_number_upper,
                             wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1],
                             B_ul[cm^3 s^-2 erg^-1], B_lu[cm^3 s^-2 erg^-1],
                             A_ul[1/s].
        """

        lines_prepared = self.lines.loc[:, [
            "line_id", "wavelength", "atomic_number", "ion_number",
            "f_ul", "f_lu", "level_number_lower", "level_number_upper",
            "nu", "B_lu", "B_ul", "A_ul"]].copy()

        # Set the index
        lines_prepared.set_index([
            "atomic_number", "ion_number",
            "level_number_lower", "level_number_upper"], inplace=True)

        return lines_prepared

    def _create_macro_atom(self):
        """
            Create a DataFrame containing *macro atom* data.
            Returns
            -------
            macro_atom: pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number,
                             target_level_number, transition_line_id,
                             transition_type, transition_probability.
            Notes:
                Refer to the docs:
                https://tardis-sn.github.io/tardis/physics/plasma/macroatom.html
        """
        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"]
                                 != -1].set_index("level_id")

        lvl_energy_lower = levels.rename(
            columns={"energy": "energy_lower"}).loc[:, ["energy_lower"]]
        lvl_energy_upper = levels.rename(
            columns={"energy": "energy_upper"}).loc[:, ["energy_upper"]]

        lines = self.lines.set_index("line_id")
        lines = lines.join(lvl_energy_lower, on="lower_level_id").join(
            lvl_energy_upper, on="upper_level_id")

        macro_atom = list()
        macro_atom_dtype = [("atomic_number", np.int), ("ion_number", np.int),
                            ("source_level_number",
                             np.int), ("target_level_number", np.int),
                            ("transition_line_id",
                             np.int), ("transition_type", np.int),
                            ("transition_probability", np.float)]

        for line_id, row in lines.iterrows():
            atomic_number, ion_number = row["atomic_number"], row["ion_number"]
            level_number_lower, level_number_upper = \
                row["level_number_lower"], row["level_number_upper"]
            nu = row["nu"]
            f_ul, f_lu = row["f_ul"], row["f_lu"]
            e_lower, e_upper = row["energy_lower"], row["energy_upper"]

            transition_probabilities_dict = dict()  # type : probability
            transition_probabilities_dict[P_EMISSION_DOWN] = 2 * \
                nu**2 * f_ul / const.c.cgs.value**2 * (e_upper - e_lower)
            transition_probabilities_dict[P_INTERNAL_DOWN] = 2 * \
                nu**2 * f_ul / const.c.cgs.value**2 * e_lower
            transition_probabilities_dict[P_INTERNAL_UP] = f_lu * \
                e_lower / (const.h.cgs.value * nu)

            macro_atom.append((atomic_number, ion_number, level_number_upper,
                               level_number_lower, line_id, P_EMISSION_DOWN,
                               transition_probabilities_dict[P_EMISSION_DOWN]))
            macro_atom.append((atomic_number, ion_number, level_number_upper,
                               level_number_lower, line_id, P_INTERNAL_DOWN,
                               transition_probabilities_dict[P_INTERNAL_DOWN]))
            macro_atom.append((atomic_number, ion_number, level_number_lower,
                               level_number_upper, line_id, P_INTERNAL_UP,
                               transition_probabilities_dict[P_INTERNAL_UP]))

        macro_atom = np.array(macro_atom, dtype=macro_atom_dtype)
        macro_atom = pd.DataFrame(macro_atom)

        macro_atom = macro_atom.sort_values(
            ["atomic_number", "ion_number", "source_level_number"])

        self.macro_atom = macro_atom

    @property
    def macro_atom_prepared(self):
        """
            Prepare the DataFrame with macro atom data for TARDIS
            Returns
            -------
            macro_atom_prepared : pandas.DataFrame
                DataFrame with the *macro atom data* with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number,
                             destination_level_number, transition_line_id
                             transition_type, transition_probability.
            Notes:
                Refer to the docs:
                https://tardis-sn.github.io/tardis/physics/plasma/macroatom.html
        """

        macro_atom_prepared = self.macro_atom.loc[:, [
            "atomic_number",
            "ion_number", "source_level_number", "target_level_number",
            "transition_type", "transition_probability",
            "transition_line_id"]].copy()

        # ToDo: choose between `target_level_number` and
        # `destination_level_number` Rename `target_level_number` to
        # `destination_level_number` used in TARDIS Personally, I think
        # `target_level_number` is better so I use it in Carsus.
        macro_atom_prepared = macro_atom_prepared.rename(columns={
            "target_level_number": "destination_level_number"})

        macro_atom_prepared = macro_atom_prepared.reset_index(drop=True)

        return macro_atom_prepared

    def _create_macro_atom_references(self):
        """
            Create a DataFrame containing *macro atom reference* data.
            Returns
            -------
            macro_atom_reference : pandas.DataFrame
                DataFrame with:
                index: no index;
                and columns: atomic_number, ion_number, source_level_number,
                count_down, count_up, count_total
        """
        macro_atom_references = self.levels.rename(
            columns={"level_number": "source_level_number"}).\
            loc[:, ["atomic_number", "ion_number",
                    "source_level_number", "level_id"]]

        count_down = self.lines.groupby("upper_level_id").size()
        count_down.name = "count_down"

        count_up = self.lines.groupby("lower_level_id").size()
        count_up.name = "count_up"

        macro_atom_references = macro_atom_references.join(
            count_down, on="level_id").join(count_up, on="level_id")
        macro_atom_references = macro_atom_references.drop("level_id", axis=1)

        macro_atom_references = macro_atom_references.fillna(0)
        macro_atom_references["count_total"] = 2 * \
            macro_atom_references["count_down"] + \
            macro_atom_references["count_up"]

        # Convert to int
        macro_atom_references["count_down"] = \
            macro_atom_references["count_down"].astype(np.int)
        macro_atom_references["count_up"] = \
            macro_atom_references["count_up"].astype(np.int)
        macro_atom_references["count_total"] = \
            macro_atom_references["count_total"].astype(np.int)

        self.macro_atom_references = macro_atom_references

    @property
    def macro_atom_references_prepared(self):
        """
            Prepare the DataFrame with macro atom references for TARDIS
            Returns
            -------
            macro_atom_references_prepared : pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number,
                             count_down, count_up, count_total.
        """
        macro_atom_references_prepared = self.macro_atom_references.loc[:, [
            "atomic_number", "ion_number", "source_level_number", "count_down",
            "count_up", "count_total"]].copy()

        macro_atom_references_prepared.set_index(
            ['atomic_number', 'ion_number', 'source_level_number'],
            inplace=True)

        return macro_atom_references_prepared

    def to_hdf(self, fname):
        """Dump the `base` attribute into an HDF5 file
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """

        with pd.HDFStore(fname, 'a') as f:
            f.put('/levels', self.levels_prepared)
            f.put('/lines', self.lines_prepared)
            f.put('/macro_atom_data', self.macro_atom_prepared)
            f.put('/macro_atom_references',
                  self.macro_atom_references_prepared)

            md5_hash = hashlib.md5()
            for key in f.keys():
                tmp = np.ascontiguousarray(f[key].values.data)
                md5_hash.update(tmp)

            uuid1 = uuid.uuid1().hex

            print("Signing AtomData: \nMD5: {}\nUUID1: {}".format(
                md5_hash.hexdigest(), uuid1))

            f.root._v_attrs['md5'] = md5_hash.hexdigest().encode('ascii')
            f.root._v_attrs['uuid1'] = uuid1.encode('ascii')
