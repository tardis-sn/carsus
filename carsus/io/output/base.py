import re
import logging
import hashlib
import uuid
import pytz
import platform
import numpy as np
import pandas as pd
import astropy.units as u
import astropy.constants as const
from scipy import interpolate
from datetime import datetime
from carsus.util import (convert_wavelength_air2vacuum,
                         serialize_pandas_object,
                         hash_pandas_object)
from carsus.model import MEDIUM_VACUUM, MEDIUM_AIR

# TODO: pass GFALL_AIR_THRESHOLD as parameter
# [nm] wavelengths above this value are given in air
GFALL_AIR_THRESHOLD = 200

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

logger = logging.getLogger(__name__)


class TARDISAtomData:
    """
    Attributes
    ----------
    levels_prepared : pandas.DataFrame
    lines_prepared : pandas.DataFrame
    collisions_prepared: pandas.DataFrame
    macro_atom_prepared : pandas.DataFrame
    macro_atom_references_prepared : pandas.DataFrame

    Methods
    -------
    to_hdf(fname)
        Dump all attributes into an HDF5 file
    """

    def __init__(self,
                 atomic_weights,
                 ionization_energies,
                 gfall_reader,
                 zeta_data,
                 chianti_reader=None,
                 cmfgen_reader=None,
                 lines_loggf_threshold=-3,
                 levels_metastable_loggf_threshold=-3,
                 collisions_temperatures=None):

        # TODO: pass these params to the function as `gfall_params`
        self.levels_lines_param = {
            "levels_metastable_loggf_threshold":
            levels_metastable_loggf_threshold,
            "lines_loggf_threshold": lines_loggf_threshold
        }

        if collisions_temperatures is None:
            collisions_temperatures = np.arange(2000, 50000, 2000)
        else:
            collisions_temperatures = np.array(collisions_temperatures)

        self.collisions_param = {
            "temperatures": collisions_temperatures
        }

        self.atomic_weights = atomic_weights
        self.ionization_energies = ionization_energies
        self.ground_levels = ionization_energies.get_ground_levels()

        self.gfall_reader = gfall_reader
        self.chianti_reader = chianti_reader
        self.cmfgen_reader = cmfgen_reader
        self._manage_priorities()

        self.levels_all = self._get_all_levels_data()
        self.lines_all = self._get_all_lines_data()
        self._create_levels_lines(**self.levels_lines_param)
        self._create_macro_atom()
        self._create_macro_atom_references()

        if not self.chianti_reader.collisions.empty:
            self.collisions = self._create_collisions()

        self.zeta_data = zeta_data

    def _manage_priorities(self):
        """ Docstring """

        # Source ID's:
        # 1: NIST
        # 2: GFALL
        # 3: Knox Long's zeta
        # 4: Chianti Database
        # 5: CMFGEN

        gf_levels = self.gfall_reader.levels
        gf_levels['source'] = 2
        levels_ls = [gf_levels]

        if self.chianti_reader is not None:
            ch_levels = self.chianti_reader.levels
            ch_levels['source'] = 4
            levels_ls.append(ch_levels)

        if self.cmfgen_reader is not None:
            cf_levels = self.cmfgen_reader.levels
            cf_levels['source'] = 5
            levels_ls.append(cf_levels)

        levels = pd.concat(levels_ls)
        levels = levels.reset_index()
        levels = levels.set_index(['atomic_number', 'ion_charge'])
        
        df_list = []
        for ion in levels.index.unique():
            max_priority = levels.loc[ion]['priority'].max()
            df = levels.loc[ion][ levels.loc[ion]['priority'] == max_priority ]
            df_list.append(df)

        levels_uq = pd.concat(df_list)
        self.gfall_ions = levels_uq[ levels_uq['source'] == 2 ].index.unique()
        self.chianti_ions = levels_uq[ levels_uq['source'] == 4 ].index.unique()
        self.cmfgen_ions = levels_uq[ levels_uq['source'] == 5 ].index.unique()

    @staticmethod
    def get_lvl_index2id(df, levels_all, ion):
        df = df.reset_index()
        lvl_index2id = levels_all.set_index(
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

        return df

    @staticmethod
    def _create_artificial_fully_ionized(levels):
        """ Create artificial levels for fully ionized ions. """

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

    @staticmethod
    def calculate_collisional_strength(row, temperatures, 
                                       kb_ev, c_ul_temperature_cols):
        """
        Function to calculation upsilon from Burgess & Tully 1992 (TType 1 - 4; Eq. 23 - 38).
        """

        c = row["cups"]
        x_knots = np.linspace(0, 1, len(row["btemp"]))
        y_knots = row["bscups"]
        delta_e = row["delta_e"]
        g_u = row["g_u"]

        ttype = row["ttype"]
        if ttype > 5: 
            ttype -= 5

        kt = kb_ev * temperatures

        spline_tck = interpolate.splrep(x_knots, y_knots)

        if ttype == 1:
            x = 1 - np.log(c) / np.log(kt / delta_e + c)
            y_func = interpolate.splev(x, spline_tck)
            upsilon = y_func * np.log(kt / delta_e + np.exp(1))

        elif ttype == 2:
            x = (kt / delta_e) / (kt / delta_e + c)
            y_func = interpolate.splev(x, spline_tck)
            upsilon = y_func

        elif ttype == 3:
            x = (kt / delta_e) / (kt / delta_e + c)
            y_func = interpolate.splev(x, spline_tck)
            upsilon = y_func / (kt / delta_e + 1)

        elif ttype == 4:
            x = 1 - np.log(c) / np.log(kt / delta_e + c)
            y_func = interpolate.splev(x, spline_tck)
            upsilon = y_func * np.log(kt / delta_e + c)

        elif ttype == 5:
            raise ValueError('Not sure what to do with ttype=5')

        #### 1992A&A...254..436B Equation 20 & 22 #####
        collisional_ul_factor = 8.63e-6 * upsilon / (g_u * temperatures**.5)

        return pd.Series(data=collisional_ul_factor, index=c_ul_temperature_cols)

    def _get_all_levels_data(self):
        """ Returns the same output than `AtomData._get_all_levels_data()` 
        with `reset_index` method applied.
        """

        gf_levels = self.gfall_reader.levels.reset_index()
        gf_levels['source'] = 'gfall'

        if len(self.chianti_ions) > 0:
            logger.info('Ingesting levels from Chianti.')
            ch_levels = self.chianti_reader.levels.reset_index()
            ch_levels['source'] = 'chianti'
        else:
            ch_levels = pd.DataFrame(columns=gf_levels.columns)

        levels = pd.concat([gf_levels, ch_levels], sort=True)
        levels['g'] = 2*levels['j'] + 1
        levels['g'] = levels['g'].astype(np.int)
        levels = levels.drop(columns=['j', 'label', 'method'])
        levels = levels.reset_index(drop=True)
        levels = levels.rename(columns={'ion_charge': 'ion_number'})
        levels = levels[['atomic_number',
                         'ion_number', 'g', 'energy', 'source']]

        levels['energy'] = levels['energy'].apply(lambda x: x*u.Unit('cm-1'))
        levels['energy'] = levels['energy'].apply(
            lambda x: x.to(u.eV, equivalencies=u.spectral()))
        levels['energy'] = levels['energy'].apply(lambda x: x.value)

        ground_levels = self.ground_levels
        ground_levels.rename(
            columns={'ion_charge': 'ion_number'}, inplace=True)
        ground_levels['source'] = 'nist'

        levels = pd.concat([ground_levels, levels], sort=True)
        levels['level_id'] = range(1, len(levels)+1)
        levels = levels.set_index('level_id')

        # Deliberately keep the "duplicated" Chianti levels.
        # These levels are not strictly duplicated: same energy
        # for different configurations. 
        # 
        # e.g. ChiantiIonReader('h_1')
        #
        # In fact, the following code should only remove the du-
        # plicated ground levels. Other duplicated levels should
        # be removed at the reader stage.
        #
        # TODO: a more clear way to get the same result could be: 
        # "keep only zero energy levels from NIST source".

        mask = (levels['source'] != 'chianti') & (
            levels[['atomic_number', 'ion_number',
                    'energy', 'g']].duplicated(keep='last'))	
        levels = levels[~mask]

        # Keep higher priority levels over GFALL: if levels with
        # different source than 'gfall' made to this point should
        # be kept.
        for ion in self.chianti_ions:
            mask = (levels['source'] == 'gfall') & (
                levels['atomic_number'] == ion[0]) & (
                    levels['ion_number'] == ion[1])
            levels.drop(levels[mask].index, inplace=True)

        levels = levels[['atomic_number',
                         'ion_number', 'g', 'energy', 'source']]

        levels = levels.reset_index()

        return levels


    def _get_all_lines_data(self):
        """ Returns the same output than `AtomData._get_all_lines_data()` """

        gf = self.gfall_reader
        ch = self.chianti_reader

        start = 1
        gf_list = []
        logger.info('Ingesting lines from GFALL.')
        for ion in self.gfall_ions:

            try:
                df = gf.lines.loc[ion]

                # To match `line_id` field with the old API we keep
                # track of how many GFALL lines we are skipping.
                if ion in self.chianti_ions:
                    df['line_id'] = range(start, len(df) + start)
                    start += len(df)
                    continue

                else:
                    df['line_id'] = range(start, len(df) + start)
                    start += len(df)

            except (KeyError, TypeError):
                continue

            df = self.get_lvl_index2id(df, self.levels_all, ion)
            df['source'] = 'gfall'
            gf_list.append(df)

        ch_list = []
        logger.info('Ingesting lines from Chianti.')
        for ion in self.chianti_ions:

            df = ch.lines.loc[ion]
            df['line_id'] = range(start, len(df) + start)
            start = len(df) + start

            df = self.get_lvl_index2id(df, self.levels_all, ion)
            df['source'] = 'chianti'
            ch_list.append(df)

        df_list = gf_list + ch_list
        lines = pd.concat(df_list, sort=True)
        lines['loggf'] = lines['gf'].apply(np.log10)

        lines.set_index('line_id', inplace=True)
        lines.drop(columns=['energy_upper', 'j_upper', 'energy_lower',
                            'j_lower', 'level_index_lower',
                            'level_index_upper'], inplace=True)

        lines.loc[lines['wavelength'] <=
                  GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_VACUUM
        lines.loc[lines['wavelength'] >
                  GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_AIR

        air_mask = lines['medium'] == MEDIUM_AIR
        gfall_mask = lines['source'] == 'gfall'
        chianti_mask = lines['source'] == 'chianti'

        lines.loc[gfall_mask, 'wavelength'] = lines.loc[
            gfall_mask, 'wavelength'].apply(lambda x: x*u.nm)
        lines.loc[chianti_mask, 'wavelength'] = lines.loc[
            chianti_mask, 'wavelength'].apply(lambda x: x*u.angstrom)
        lines['wavelength'] = lines['wavelength'].apply(
            lambda x: x.to('angstrom'))
        lines['wavelength'] = lines['wavelength'].apply(lambda x: x.value)

        # Why not for Chianti?
        lines.loc[air_mask & gfall_mask,
                  'wavelength'] = convert_wavelength_air2vacuum(
            lines.loc[air_mask, 'wavelength'])

        lines.drop(columns=['medium'], inplace=True)
        lines = lines[['lower_level_id', 'upper_level_id',
                       'wavelength', 'gf', 'loggf', 'source']]

        return lines

    def _create_levels_lines(self, lines_loggf_threshold=-3,
                             levels_metastable_loggf_threshold=-3):
        """ Returns almost the same output than`AtomData.create_levels_lines` method """

        levels_all = self.levels_all
        lines_all = self.lines_all
        ionization_energies = self.ionization_energies.base.reset_index()
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

    def _create_collisions(self):
        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")

        ch_list = []
        ch = self.chianti_reader

        logger.info('Ingesting collisions from Chianti')
        for ion in self.chianti_ions:

            df = ch.collisions.loc[ion]
            df = self.get_lvl_index2id(df, self.levels_all, ion)
            ch_list.append(df)

        collisions = pd.concat(ch_list, sort=True)
        collisions['source'] = 'chianti'

        # Keep this value to compare against SQL
        collisions['ds_id'] = 4

        # `e_col_id` number starts after the last line id
        start = self.lines_all.index[-1] + 1
        collisions['e_col_id'] = range(start, start + len(collisions))
        collisions = collisions.reset_index()

        # Join atomic_number, ion_number, level_number_lower, level_number_upper
        lower_levels = levels.rename(columns={"level_number": "level_number_lower", "g": "g_l", "energy": "energy_lower"}). \
                              loc[:, ["atomic_number", "ion_number", "level_number_lower", "g_l", "energy_lower"]]
        upper_levels = levels.rename(columns={"level_number": "level_number_upper", "g": "g_u", "energy": "energy_upper"}). \
                              loc[:, ["level_number_upper", "g_u", "energy_upper"]]

        collisions = collisions.join(lower_levels, on="lower_level_id").join(
                                            upper_levels, on="upper_level_id")

        # Calculate delta_e
        kb_ev = const.k_B.cgs.to('eV / K').value
        collisions["delta_e"] = (collisions["energy_upper"] - collisions["energy_lower"])/kb_ev

        # Calculate g_ratio
        collisions["g_ratio"] = collisions["g_l"] / collisions["g_u"]

        temperatures = self.collisions_param['temperatures']

        # Derive columns for collisional strengths
        c_ul_temperature_cols = ['t{:06d}'.format(t) for t in temperatures]

        collisions = collisions.rename(columns={'ion_charge': 'ion_number',
                                                'temperatures': 'btemp',
                                                'collision_strengths': 'bscups'})
        collisions = collisions[['e_col_id', 'lower_level_id',
                                 'upper_level_id', 'ds_id',
                                 'btemp', 'bscups', 'ttype', 'cups',
                                 'gf', 'atomic_number', 'ion_number',
                                 'level_number_lower', 'g_l',
                                 'energy_lower', 'level_number_upper', 
                                 'g_u', 'energy_upper', 'delta_e', 
                                    'g_ratio']]

        collisional_ul_factors = collisions.apply(self.calculate_collisional_strength, 
                                                  axis=1, args=(temperatures, kb_ev, 
                                                                c_ul_temperature_cols))

        collisions = collisions.join(collisional_ul_factors)
        collisions = collisions.set_index('e_col_id')

        return collisions

    @property
    def levels_prepared(self):
        """
        Prepare the DataFrame with levels for TARDIS.

        Returns
        -------
        pandas.DataFrame
        """

        levels_prepared = self.levels.loc[:, [
            "atomic_number", "ion_number", "level_number",
            "energy", "g", "metastable"]].copy()

        levels_prepared.set_index(
            ["atomic_number", "ion_number", "level_number"], inplace=True)

        return levels_prepared

    @property
    def lines_prepared(self):
        """
        Prepare the DataFrame with lines for TARDIS.

        Returns
        -------
        pandas.DataFrame
        """

        lines_prepared = self.lines.loc[:, [
            "line_id", "wavelength", "atomic_number", "ion_number",
            "f_ul", "f_lu", "level_number_lower", "level_number_upper",
            "nu", "B_lu", "B_ul", "A_ul"]].copy()

        # TODO: store units in metadata
        # wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1],
        # B_ul[cm^3 s^-2 erg^-1], B_lu[cm^3 s^-2 erg^-1],
        # A_ul[1/s].

        lines_prepared.set_index([
            "atomic_number", "ion_number",
            "level_number_lower", "level_number_upper"], inplace=True)

        return lines_prepared

    @property
    def collisions_prepared(self):
        """
        Prepare the DataFrame with electron collisions for TARDIS.

        Returns
        -------
        pandas.DataFrame
        """

        collisions_columns = ['atomic_number', 'ion_number', 'level_number_upper',
                              'level_number_lower', 'g_ratio', 'delta_e'] + \
                              sorted([col for col in self.collisions.columns if re.match('^t\d+$', col)])

        collisions_prepared = self.collisions.loc[:, collisions_columns].copy()

        collisions_prepared.set_index([
                    "atomic_number",
                    "ion_number",
                    "level_number_lower",
                    "level_number_upper"],
                    inplace=True)

        return collisions_prepared

    def _create_macro_atom(self):
        """
        Create a DataFrame containing macro atom data.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Refer to the docs: https://tardis-sn.github.io/tardis/physics/plasma/macroatom.html
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

            transition_probabilities_dict = dict()
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
        
        Notes
        -----
        Refer to the docs: https://tardis-sn.github.io/tardis/physics/plasma/macroatom.html
        """

        macro_atom_prepared = self.macro_atom.loc[:, [
            "atomic_number",
            "ion_number", "source_level_number", "target_level_number",
            "transition_type", "transition_probability",
            "transition_line_id"]].copy()

        macro_atom_prepared = macro_atom_prepared.rename(columns={
            "target_level_number": "destination_level_number"})

        macro_atom_prepared = macro_atom_prepared.reset_index(drop=True)

        return macro_atom_prepared

    def _create_macro_atom_references(self):
        """
        Create a DataFrame containing macro atom reference data.
    
        Returns
        -------
        pandas.DataFrame
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
        pandas.DataFrame
        """
        macro_atom_references_prepared = self.macro_atom_references.loc[:, [
            "atomic_number", "ion_number", "source_level_number", "count_down",
            "count_up", "count_total"]].copy()

        macro_atom_references_prepared.set_index(
            ['atomic_number', 'ion_number', 'source_level_number'],
            inplace=True)

        return macro_atom_references_prepared

    def to_hdf(self, fname):
        """
        Dump the `base` attribute into an HDF5 file
        
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file
        """

        with pd.HDFStore(fname, 'w') as f:
            f.put('/atom_data', self.atomic_weights.base)
            f.put('/ionization_data', self.ionization_energies.base)
            f.put('/zeta_data', self.zeta_data.base)
            f.put('/levels', self.levels_prepared)
            f.put('/lines', self.lines_prepared)
            f.put('/collision_data', self.collisions_prepared)
            f.put('/macro_atom_data', self.macro_atom_prepared)
            f.put('collision_data_temperatures', 
                   pd.Series(self.collisions_param['temperatures']))
            f.put('/macro_atom_references',
                  self.macro_atom_references_prepared)

            meta = []
            md5_hash = hashlib.md5()
            for key in f.keys():
                # Update the total MD5 sum
                md5_hash.update(serialize_pandas_object(f[key]).to_buffer())
                
                # Save the individual Series/DataFrame MD5
                md5 = hash_pandas_object(f[key])
                meta.append(('md5sum', key.lstrip('/'), md5[:20]))

            # Save datasets versions
            meta.append(('datasets', 'nist_weights', 
                         self.atomic_weights.version))
            meta.append(('datasets', 'nist_spectra', 
                         self.ionization_energies.version))

            meta.append(('datasets', 'gfall.dat',
                         self.gfall_reader.md5[:20]))

            if self.chianti_reader is not None:
                meta.append(('datasets', 'chianti_data', 
                             self.chianti_reader.version))

            # Save relevant package versions
            meta.append(('software', 'python', platform.python_version()))
            imports = ['carsus', 'astropy', 'numpy', 'pandas', 'pyarrow', 
                       'tables', 'ChiantiPy']
 
            for package in imports:
                meta.append(('software', package,
                             __import__(package).__version__))

            meta_df = pd.DataFrame.from_records(meta, columns=['field', 'key',
                        'value'], index=['field', 'key'])

            uuid1 = uuid.uuid1().hex

            logger.info(f"Signing TARDISAtomData.")
            logger.info(f"MD5: {md5_hash.hexdigest()}")
            logger.info(f"UUID1: {uuid1}")

            f.root._v_attrs['md5'] = md5_hash.hexdigest()
            f.root._v_attrs['uuid1'] = uuid1
            f.put('/meta', meta_df)

            utc = pytz.timezone('UTC')
            timestamp = datetime.now(utc).strftime("%b %d, %Y %H:%M:%S UTC")
            f.root._v_attrs['date'] = timestamp

            self.meta = meta_df
            
            return
