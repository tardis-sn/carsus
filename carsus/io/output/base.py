import copy
import logging
import re

import astropy.constants as const
import astropy.units as u
import numpy as np
import pandas as pd
from scipy import interpolate

from carsus.model import MEDIUM_AIR, MEDIUM_VACUUM
from carsus.util import (convert_atomic_number2symbol,
                         convert_wavelength_air2vacuum, hash_pandas_object,
                         serialize_pandas_object)

# Wavelengths above this value are given in air
GFALL_AIR_THRESHOLD = 2000 * u.AA
P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

logger = logging.getLogger(__name__)


class TARDISAtomData:
    """
    Attributes
    ----------

    levels : pandas.DataFrame
    lines : pandas.DataFrame
    collisions : pandas.DataFrame
    macro_atom : pandas.DataFrame
    macro_atom_references : pandas.DataFrame
    levels_prepared : pandas.DataFrame
    lines_prepared : pandas.DataFrame
    collisions_prepared: pandas.DataFrame
    macro_atom_prepared : pandas.DataFrame
    macro_atom_references_prepared : pandas.DataFrame

    """

    def __init__(self,
                 atomic_weights,
                 ionization_energies,
                 gfall_reader,
                 zeta_data,
                 chianti_reader=None,
                 cmfgen_reader=None,
                 levels_lines_param={"levels_metastable_loggf_threshold": -3,
                                     "lines_loggf_threshold": -3},
                 collisions_param={"temperatures": np.arange(2000, 50000, 2000)}
                ):

        self.atomic_weights = atomic_weights

        if (cmfgen_reader is not None) and hasattr(cmfgen_reader, 'ionization_energies'):
            combined_ionization_energies = copy.deepcopy(ionization_energies)
            combined_ionization_energies.base = cmfgen_reader.ionization_energies.combine_first(ionization_energies.base)
            self.ionization_energies = combined_ionization_energies
        else:
            self.ionization_energies = ionization_energies

        self.gfall_reader = gfall_reader
        self.zeta_data = zeta_data
        self.chianti_reader = chianti_reader
        self.cmfgen_reader = cmfgen_reader
        self.levels_lines_param = levels_lines_param
        self.collisions_param = collisions_param

        self.levels_all = self._get_all_levels_data()
        self.lines_all = self._get_all_lines_data()
        self.levels, self.lines = self.create_levels_lines(**levels_lines_param)
        self.create_macro_atom()
        self.create_macro_atom_references()

        if ((cmfgen_reader is not None) and hasattr(cmfgen_reader, "collisions")) and (
            (chianti_reader is not None) and hasattr(chianti_reader, "collisions")
        ):
            raise ValueError(
                "Both Chianti and CMFGEN readers contain the collisions dataframe. "
                "Please set collisions=True in one or the other but not both."
            )
        else:
            if hasattr(cmfgen_reader, "collisions"):
                collisions = cmfgen_reader.collisions
                collisions.index = collisions.index.rename(
                    ['atomic_number', 'ion_number', 'level_number_lower', 'level_number_upper']
                )

                self.collisions = collisions
                self.collisions_metadata = cmfgen_reader.collisional_metadata

            elif hasattr(chianti_reader, "collisions"):
                self.collisions = self.create_collisions(**collisions_param)
                self.collisions_metadata = pd.Series({
                    "temperatures": collisions_param["temperatures"],
                    "dataset": ["chianti"],
                    "info": None
                })
            else:
                logger.warning("No source of collisions was selected.")

        if (cmfgen_reader is not None) and hasattr(cmfgen_reader, 'cross_sections'):
            self.cross_sections = self.create_cross_sections()

        logger.info('Finished.')

    @staticmethod
    def solve_priorities(levels):
        """ 
        Returns a list of unique species per data source. 
        

        Notes
        -----

        The `ds_id` field is the data source identifier.

        1 : NIST
        2 : GFALL
        3 : Knox Long's Zeta
        4 : Chianti Database
        5 : CMFGEN

        """
        levels = levels.set_index(['atomic_number', 'ion_number'])
        levels = levels.sort_index()  # To supress warnings

        lvl_list = []
        for ion in levels.index.unique():
            max_priority = levels.loc[ion, 'priority'].max()
            lvl = levels.loc[ion][ levels.loc[ion, 'priority'] == max_priority ]
            lvl_list.append(lvl)

        levels_uq = pd.concat(lvl_list, sort=True)
        gfall_ions = levels_uq[ levels_uq['ds_id'] == 2 ].index.unique()
        chianti_ions = levels_uq[ levels_uq['ds_id'] == 4 ].index.unique()
        cmfgen_ions = levels_uq[ levels_uq['ds_id'] == 5 ].index.unique()

        assert set(gfall_ions).intersection(set(chianti_ions))\
                                .intersection(set(cmfgen_ions)) == set([])

        return gfall_ions, chianti_ions, cmfgen_ions

    @staticmethod
    def get_lvl_index2id(df, levels_all):
        """
        Matches level indexes with level IDs for a given DataFrame. 
        
        """
        # TODO: re-write this method without a for loop
        ion = df.index.unique()
        lvl_index2id = levels_all.set_index(
                            ['atomic_number', 'ion_number']).loc[ion]
        lvl_index2id = lvl_index2id.reset_index()

        lower_level_id = []
        upper_level_id = []
        
        df = df.reset_index()
        for row in df.itertuples():

            llid = row.level_index_lower
            ulid = row.level_index_upper

            upper = lvl_index2id.at[ulid, 'level_id']
            lower = lvl_index2id.at[llid, 'level_id']

            lower_level_id.append(lower)
            upper_level_id.append(upper)

        df['lower_level_id'] = pd.Series(lower_level_id)
        df['upper_level_id'] = pd.Series(upper_level_id)

        return df

    @staticmethod
    def _create_artificial_fully_ionized(levels):
        """
        Returns a DataFrame with fully ionized levels.

        """
        fully_ionized_levels = []

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
        """
        Returns metastable flag column for the `levels` DataFrame.

        Parameters
        ----------
        levels : pandas.DataFrame
           Energy levels dataframe.

        lines : pandas.DataFrame
           Transition lines dataframe.

        levels_metastable_loggf_threshold : int
           loggf threshold value.

        """
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
        """
        Create Einstein coefficients columns for the `lines` DataFrame.

        Parameters
        ----------
        lines : pandas.DataFrame
           Transition lines dataframe.

        """
        einstein_coeff = (4 * np.pi ** 2 * const.e.gauss.value **
                          2) / (const.m_e.cgs.value * const.c.cgs.value)

        lines['B_lu'] = einstein_coeff * lines['f_lu'] / \
            (const.h.cgs.value * lines['nu'])

        lines['B_ul'] = einstein_coeff * lines['f_ul'] / \
            (const.h.cgs.value * lines['nu'])

        lines['A_ul'] = 2 * einstein_coeff * lines['nu'] ** 2 / \
            const.c.cgs.value ** 2 * lines['f_ul']

    @staticmethod
    def _calculate_collisional_strength(row, temperatures, 
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
        """ 
        The resulting DataFrame contains stacked energy levels from GFALL,
        Chianti (optional), CMFGEN (optional) and NIST ground levels. Only
        one source of levels is kept based on priorities.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData._get_all_levels_data()`.

        """

        logger.info('Ingesting energy levels.')
        gf_levels = self.gfall_reader.levels
        gf_levels['ds_id'] = 2

        if self.chianti_reader is not None:
            ch_levels = self.chianti_reader.levels
            ch_levels['ds_id'] = 4
        else:
            ch_levels = pd.DataFrame(columns=gf_levels.columns)

        if self.cmfgen_reader is not None:
            cf_levels = self.cmfgen_reader.levels
            cf_levels['ds_id'] = 5
        else:
            cf_levels = pd.DataFrame(columns=gf_levels.columns)

        levels = pd.concat([gf_levels, ch_levels, cf_levels], sort=True)
        levels['g'] = 2*levels['j'] + 1
        levels['g'] = levels['g'].astype(np.int)
        levels = levels.drop(columns=['j', 'label', 'method'])
        levels = levels.reset_index()
        levels = levels.rename(columns={'ion_charge': 'ion_number'})
        levels = levels[['atomic_number', 'ion_number', 'g', 'energy', 
                         'ds_id', 'priority']]
        levels['energy'] = u.Quantity(levels['energy'], 'cm-1').to(
            'eV', equivalencies=u.spectral()).value
 
        # Solve priorities and set attributes for later use.
        self.gfall_ions, self.chianti_ions, self.cmfgen_ions = self.solve_priorities(levels)

        to_string = lambda x: [f"{convert_atomic_number2symbol(ion[0])} {ion[1]}" \
                                    for ion in sorted(x)]

        gfall_str = ', '.join(to_string(self.gfall_ions))
        logger.info(f'GFALL selected species: {gfall_str}.')
        
        if len(self.chianti_ions) > 0:
            chianti_str = ', '.join(to_string(self.chianti_ions))
            logger.info(f'Chianti selected species: {chianti_str}.')
        
        if len(self.cmfgen_ions) > 0: 
            cmfgen_str = ', '.join(to_string(self.cmfgen_ions))
            logger.info(f'CMFGEN selected species: {cmfgen_str}.')

        # Concatenate ground levels from NIST
        ground_levels = self.ionization_energies.get_ground_levels()
        ground_levels = ground_levels.rename(columns={'ion_charge': 'ion_number'})
        ground_levels['ds_id'] = 1

        levels = pd.concat([ground_levels, levels], sort=True)
        levels['level_id'] = range(1, len(levels)+1)
        levels = levels.set_index('level_id')

        # The following code should only remove the duplicated
        # ground levels. Other duplicated levels should be re-
        # moved at the reader stage.

        mask = (levels['energy'] == 0.) & (levels[['atomic_number',
                    'ion_number', 'energy', 'g']].duplicated(keep='last'))
        levels = levels[~mask]

        # Filter levels by priority
        for ion in self.chianti_ions:
            mask = (levels['ds_id'] != 4) & (
                        levels['atomic_number'] == ion[0]) & (
                            levels['ion_number'] == ion[1])
            levels = levels.drop(levels[mask].index)

        for ion in self.cmfgen_ions:
            mask = (levels['ds_id'] != 5) & (
                        levels['atomic_number'] == ion[0]) & (
                            levels['ion_number'] == ion[1])
            levels = levels.drop(levels[mask].index)

        levels = levels[['atomic_number', 'ion_number', 'g', 'energy', 
                         'ds_id']]
        levels = levels.reset_index()

        return levels


    def _get_all_lines_data(self):
        """        
        The resulting DataFrame contains stacked transition lines for 
        GFALL, Chianti (optional) and CMFGEN (optional).

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData._get_all_lines_data()`.

        """

        logger.info('Ingesting transition lines.')
        gf_lines = self.gfall_reader.lines
        gf_lines['ds_id'] = 2

        if self.chianti_reader is not None:
            ch_lines = self.chianti_reader.lines
            ch_lines['ds_id'] = 4
        else:
            ch_lines = pd.DataFrame(columns=gf_lines.columns)

        if self.cmfgen_reader is not None:
            cf_lines = self.cmfgen_reader.lines
            cf_lines['ds_id'] = 5
        else:
            cf_lines = pd.DataFrame(columns=gf_lines.columns)

        lines = pd.concat([gf_lines, ch_lines, cf_lines], sort=True)
        lines = lines.reset_index()
        lines = lines.rename(columns={'ion_charge': 'ion_number'})
        lines['line_id'] = range(1, len(lines)+1)

        # Filter lines by priority
        for ion in self.chianti_ions:
            mask = (lines['ds_id'] != 4) & (
                        lines['atomic_number'] == ion[0]) & (
                            lines['ion_number'] == ion[1])
            lines = lines.drop(lines[mask].index)

        for ion in self.cmfgen_ions:
            mask = (lines['ds_id'] != 5) & (
                        lines['atomic_number'] == ion[0]) & (
                            lines['ion_number'] == ion[1])
            lines = lines.drop(lines[mask].index)

        lines = lines.set_index(['atomic_number', 'ion_number'])
        lines = lines.sort_index()  # To supress warnings
        ions = set(self.gfall_ions).union(set(self.chianti_ions))\
                    .union((set(self.cmfgen_ions)))

        logger.info('Matching levels and lines.')
        lns_list = [ self.get_lvl_index2id(lines.loc[ion], self.levels_all)
                        for ion in ions]
        lines = pd.concat(lns_list, sort=True)
        lines = lines.set_index('line_id').sort_index()

        lines['loggf'] = np.log10(lines['gf'])
        lines = lines.drop(columns=['energy_upper', 'j_upper', 'energy_lower',
                            'j_lower', 'level_index_lower',
                            'level_index_upper'])

        lines['wavelength'] = u.Quantity(lines['wavelength'], 'nm').to(
            'AA').value

        lines.loc[lines['wavelength'] <=
                  GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_VACUUM

        lines.loc[lines['wavelength'] >
                  GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_AIR

        # Chianti wavelengths are already given in vacuum
        gfall_mask = lines['ds_id'] == 2
        air_mask = lines['medium'] == MEDIUM_AIR
        lines.loc[air_mask & gfall_mask,
                  'wavelength'] = convert_wavelength_air2vacuum(
            lines.loc[air_mask, 'wavelength'])

        lines = lines[['lower_level_id', 'upper_level_id',
                       'wavelength', 'gf', 'loggf', 'ds_id']]

        return lines

    def create_levels_lines(self, lines_loggf_threshold=-3,
                            levels_metastable_loggf_threshold=-3):
        """
        Generates the definitive `lines` and `levels` DataFrames by adding
        new columns and making some calculations.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData.create_levels_lines` method.

        """

        ionization_energies = self.ionization_energies.base.reset_index()
        ionization_energies = ionization_energies.rename(columns={'ion_charge': 'ion_number'})

        # Culling autoionization levels
        levels_w_ionization_energies = pd.merge(self.levels_all,
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
        lines = self.lines_all.join(pd.DataFrame(index=levels.index),
                                    on="lower_level_id", how="inner").\
            join(pd.DataFrame(index=levels.index),
                 on="upper_level_id", how="inner")

        # Culling lines with low gf values
        lines = lines.loc[lines["loggf"] > lines_loggf_threshold]

        # Do not clean levels that don't exist in lines

        # Create the metastable flags for levels
        levels["metastable"] = \
            self._create_metastable_flags(levels, self.lines_all,
                                          levels_metastable_loggf_threshold)

        # Create levels numbers
        levels = levels.sort_values(
            ["atomic_number", "ion_number", "energy", "g"])

        levels["level_number"] = levels.groupby(['atomic_number',
                                                 'ion_number'])['energy']. \
            transform(lambda x: np.arange(len(x))).values

        levels["level_number"] = levels["level_number"].astype(np.int)

        levels = levels[['atomic_number', 'ion_number', 'g', 'energy',
                         'metastable', 'level_number', 'ds_id']]

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
            lines['wavelength'], 'AA').to('Hz', u.spectral())

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

        return levels, lines

    def create_collisions(self, temperatures=np.arange(2000, 50000, 2000)):
        """
        Generates the definitive `collisions` DataFrame by adding new columns
        and making some calculations.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Produces the same output than `AtomData.create_collisions` method.

        """

        logger.info('Ingesting collisional strengths.')
        ch_collisions = self.chianti_reader.collisions
        ch_collisions['ds_id'] = 4

        # Not really needed because we have only one source of collisions
        collisions = pd.concat([ch_collisions], sort=True)
        ions = self.chianti_ions

        collisions = collisions.reset_index()
        collisions = collisions.rename(columns={'ion_charge': 'ion_number'})
        collisions = collisions.set_index(['atomic_number', 'ion_number'])

        logger.info('Matching collisions and levels.')
        col_list = [ self.get_lvl_index2id(collisions.loc[ion], self.levels_all)
                        for ion in ions]
        collisions = pd.concat(col_list, sort=True)
        collisions = collisions.sort_values(by=['lower_level_id', 'upper_level_id'])

        # `e_col_id` number starts after the last line id
        start = self.lines_all.index[-1] + 1
        collisions['e_col_id'] = range(start, start + len(collisions))

        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")

        # Join atomic_number, ion_number, level_number_lower, level_number_upper
        collisions = collisions.set_index(['atomic_number', 'ion_number'])
        lower_levels = levels.rename(columns={"level_number": "level_number_lower", 
                                              "g": "g_l", "energy": "energy_lower"}). \
                              loc[:, ["atomic_number", "ion_number", "level_number_lower",
                                      "g_l", "energy_lower"]]

        upper_levels = levels.rename(columns={"level_number": "level_number_upper",
                                              "g": "g_u", "energy": "energy_upper"}). \
                              loc[:, ["level_number_upper", "g_u", "energy_upper"]]

        collisions = collisions.join(lower_levels, on="lower_level_id").join(
                                     upper_levels, on="upper_level_id")

        # Calculate delta_e
        kb_ev = const.k_B.cgs.to('eV / K').value
        collisions["delta_e"] = (collisions["energy_upper"] - collisions["energy_lower"])/kb_ev

        # Calculate g_ratio
        collisions["g_ratio"] = collisions["g_l"] / collisions["g_u"]

        # Derive columns for collisional strengths
        c_ul_temperature_cols = ['t{:06d}'.format(t) for t in temperatures]

        collisions = collisions.rename(columns={'temperatures': 'btemp',
                                                'collision_strengths': 'bscups'})

        collisions = collisions[['e_col_id', 'lower_level_id',
                                 'upper_level_id', 'ds_id',
                                 'btemp', 'bscups', 'ttype', 'cups',
                                 'gf', 'atomic_number', 'ion_number',
                                 'level_number_lower', 'g_l',
                                 'energy_lower', 'level_number_upper', 
                                 'g_u', 'energy_upper', 'delta_e', 
                                 'g_ratio']]

        collisional_ul_factors = collisions.apply(self._calculate_collisional_strength, 
                                                  axis=1, args=(temperatures, kb_ev, 
                                                                c_ul_temperature_cols))

        collisions = pd.concat([collisions, collisional_ul_factors], axis=1)
        collisions = collisions.set_index('e_col_id')

        return collisions

    def create_cross_sections(self):
        """
        Create a DataFrame containing photoionization cross-sections.

        Returns
        -------
        pandas.DataFrame

        """

        logger.info('Ingesting photoionization cross-sections.')
        cross_sections = self.cmfgen_reader.cross_sections.reset_index()

        logger.info('Matching levels and cross sections.')
        cross_sections = cross_sections.rename(columns={'ion_charge': 'ion_number'})
        cross_sections = cross_sections.set_index(['atomic_number', 'ion_number'])

        cross_sections['level_index_lower'] = cross_sections['level_index'].values
        cross_sections['level_index_upper'] = cross_sections['level_index'].values
        phixs_list = [ self.get_lvl_index2id(cross_sections.loc[ion], self.levels_all) 
                        for ion in self.cmfgen_ions ]

        cross_sections = pd.concat(phixs_list, sort=True)
        cross_sections = cross_sections.sort_values(by=['lower_level_id', 'upper_level_id'])
        cross_sections['level_id'] = cross_sections['lower_level_id']

        # `x_sect_id` number starts after the last `line_id`, just a convention
        start = self.lines_all.index[-1] + 1
        cross_sections['x_sect_id'] = range(start, start + len(cross_sections))

        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels['level_id'] != -1].set_index('level_id')
        level_number = levels.loc[:, ['level_number']]
        cross_sections = cross_sections.join(level_number, on='level_id')

        # Levels are already cleaned, just drop the NaN's after join
        cross_sections = cross_sections.dropna()

        cross_sections['energy'] = u.Quantity(cross_sections['energy'], 'Ry').to('Hz', equivalencies=u.spectral())
        cross_sections['sigma'] = u.Quantity(cross_sections['sigma'], 'Mbarn').to('cm2')
        cross_sections['level_number'] = cross_sections['level_number'].astype('int')
        cross_sections = cross_sections.rename(columns={'energy':'nu', 'sigma':'x_sect'})

        return cross_sections


    def create_macro_atom(self):
        """
        Create a DataFrame containing macro atom data.

        Returns
        -------
        pandas.DataFrame

        Notes
        -----
        Refer to the docs: https://tardis-sn.github.io/tardis/physics/setup/plasma/macroatom.html

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

    def create_macro_atom_references(self):
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
    def ionization_energies_prepared(self):
        """
        Prepare the DataFrame with ionization energies for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """
        ionization_energies_prepared = self.ionization_energies.base.copy()
        ionization_energies_prepared = ionization_energies_prepared.reset_index()
        ionization_energies_prepared['ion_charge'] += 1
        ionization_energies_prepared = ionization_energies_prepared.rename(columns={'ion_charge': 'ion_number'})
        ionization_energies_prepared = ionization_energies_prepared.set_index(['atomic_number', 'ion_number'])

        return ionization_energies_prepared.squeeze()


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

        levels_prepared = levels_prepared.set_index(
            ["atomic_number", "ion_number", "level_number"])

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

        lines_prepared = lines_prepared.set_index([
            "atomic_number", "ion_number",
            "level_number_lower", "level_number_upper"])

        return lines_prepared

    @property
    def collisions_prepared(self):
        """
        Prepare the DataFrame with electron collisions for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """

        collisions_index = ["atomic_number",
                            "ion_number",
                            "level_number_lower",
                            "level_number_upper"]

        if "chianti" in self.collisions_metadata.dataset:
            collisions_columns = collisions_index + ['g_ratio', 'delta_e'] + \
                                sorted([col for col in self.collisions.columns if re.match('^t\d+$', col)])

        elif "cmfgen" in self.collisions_metadata.dataset:
            collisions_columns = collisions_index + list(self.collisions.columns)

        else:
            raise ValueError("Unknown source of collisional data")

        collisions_prepared = self.collisions.reset_index().loc[:, collisions_columns].copy()
        collisions_prepared = collisions_prepared.set_index(collisions_index)

        return collisions_prepared

    @property
    def cross_sections_prepared(self):
        """
        Prepare the DataFrame with photoionization cross-sections for TARDIS.

        Returns
        -------
        pandas.DataFrame

        """
        cross_sections_prepared = self.cross_sections.set_index(['atomic_number', 'ion_number', 'level_number'])
        cross_sections_prepared = cross_sections_prepared[['nu', 'x_sect']]

        return cross_sections_prepared

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

        macro_atom_prepared = self.macro_atom.loc[:, [
            "atomic_number",
            "ion_number", "source_level_number", "target_level_number",
            "transition_type", "transition_probability",
            "transition_line_id"]].copy()

        macro_atom_prepared = macro_atom_prepared.rename(columns={
            "target_level_number": "destination_level_number"})

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
        macro_atom_references_prepared = self.macro_atom_references.loc[:, [
            "atomic_number", "ion_number", "source_level_number", "count_down",
            "count_up", "count_total"]].copy()

        macro_atom_references_prepared = macro_atom_references_prepared.set_index(
            ['atomic_number', 'ion_number', 'source_level_number'])

        return macro_atom_references_prepared

    def to_hdf(self, fname):
        """
        Dump `prepared` attributes into an HDF5 file.
        
        Parameters
        ----------
        fname : path
           Path to the HDF5 output file.

        """
        import hashlib
        import platform
        import uuid
        from datetime import datetime

        import pytz

        from carsus import FORMAT_VERSION

        with pd.HDFStore(fname, 'w') as f:
            f.put('/atom_data', self.atomic_weights.base)
            f.put('/ionization_data', self.ionization_energies_prepared)
            f.put('/zeta_data', self.zeta_data.base)
            f.put('/levels_data', self.levels_prepared)
            f.put('/lines_data', self.lines_prepared)
            f.put('/macro_atom_data', self.macro_atom_prepared)
            f.put('/macro_atom_references',
                  self.macro_atom_references_prepared)

            if hasattr(self, 'collisions'):
                f.put('/collisions_data', self.collisions_prepared)
                f.put('/collisions_metadata', self.collisions_metadata)

            if hasattr(self, 'cross_sections'):
                f.put('/photoionization_data', self.cross_sections_prepared)

            meta = []
            meta.append(('format', 'version', FORMAT_VERSION))

            total_checksum = hashlib.md5()
            for key in f.keys():
                # update the total checksum to sign the file
                total_checksum.update(serialize_pandas_object(f[key]).to_buffer())

                # save individual DataFrame/Series checksum
                checksum = hash_pandas_object(f[key])
                meta.append(('md5sum', key.lstrip('/'), checksum))

            # data sources versions
            meta.append(('datasets', 'nist_weights', 
                         self.atomic_weights.version))

            meta.append(('datasets', 'nist_spectra', 
                         self.ionization_energies.version))

            meta.append(('datasets', 'gfall',
                         self.gfall_reader.version))

            meta.append(('datasets', 'zeta',
                         self.zeta_data.version))

            if self.chianti_reader is not None:
                meta.append(('datasets', 'chianti', 
                             self.chianti_reader.version))

            if self.cmfgen_reader is not None:
                meta.append(('datasets', 'cmfgen',
                             self.cmfgen_reader.version))

            # relevant package versions
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
            logger.info(f"Format Version: {FORMAT_VERSION}")
            logger.info(f"MD5: {total_checksum.hexdigest()}")
            logger.info(f"UUID1: {uuid1}")

            f.root._v_attrs['MD5'] = total_checksum.hexdigest()
            f.root._v_attrs['UUID1'] = uuid1
            f.root._v_attrs['FORMAT_VERSION'] = FORMAT_VERSION

            tz = pytz.timezone('UTC')
            date = datetime.now(tz).isoformat()
            f.root._v_attrs['DATE'] = date

            self.meta = meta_df
            f.put('/metadata', meta_df)
