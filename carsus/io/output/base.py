import numpy as np
import pandas as pd
from carsus.util import parse_selected_species, convert_wavelength_air2vacuum
from carsus.model import MEDIUM_VACUUM, MEDIUM_AIR
from carsus.io.kurucz import GFALLReader
from astropy import units as u
from astropy import constants as const

GFALL_AIR_THRESHOLD = 200  # [nm], wavelengths above this value are given in air


class TARDISAtomData:

    """
    Attributes
    ----------
    levels_prepared : pandas.DataFrame
    lines_prepared : pandas.DataFrame

    Methods
    -------
    to_hdf(fname)
        Dump `levels_prepared` and `lines_prepared` attributes into an HDF5 file
    """

    def __init__(self, gfall_reader, ionization_energies, ions, lines_loggf_threshold=-3, \
        levels_metastable_loggf_threshold=-3):

        self.levels_lines_param = {
            "levels_metastable_loggf_threshold": levels_metastable_loggf_threshold,
            "lines_loggf_threshold": lines_loggf_threshold
        }

        self.ions = parse_selected_species(ions)
        self.gfall_reader = gfall_reader

        self.ionization_energies = ionization_energies.base
        self.ground_levels = ionization_energies.get_ground_levels()
        self.ground_levels.rename(columns={'ion_charge': 'ion_number'}, inplace=True)

        self.levels_all = self._get_all_levels_data().reset_index()
        self.lines_all = self._get_all_lines_data(self.levels_all)

        self._create_levels_lines(**self.levels_lines_param)

    @staticmethod
    def _create_artificial_fully_ionized(levels):
        """ Create artificial levels for fully ionized ions """
        fully_ionized_levels = list()

        for atomic_number, _ in levels.groupby("atomic_number"):
            fully_ionized_levels.append(
                (-1, atomic_number, atomic_number, 0, 0.0, 1, True)
            )

        levels_columns = ["level_id", "atomic_number", "ion_number", "level_number", "energy", "g", "metastable"]
        fully_ionized_levels_dtypes = [(key, levels.dtypes[key]) for key in levels_columns]

        fully_ionized_levels = np.array(fully_ionized_levels, dtype=fully_ionized_levels_dtypes)

        return pd.DataFrame(data=fully_ionized_levels)

    @staticmethod
    def _create_metastable_flags(levels, lines, levels_metastable_loggf_threshold=-3):
        # Filter lines on the loggf threshold value
        metastable_lines = lines.loc[lines["loggf"] > levels_metastable_loggf_threshold]

        # Count the remaining strong transitions
        metastable_lines_grouped = metastable_lines.groupby("upper_level_id")
        metastable_counts = metastable_lines_grouped["upper_level_id"].count()
        metastable_counts.name = "metastable_counts"

        # If there are no strong transitions for a level (the count is NaN) then the metastable flag is True
        # else (the count is a natural number) the metastable flag is False
        levels = levels.join(metastable_counts)
        metastable_flags = levels["metastable_counts"].isnull()
        metastable_flags.name = "metastable"

        return metastable_flags

    def _get_all_levels_data(self):
        """ Returns the same output than `AtomData._get_all_levels_data()` """
        gf = self.gfall_reader
        gf.levels['g'] = 2*gf.levels['j'] + 1
        gf.levels['g'] = gf.levels['g'].map(np.int)

        ions_df = pd.DataFrame.from_records(self.ions, columns=["atomic_number", "ion_charge"])
        ions_df = ions_df.set_index(['atomic_number', 'ion_charge'])
        levels = gf.levels.reset_index().join(ions_df, how="inner", on=["atomic_number", "ion_charge"]).\
                                          set_index(["atomic_number", "ion_charge"])
        levels = levels.drop(columns=['j', 'label', 'method'])
        levels['level_id'] = range(1, len(levels)+1)
        levels = levels.reset_index().reset_index(drop=True)
        levels = levels.rename(columns={'ion_charge': 'ion_number'})
        levels = levels[['atomic_number', 'ion_number', 'g', 'energy']]

        levels['energy'] = levels['energy'].apply(lambda x: x*u.Unit('cm-1'))
        levels['energy'] = levels['energy'].apply(lambda x: x.to(u.eV, equivalencies=u.spectral()))
        levels['energy'] = levels['energy'].apply(lambda x: x.value)

        levels = pd.concat([self.ground_levels, levels], sort=True)
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

            except (KeyError, TypeError) as e:
                continue

            df = df.reset_index()
            lvl_index2id = levels.set_index(['atomic_number', 'ion_number']).loc[ion]
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
        lines.drop(columns=['energy_upper', 'j_upper', 'energy_lower', 'j_lower', \
            'level_index_lower', 'level_index_upper'], inplace=True)

        lines.loc[lines['wavelength'] <= GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_VACUUM
        lines.loc[lines['wavelength'] > GFALL_AIR_THRESHOLD, 'medium'] = MEDIUM_AIR
        lines['wavelength'] = lines['wavelength'].apply(lambda x: x*u.nm)
        lines['wavelength'] = lines['wavelength'].apply(lambda x: x.to('angstrom'))
        lines['wavelength'] = lines['wavelength'].apply(lambda x: x.value)

        air_mask = lines['medium'] == MEDIUM_AIR
        lines.loc[air_mask, 'wavelength'] = convert_wavelength_air2vacuum(
                lines.loc[air_mask, 'wavelength'])
        lines.drop(columns=['medium'], inplace=True)
        lines = lines[['lower_level_id', 'upper_level_id', 'wavelength', 'gf', 'loggf']]

        return lines

    def _create_levels_lines(self, lines_loggf_threshold=-3, levels_metastable_loggf_threshold=-3):
        """ Returns almost the same output than `AtomData.create_levels_lines` method """
        levels_all = self.levels_all
        lines_all = self.lines_all
        ionization_energies = self.ionization_energies.reset_index()
        ionization_energies['ion_number'] -= 1
        
        # Culling autoionization levels
        levels_w_ionization_energies = pd.merge(levels_all, ionization_energies, how='left', \
            on=["atomic_number", "ion_number"])
        mask = levels_w_ionization_energies["energy"] < levels_w_ionization_energies["ionization_energy"]
        levels = levels_w_ionization_energies[mask].copy()
        levels = levels.set_index('level_id').sort_values(by=['atomic_number', 'ion_number'])
        levels = levels.drop(columns='ionization_energy')

        # Clean lines
        lines = lines_all.join(pd.DataFrame(index=levels.index), on="lower_level_id", how="inner").\
            join(pd.DataFrame(index=levels.index), on="upper_level_id", how="inner")

        # Culling lines with low gf values
        lines = lines.loc[lines["loggf"] > lines_loggf_threshold]

        # Do not clean levels that don't exist in lines

        # Create the metastable flags for levels
        levels["metastable"] = self._create_metastable_flags(levels, lines_all, \
            levels_metastable_loggf_threshold)

        # Create levels numbers
        levels = levels.sort_values(["atomic_number", "ion_number", "energy", "g"])
        levels["level_number"] = levels.groupby(['atomic_number', 'ion_number'])['energy']. \
            transform(lambda x: np.arange(len(x))).values
        levels["level_number"] = levels["level_number"].astype(np.int)

        levels = levels[['atomic_number', 'energy', 'g', 'ion_number', \
            'level_number', 'metastable']]

        # Join atomic_number, ion_number, level_number_lower, level_number_upper on lines
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
        lines = lines.join(lower_levels, on="lower_level_id").join(upper_levels, on="upper_level_id")

        # Calculate absorption oscillator strength f_lu and emission oscillator strength f_ul
        lines["f_lu"] = lines["gf"] / lines["g_l"]
        lines["f_ul"] = lines["gf"] / lines["g_u"]

        # Calculate frequency
        lines['nu'] = u.Quantity(lines['wavelength'], 'angstrom').to('Hz', u.spectral())

        # Calculate Einstein coefficients
        einstein_coeff = (4 * np.pi ** 2 * const.e.gauss.value ** 2) / (const.m_e.cgs.value * const.c.cgs.value)
        lines['B_lu'] = einstein_coeff * lines['f_lu'] / (const.h.cgs.value * lines['nu'])
        lines['B_ul'] = einstein_coeff * lines['f_ul'] / (const.h.cgs.value * lines['nu'])
        lines['A_ul'] = 2 * einstein_coeff * lines['nu'] ** 2 / const.c.cgs.value ** 2 * lines['f_ul']

        # Reset indexes because `level_id` cannot be an index once we
        # add artificial levels for fully ionized ions that don't have ids (-1)
        lines = lines.reset_index()
        levels = levels.reset_index()

        # Create and append artificial levels for fully ionized ions
        artificial_fully_ionized_levels = self._create_artificial_fully_ionized(levels)
        levels = levels.append(artificial_fully_ionized_levels, ignore_index=True)
        levels = levels.sort_values(["atomic_number", "ion_number", "level_number"])

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
                columns: atomic_number, ion_number, level_number, energy[eV], g[1], metastable.
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
                    columns: line_id, atomic_number, ion_number, level_number_lower, level_number_upper,
                             wavelength[angstrom], nu[Hz], f_lu[1], f_ul[1], B_ul[?], B_ul[?], A_ul[1/s].
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
