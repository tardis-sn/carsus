import re, logging

import numpy as np
import pandas as pd
import astropy.units as u

from astropy import units as u
from astropy import constants as const
from sqlalchemy import and_
from pyparsing import ParseException
from carsus.model import DataSource, Ion, Level, LevelEnergy,\
    Line, LineWavelength, LineGFValue, MEDIUM_VACUUM, MEDIUM_AIR
from carsus.io.base import IngesterError, BaseParser
from carsus.util import (convert_atomic_number2symbol, 
                         parse_selected_species,
                         convert_wavelength_air2vacuum)
from carsus.io.nist.ionization import NISTIonizationEnergies

GFALL_AIR_THRESHOLD = 200  # [nm], wavelengths above this value are given in air

P_EMISSION_DOWN = -1
P_INTERNAL_DOWN = 0
P_INTERNAL_UP = 1

logger = logging.getLogger(__name__)

class GFALLReader(object):
    """
        Class for extracting lines and levels data from kurucz gfall files

        Attributes
        ----------
        fname: path to gfall.dat

        Methods
        --------
        gfall_raw:
            Return pandas DataFrame representation of gfall

    """

    gfall_fortran_format = ('F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,'
                             'A10,F6.2,F6.2,F6.2,A4,I2,I2,I3,F6.3,I3,F6.3,I5,I5,'
                             '1X,I1,A1,1X,I1,A1,I1,A3,I5,I5,I6')

    gfall_columns = ['wavelength', 'loggf', 'element_code', 'e_first', 'j_first',
               'blank1', 'label_first', 'e_second', 'j_second', 'blank2',
               'label_second', 'log_gamma_rad', 'log_gamma_stark',
               'log_gamma_vderwaals', 'ref', 'nlte_level_no_first',
               'nlte_level_no_second', 'isotope', 'log_f_hyperfine',
               'isotope2', 'log_iso_abundance', 'hyper_shift_first',
               'hyper_shift_second', 'blank3', 'hyperfine_f_first',
               'hyperfine_note_first', 'blank4', 'hyperfine_f_second',
               'hyperfine_note_second', 'line_strength_class', 'line_code',
               'lande_g_first', 'lande_g_second', 'isotopic_shift']

    default_unique_level_identifier = ['energy', 'j']
    def __init__(self, fname, unique_level_identifier=None):
        """

        Parameters
        ----------
        fname: str
            path to the gfall file

        unique_level_identifier: list
            list of attributes to identify unique levels from. Will always use
            atomic_number and ion charge in addition.
        """
        self.fname = fname
        self._gfall_raw = None
        self._gfall = None
        self._levels = None
        self._lines = None
        if unique_level_identifier is None:
            logger.warn('A specific combination to identify unique levels from '
                        'the gfall data has not been given. Defaulting to '
                        '["energy", "j"].')
            self.unique_level_identifier = self.default_unique_level_identifier


    @property
    def gfall_raw(self):
        if self._gfall_raw is None:
            self._gfall_raw = self.read_gfall_raw()
        return self._gfall_raw

    @property
    def gfall(self):
        if self._gfall is None:
            self._gfall = self.parse_gfall()
        return self._gfall

    @property
    def levels(self):
        if self._levels is None:
            self._levels = self.extract_levels()
        return self._levels

    @property
    def lines(self):
        if self._lines is None:
            self._lines = self.extract_lines()
        return self._lines

    def read_gfall_raw(self, fname=None):
        """
        Reading in a normal gfall.dat

        Parameters
        ----------
        fname: ~str
            path to gfall.dat

        Returns
        -------
            pandas.DataFrame
                pandas Dataframe represenation of gfall
        """

        if fname is None:
            fname = self.fname

        logger.info('Parsing GFALL {0}'.format(fname))

        # FORMAT(F11.4,F7.3,F6.2,F12.3,F5.2,1X,A10,F12.3,F5.2,1X,A10,
        # 3F6.2,A4,2I2,I3,F6.3,I3,F6.3,2I5,1X,A1,A1,1X,A1,A1,i1,A3,2I5,I6)


        number_match = re.compile(r'\d+(\.\d+)?')
        type_match = re.compile(r'[FIXA]')
        type_dict = {'F': np.float64, 'I': np.int64, 'X': str, 'A': str}
        field_types = tuple([type_dict[item] for item in number_match.sub(
            '', self.gfall_fortran_format).split(',')])

        field_widths = type_match.sub('', self.gfall_fortran_format)
        field_widths = map(int, re.sub(r'\.\d+', '', field_widths).split(','))

        field_type_dict = {col:dtype for col, dtype in zip(self.gfall_columns, field_types)}
        gfall = pd.read_fwf(fname, widths=field_widths, skip_blank_lines=True,
                            names=self.gfall_columns, dtypes=field_type_dict)
        #remove empty lines
        gfall = gfall[~gfall.isnull().all(axis=1)].reset_index(drop=True)

        return gfall

    def parse_gfall(self, gfall_raw=None):
        """
        Parse raw gfall DataFrame

        Parameters
        ----------
        gfall_raw: pandas.DataFrame

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """


        gfall = gfall_raw if gfall_raw is not None else self.gfall_raw.copy()
        gfall = gfall.rename(columns={'e_first':'energy_first',
                                      'e_second':'energy_second'})
        double_columns = [item.replace('_first', '') for item in gfall.columns if
                          item.endswith('first')]

        # due to the fact that energy is stored in 1/cm
        order_lower_upper = (gfall["energy_first"].abs() <
                             gfall["energy_second"].abs())

        for column in double_columns:
            data = pd.concat([gfall['{0}_first'.format(column)][order_lower_upper],
                              gfall['{0}_second'.format(column)][~order_lower_upper]])

            gfall['{0}_lower'.format(column)] = data

            data = pd.concat([gfall['{0}_first'.format(column)][~order_lower_upper], \
                              gfall['{0}_second'.format(column)][order_lower_upper]])

            gfall['{0}_upper'.format(column)] = data

            del gfall['{0}_first'.format(column)]
            del gfall['{0}_second'.format(column)]

        # Clean labels
        gfall["label_lower"] = gfall["label_lower"].str.strip()
        gfall["label_upper"] = gfall["label_upper"].str.strip()

        gfall["label_lower"] = gfall["label_lower"].str.replace('\s+', ' ')
        gfall["label_upper"] = gfall["label_upper"].str.replace('\s+', ' ')

        # Ignore lines with the labels "AVARAGE ENERGIES" and "CONTINUUM"
        ignored_labels = ["AVERAGE", "ENERGIES", "CONTINUUM"]
        gfall = gfall.loc[~((gfall["label_lower"].isin(ignored_labels)) |
                            (gfall["label_upper"].isin(ignored_labels)))].copy()

        gfall['energy_lower_predicted'] = gfall["energy_lower"] < 0
        gfall["energy_lower"] = gfall["energy_lower"].abs()
        gfall['energy_upper_predicted'] = gfall["energy_upper"] < 0
        gfall["energy_upper"] = gfall["energy_upper"].abs()

        gfall['atomic_number'] = gfall.element_code.astype(int)
        gfall['ion_charge'] = ((gfall.element_code.values -
                                gfall.atomic_number.values) * 100).round().astype(int)

        del gfall['element_code']

        return gfall

    def extract_levels(self, gfall=None, selected_columns=None):
        """
        Extract levels from `gfall`. We first generate a concatenated DataFrame
        of all lower and upper levels. Then we drop the duplicate leves

        Parameters
        ----------
        gfall: pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """

        if gfall is None:
            gfall = self.gfall

        if selected_columns is None:
            selected_columns = ['atomic_number', 'ion_charge', 'energy', 'j',
                                'label', 'theoretical']

        column_renames = {'energy_{0}': 'energy', 'j_{0}': 'j', 'label_{0}': 'label',
                          'energy_{0}_predicted': 'theoretical'}

        e_lower_levels = gfall.rename(
            columns=dict([(key.format('lower'), value)
                          for key, value in column_renames.items()]))

        e_upper_levels = gfall.rename(
            columns=dict([(key.format('upper'), value)
                          for key, value in column_renames.items()]))

        levels = pd.concat([e_lower_levels[selected_columns],
                            e_upper_levels[selected_columns]])
        unique_level_id = ['atomic_number', 'ion_charge'] + self.unique_level_identifier

        levels.drop_duplicates(unique_level_id, inplace=True)
        levels = levels.sort_values(['atomic_number', 'ion_charge', 'energy',
                                     'j', 'label'])

        levels["method"] = levels["theoretical"].\
            apply(lambda x: "theor" if x else "meas")  # Theoretical or measured
        levels.drop("theoretical", 1, inplace=True)

        levels["level_index"] = levels.groupby(['atomic_number', 'ion_charge'])['j'].\
            transform(lambda x: np.arange(len(x), dtype=np.int64)).values
        levels["level_index"] = levels["level_index"].astype(int)

        # ToDo: The commented block below does not work with all lines. Find a way to parse it.
        # levels[["configuration", "term"]] = levels["label"].str.split(expand=True)
        # levels["configuration"] = levels["configuration"].str.strip()
        # levels["term"] = levels["term"].str.strip()

        levels.set_index(["atomic_number", "ion_charge", "level_index"], inplace=True)
        return levels

    def extract_lines(self, gfall=None, levels=None, selected_columns=None):
        """
        Extract lines from `gfall`

        Parameters
        ----------
        gfall: pandas.DataFrame
        selected_columns: list
            list of which columns to select (optional - default=None which selects
            a default set of columns)

        Returns
        -------
            pandas.DataFrame
                a level DataFrame
        """
        if gfall is None:
            gfall = self.gfall

        if levels is None:
            levels = self.levels

        if selected_columns is None:
            selected_columns = ['atomic_number', 'ion_charge']
            selected_columns += [item + '_lower' for item in self.unique_level_identifier]
            selected_columns += [item + '_upper' for item in self.unique_level_identifier]
            selected_columns += ['wavelength', 'loggf']


        logger.info('Extracting line data: {0}'.format(', '.join(selected_columns)))
        unique_level_id = ['atomic_number', 'ion_charge'] + self.unique_level_identifier
        levels_idx = levels.reset_index()
        levels_idx = levels_idx.set_index(unique_level_id)

        lines = gfall[selected_columns].copy()
        lines["gf"] = np.power(10, lines["loggf"])
        lines = lines.drop(["loggf"], 1)

        # Assigning levels to lines

        levels_unique_idxed = self.levels.reset_index().set_index(['atomic_number', 'ion_charge'] + self.unique_level_identifier)

        lines_lower_unique_idx = (['atomic_number', 'ion_charge'] +
                                  [item + '_lower' for item in self.unique_level_identifier])
        lines_upper_unique_idx = (['atomic_number', 'ion_charge'] +
                                  [item + '_upper' for item in self.unique_level_identifier])
        lines_lower_idx = lines.set_index(lines_lower_unique_idx)
        lines_lower_idx['level_index_lower'] = levels_unique_idxed['level_index']
        lines_upper_idx = lines_lower_idx.reset_index().set_index(lines_upper_unique_idx)
        lines_upper_idx['level_index_upper'] = levels_unique_idxed['level_index']
        lines = lines_upper_idx.reset_index().set_index(
            ['atomic_number', 'ion_charge', 'level_index_lower', 'level_index_upper'])


        return lines


class GFALLIngester(object):
    """
        Class for ingesting data from kurucz dfall files

        Attributes
        ----------
        session: SQLAlchemy session
        fname: str
            The name of the gfall file to read
        ions: str
            Ingest levels and lines only for these ions. If set to None then ingest all.
            (default: None)
        data_source: DataSource instance
            The data source of the ingester

        gfall_reader : GFALLReaderinstance

        Methods
        -------
        ingest(session)
            Persists data into the database
    """
    def __init__(self, session, fname, ions=None, ds_short_name="ku_latest"):
        self.session = session
        self.gfall_reader = GFALLReader(fname)
        if ions is not None:
            try:
                ions = parse_selected_species(ions)
            except ParseException:
                raise ValueError('Input is not a valid species string {}'.format(ions))
            ions = pd.DataFrame.from_records(ions, columns=["atomic_number", "ion_charge"])
            self.ions = ions.set_index(['atomic_number', 'ion_charge'])
        else:
            self.ions = None

        self.data_source = DataSource.as_unique(self.session, short_name=ds_short_name)
        if self.data_source.data_source_id is None:  # To get the id if a new data source was created
            self.session.flush()

    def get_lvl_index2id(self, ion):
        """ Return a DataFrame that maps levels indexes to ids """

        q_ion_lvls = self.session.query(Level.level_id.label("id"),
                                        Level.level_index.label("index")). \
            filter(and_(Level.ion == ion,
                        Level.data_source == self.data_source))

        lvl_index2id = list()
        for id, index in q_ion_lvls:
            lvl_index2id.append((index, id))

        lvl_index2id_dtype = [("index", np.int), ("id", np.int)]
        lvl_index2id = np.array(lvl_index2id, dtype=lvl_index2id_dtype)
        lvl_index2id = pd.DataFrame.from_records(lvl_index2id, index="index")

        return lvl_index2id

    def ingest_levels(self, levels=None):

        if levels is None:
            levels = self.gfall_reader.levels

        # Select ions
        if self.ions is not None:
            levels = levels.reset_index().\
                                  join(self.ions, how="inner",
                                       on=["atomic_number", "ion_charge"]).\
                                  set_index(["atomic_number", "ion_charge", "level_index"])

        print("Ingesting levels from {}".format(self.data_source.short_name))

        for ion_index, ion_levels in levels.groupby(level=["atomic_number", "ion_charge"]):

            atomic_number, ion_charge = ion_index
            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting levels for {} {}".format(convert_atomic_number2symbol(atomic_number), ion_charge))

            for index, row in ion_levels.iterrows():

                level_index = index[2]  # index: (atomic_number, ion_charge, level_index)

                ion.levels.append(
                    Level(level_index=level_index,
                          data_source=self.data_source,
                          J=row["j"],
                          energies=[
                              LevelEnergy(quantity=row["energy"]*u.Unit("cm-1"),
                                          method=row["method"],
                                          data_source=self.data_source)
                          ])
                )

    def ingest_lines(self, lines=None):

        if lines is None:
            lines = self.gfall_reader.lines

        # Select ions
        if self.ions is not None:
            lines = lines.reset_index(). \
                join(self.ions, how="inner",
                     on=["atomic_number", "ion_charge"]). \
                set_index(["atomic_number", "ion_charge", "level_index_lower", "level_index_upper"])

        print("Ingesting lines from {}".format(self.data_source.short_name))

        for ion_index, ion_lines in lines.groupby(level=["atomic_number", "ion_charge"]):

            atomic_number, ion_charge = ion_index
            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            print("Ingesting lines for {} {}".format(convert_atomic_number2symbol(atomic_number), ion_charge))

            lvl_index2id = self.get_lvl_index2id(ion)

            for index, row in ion_lines.iterrows():

                # index: (atomic_number, ion_charge, lower_level_index, upper_level_index)
                lower_level_index, upper_level_index = index[2:]

                try:
                    lower_level_id = int(lvl_index2id.loc[lower_level_index])
                    upper_level_id = int(lvl_index2id.loc[upper_level_index])
                except KeyError:
                    raise IngesterError("Levels from this source have not been found."
                                        "You must ingest levels before transitions")

                medium = MEDIUM_VACUUM if row["wavelength"] <= GFALL_AIR_THRESHOLD else MEDIUM_AIR

                # Create a new line
                line = Line(
                    lower_level_id=lower_level_id,
                    upper_level_id=upper_level_id,
                    data_source=self.data_source,
                    wavelengths=[
                        LineWavelength(quantity=row["wavelength"] * u.nm,
                                       medium=medium,
                                       data_source=self.data_source)
                    ],
                    gf_values=[
                        LineGFValue(quantity=row["gf"],
                                    data_source=self.data_source)
                    ]
                )

                self.session.add(line)

    def ingest(self, levels=True, lines=True):

        if levels:
            self.ingest_levels()
            self.session.flush()

        if lines:
            self.ingest_lines()
            self.session.flush()


class GFALL(BaseParser):
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
    def __init__(self, fname, ions, lines_loggf_threshold=-3, \
        levels_metastable_loggf_threshold=-3):

        self.ions = parse_selected_species(ions)
        self.gfall_reader = GFALLReader(fname)
        self._create_ionization_data()
        self.levels_all = self._get_all_levels_data().reset_index()
        self.lines_all = self._get_all_lines_data(self.levels_all)
        self._create_levels_lines()
        self._create_macro_atom()
        self._create_macro_atom_references()

    def _create_ionization_data(self):
        atoms = set([convert_atomic_number2symbol(i[0]) for i in self.ions])
        atoms = ', '.join(atoms)
        parser = NISTIonizationEnergies(atoms)
        ground_levels = parser.get_ground_levels()
        ground_levels = ground_levels.rename(columns={'ion_charge': 'ion_number'})

        self.ionization_energies = parser.base
        self.ground_levels = ground_levels

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

        lines = pd.concat(df_list)
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

        levels = pd.concat([self.ground_levels, levels])
        levels['level_id'] = range(1, len(levels)+1)
        levels = levels.set_index('level_id')
        levels = levels.drop_duplicates(keep='last')

        return levels

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

    def _create_macro_atom(self):
        """
            Create a DataFrame containing *macro atom* data.

            Returns
            -------
            macro_atom: pandas.DataFrame
                DataFrame with:
                    index: none;
                    columns: atomic_number, ion_number, source_level_number, target_level_number,
                        transition_line_id, transition_type, transition_probability.

            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html

        """
        # Exclude artificially created levels from levels
        levels = self.levels.loc[self.levels["level_id"] != -1].set_index("level_id")

        lvl_energy_lower = levels.rename(columns={"energy": "energy_lower"}).loc[:, ["energy_lower"]]
        lvl_energy_upper = levels.rename(columns={"energy": "energy_upper"}).loc[:, ["energy_upper"]]

        lines = self.lines.set_index("line_id")
        lines = lines.join(lvl_energy_lower, on="lower_level_id").join(lvl_energy_upper, on="upper_level_id")

        macro_atom = list()
        macro_atom_dtype = [("atomic_number", np.int), ("ion_number", np.int),
                            ("source_level_number", np.int), ("target_level_number", np.int),
                            ("transition_line_id", np.int), ("transition_type", np.int), ("transition_probability", np.float)]

        for line_id, row in lines.iterrows():
            atomic_number, ion_number = row["atomic_number"], row["ion_number"]
            level_number_lower, level_number_upper = row["level_number_lower"], row["level_number_upper"]
            nu = row["nu"]
            f_ul, f_lu = row["f_ul"], row["f_lu"]
            e_lower, e_upper = row["energy_lower"], row["energy_upper"]

            transition_probabilities_dict = dict()  # type : probability
            transition_probabilities_dict[P_EMISSION_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * (e_upper - e_lower)
            transition_probabilities_dict[P_INTERNAL_DOWN] = 2 * nu**2 * f_ul / const.c.cgs.value**2 * e_lower
            transition_probabilities_dict[P_INTERNAL_UP] = f_lu * e_lower / (const.h.cgs.value * nu)

            macro_atom.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_EMISSION_DOWN, transition_probabilities_dict[P_EMISSION_DOWN]))
            macro_atom.append((atomic_number, ion_number, level_number_upper, level_number_lower,
                                    line_id, P_INTERNAL_DOWN, transition_probabilities_dict[P_INTERNAL_DOWN]))
            macro_atom.append((atomic_number, ion_number, level_number_lower, level_number_upper,
                                    line_id, P_INTERNAL_UP, transition_probabilities_dict[P_INTERNAL_UP]))

        macro_atom = np.array(macro_atom, dtype=macro_atom_dtype)
        macro_atom = pd.DataFrame(macro_atom)

        macro_atom = macro_atom.sort_values(["atomic_number", "ion_number", "source_level_number"])

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
                    columns: atomic_number, ion_number, source_level_number, destination_level_number,
                        transition_line_id, transition_type, transition_probability.
            Notes:
                Refer to the docs: http://tardis.readthedocs.io/en/latest/physics/plasma/macroatom.html
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
                and columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total
        """
        macro_atom_references = self.levels.rename(columns={"level_number": "source_level_number"}).\
                                       loc[:, ["atomic_number", "ion_number", "source_level_number", "level_id"]]

        count_down = self.lines.groupby("upper_level_id").size()
        count_down.name = "count_down"

        count_up = self.lines.groupby("lower_level_id").size()
        count_up.name = "count_up"

        macro_atom_references = macro_atom_references.join(count_down, on="level_id").join(count_up, on="level_id")
        macro_atom_references = macro_atom_references.drop("level_id", axis=1)

        macro_atom_references = macro_atom_references.fillna(0)
        macro_atom_references["count_total"] = 2*macro_atom_references["count_down"] + macro_atom_references["count_up"]

        # Convert to int
        macro_atom_references["count_down"] = macro_atom_references["count_down"].astype(np.int)
        macro_atom_references["count_up"] = macro_atom_references["count_up"].astype(np.int)
        macro_atom_references["count_total"] = macro_atom_references["count_total"].astype(np.int)

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
                    columns: atomic_number, ion_number, source_level_number, count_down, count_up, count_total.
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
            f.append('/levels', self.levels_prepared)
            f.append('/lines', self.lines_prepared)
