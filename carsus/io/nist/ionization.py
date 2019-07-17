"""
Input module for the NIST Ionization Energies database
http://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
"""

import requests
import numpy as np
import pandas as pd

from bs4 import BeautifulSoup
from io import StringIO
from astropy import units as u
from uncertainties import ufloat_fromstr
from pyparsing import ParseException
from carsus.model import Ion, IonizationEnergy, Level, LevelEnergy
from carsus.io.base import BaseParser, BaseIngester
from carsus.io.nist.ionization_grammar import level

IONIZATION_ENERGIES_URL = 'https://physics.nist.gov/cgi-bin/ASD/ie.pl'


def download_ionization_energies(
        spectra='h-uuh',
        e_out=0,
        e_unit=1,
        format_=1,
        at_num_out=True,
        sp_name_out=False,
        ion_charge_out=True,
        el_name_out=False,
        seq_out=False,
        shells_out=True,
        conf_out=False,
        level_out=True,
        ion_conf_out=False,
        unc_out=True,
        biblio=False):
    """
        Downloader function for the Ionization Energies Data from the NIST Atomic Spectra Database
        Parameters
        ----------
        spectra: str
            (default value = 'h-uuh')
        Returns
        -------
        str
            Preformatted text data
        """
    data = {'spectra': spectra, 'units': e_unit,
            'format': format_, 'at_num_out': at_num_out, 'sp_name_out': sp_name_out,
            'ion_charge_out': ion_charge_out, 'el_name_out': el_name_out,
            'seq_out': seq_out, 'shells_out': shells_out, 'conf_out': conf_out,
            'level_out': level_out, 'ion_conf_out': ion_conf_out, 'e_out': e_out,
            'unc_out': unc_out, 'biblio': biblio}

    data = {k: v for k, v in data.items() if v is not False}

    print("Downloading ionization energies from the NIST Atomic Spectra Database")
    r = requests.post(IONIZATION_ENERGIES_URL, data=data)
    return r.text


class NISTIonizationEnergiesParser(BaseParser):
    """
        Class for parsers for the Ionization Energies Data from the NIST Atomic Spectra Database
        Attributes
        ----------
        base : pandas.DataFrame
        grammar : pyparsing.ParseElement
            (default value = isotope)
        columns : list of str
            (default value = COLUMNS)
        Methods
        -------
        load(input_data)
            Parses the input data and stores the results in the `base` attribute
        prepare_ion_energies()
            Returns a new dataframe created from `base` that contains ionization energies data
    """

    def load(self, input_data):
        soup = BeautifulSoup(input_data, 'html5lib')
        pre_tag = soup.pre
        for a in pre_tag.find_all("a"):
            a = a.sting
        text_data = pre_tag.get_text()
        processed_text_data = ''
        for line in text_data.split('\n')[2:]:
            if line.startswith('----'):
                continue
            if line.startswith('If'):
                break
            if line.startswith('Notes'):
                break
            line.strip('|')
            processed_text_data += line + '\n'
        column_names = ['atomic_number', 'ion_charge', 'ground_shells', 'ground_level', 'ionization_energy_str']
        base = pd.read_csv(StringIO(processed_text_data), sep='|', header=None,
                         usecols=range(5), names=column_names)
        for column in ['ground_shells', 'ground_level', 'ionization_energy_str']:
                base[column] = base[column].map(lambda x: x.strip())
        self.base = base

    def prepare_ioniz_energies(self):
        """ Returns a new dataframe created from `base` that contains ionization energies data """
        ioniz_energies = self.base.copy()

        def parse_ioniz_energy_str(row):
            ioniz_energy_str = row['ionization_energy_str']
            if ioniz_energy_str == '':
                return None
            if ioniz_energy_str.startswith('('):
                method = 'theor' # theoretical
                ioniz_energy_str = ioniz_energy_str[1:-1]  # .strip('()') wasn't working for '(217.7185766(10))' 
                #.replace('))', ')') - not clear why that exists
            elif ioniz_energy_str.startswith('['):
                method = 'intrpl' # interpolated
                ioniz_energy_str = ioniz_energy_str.strip('[]')
            else:
                method = 'meas'  # measured
            # ToDo: Some value are given without uncertainty. How to be with them?
            ioniz_energy = ufloat_fromstr(ioniz_energy_str)
            return pd.Series([ioniz_energy.nominal_value, ioniz_energy.std_dev, method])

        ioniz_energies[['ionization_energy_value', 'ionization_energy_uncert',
                      'ionization_energy_method']] = ioniz_energies.apply(parse_ioniz_energy_str, axis=1)
        ioniz_energies.drop('ionization_energy_str', axis=1, inplace=True)
        ioniz_energies.set_index(['atomic_number', 'ion_charge'], inplace=True)

        # discard null values
        ioniz_energies = ioniz_energies[pd.notnull(ioniz_energies["ionization_energy_value"])]

        return ioniz_energies

    def prepare_ground_levels(self):
        """ Returns a new dataframe created from `base` that contains the ground levels data """

        ground_levels = self.base.loc[:, ["atomic_number", "ion_charge",
                                             "ground_shells", "ground_level"]].copy()

        def parse_ground_level(row):
            ground_level = row["ground_level"]
            lvl = pd.Series(index=["term", "spin_multiplicity", "L", "parity", "J"])

            try:
                lvl_tokens = level.parseString(ground_level)
            except ParseException:
                raise

            lvl["parity"] = lvl_tokens["parity"]

            try:
                lvl["J"] = lvl_tokens["J"]
            except KeyError:
                pass

            try:
                lvl["term"] = "".join([str(_) for _ in lvl_tokens["ls_term"]])
                lvl["spin_multiplicity"] = lvl_tokens["ls_term"]["mult"]
                lvl["L"] = lvl_tokens["ls_term"]["L"]
            except KeyError:
                # The term is not LS
                pass

            try:
                lvl["term"] = "".join([str(_) for _ in lvl_tokens["jj_term"]])
            except KeyError:
                # The term is not JJ
                pass

            return lvl

        ground_levels[["term", "spin_multiplicity",
                          "L", "parity", "J"]] = ground_levels.apply(parse_ground_level, axis=1)

        ground_levels.rename(columns={"ground_shells": "configuration"}, inplace=True)
        ground_levels.set_index(['atomic_number', 'ion_charge'], inplace=True)

        return ground_levels


class NISTIonizationEnergiesIngester(BaseIngester):
    """
        Class for ingesters for the Ionization Energies Data from the NIST Atomic Spectra Database
        Attributes
        ----------
        session: SQLAlchemy session

        data_source: DataSource instance
            The data source of the ingester

        parser : BaseParser instance
            (default value = NISTIonizationEnergiesParser())

        downloader : function
            (default value = download_ionization_energies)
        spectra: str
            (default value = 'h-uuh')

        Methods
        -------
        download()
            Downloads the data with the 'downloader' and loads the `parser` with it
        ingest(session)
            Persists the downloaded data into the database
        """

    def __init__(self, session, ds_short_name="nist-asd", downloader=None, parser=None, spectra="h-uuh"):
        if parser is None:
            parser = NISTIonizationEnergiesParser()
        if downloader is None:
            downloader = download_ionization_energies
        self.spectra = spectra
        super(NISTIonizationEnergiesIngester, self). \
            __init__(session, ds_short_name=ds_short_name, parser=parser, downloader=downloader)

    def download(self):
        data = self.downloader(spectra=self.spectra)
        self.parser(data)

    def ingest_ionization_energies(self, ioniz_energies=None):

        if ioniz_energies is None:
            ioniz_energies = self.parser.prepare_ioniz_energies()

        print("Ingesting ionization energies from {}".format(self.data_source.short_name))

        for index, row in ioniz_energies.iterrows():
            atomic_number, ion_charge = index
            # Query for an existing ion; create if doesn't exists
            ion = Ion.as_unique(self.session,
                                atomic_number=atomic_number, ion_charge=ion_charge)
            ion.energies = [
                IonizationEnergy(ion=ion,
                                 data_source=self.data_source,
                                 quantity=row['ionization_energy_value'] * u.eV,
                                 uncert=row['ionization_energy_uncert'],
                                 method=row['ionization_energy_method'])
            ]
            # No need to add ion to the session, because
            # that was done in `as_unique`

    def ingest_ground_levels(self, ground_levels=None):

        if ground_levels is None:
            ground_levels = self.parser.prepare_ground_levels()

        print("Ingesting ground levels from {}".format(self.data_source.short_name))

        for index, row in ground_levels.iterrows():
            atomic_number, ion_charge = index

            # Replace nan with None
            row = row.where(pd.notnull(row), None)

            ion = Ion.as_unique(self.session, atomic_number=atomic_number, ion_charge=ion_charge)

            try:
                spin_multiplicity = int(row["spin_multiplicity"])
            except TypeError:  # Raised when the variable is None
                spin_multiplicity = None

            try:
                parity = int(row["parity"])
            except TypeError:  # Raised when the variable is None
                parity = None

            ion.levels.append(
                Level(data_source=self.data_source,
                      configuration=row["configuration"],
                      term=row["term"],
                      L=row["L"],
                      spin_multiplicity=spin_multiplicity,
                      parity=parity,
                      J=row["J"],
                      energies=[
                          LevelEnergy(quantity=0, data_source=self.data_source)
                      ])
            )

    def ingest(self, ionization_energies=True, ground_levels=True):

        # Download data if needed
        if self.parser.base is None:
            self.download()

        if ionization_energies:
            self.ingest_ionization_energies()
            self.session.flush()

        if ground_levels:
            self.ingest_ground_levels()
            self.session.flush()


class NISTIonizationEnergies:
    """ Docstring """
    def __init__(self, spectra):
        input_data = download_ionization_energies(spectra)
        self.parser = NISTIonizationEnergiesParser(input_data)
        self._prepare_data()

    def _prepare_data(self):
        self.ionization_data = pd.DataFrame()
        self.ionization_data['atomic_number'] = self.parser.base['atomic_number']
        self.ionization_data['ion_number'] = self.parser.base['ion_charge'] + 1
        self.ionization_data['ionization_energy'] = self.parser.base['ionization_energy_str'].str.strip('[]()').astype(np.float64)
        self.ionization_data.set_index(['atomic_number', 'ion_number'], inplace=True)

        # Convert DataFrame to Series
        self.ionization_data = self.ionization_data['ionization_energy']

    def to_hdf(self, fname):
        with pd.HDFStore(fname, 'a') as f:
            f.append('/ionization_data', self.ionization_data)
