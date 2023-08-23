"""
Input module for the NIST Ionization Energies database
http://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
"""

import logging
import os
from io import StringIO

import numpy as np
import pandas as pd
import requests
from astropy import units as u
from bs4 import BeautifulSoup
from pyparsing import ParseException

import carsus
from carsus.io.base import BaseIngester, BaseParser
from carsus.io.nist.ionization_grammar import level
from carsus.io.util import retry_request
from carsus.model import Ion, IonizationEnergy, Level, LevelEnergy
from carsus.util import convert_atomic_number2symbol
from uncertainties import ufloat_fromstr

IONIZATION_ENERGIES_URL = 'https://physics.nist.gov/cgi-bin/ASD/ie.pl'
IONIZATION_ENERGIES_VERSION_URL = 'https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml'

CARSUS_DATA_NIST_IONIZATION_URL = 'https://raw.githubusercontent.com/s-rathi/carsus-data-nist/main/html_files/ionization_energies.html'

logger = logging.getLogger(__name__)

basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data', 'basic_atomic_data.csv')
basic_atomic_data =pd.read_csv(basic_atomic_data_fname)

def download_ionization_energies(
        nist_url,
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
        Downloads ionization energies data from the NIST Atomic Spectra Database
        Parameters
        ----------
        nist_url: bool
            If False or None, downloads data from the carsus-dat-nist repository,
            else, downloads data from the NIST Atomic Weights and Isotopic Compositions Database.
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
    data = {k:"on" if v is True else v for k, v in data.items()}

    if not nist_url:
        logger.info("Downloading ionization energies from the carsus-data-nist repo.")
        basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data', 'basic_atomic_data.csv')
        basic_atomic_data =pd.read_csv(basic_atomic_data_fname)     
        
        atomic_number_mapping = dict(zip(basic_atomic_data['symbol'], basic_atomic_data['atomic_number']))

        atomic_numbers = [atomic_number_mapping.get(name) for name in spectra.split('-')]
        if atomic_numbers is None:
            return "Invalid atomic name"

        max_atomic_number = max(atomic_numbers)
        response = requests.get(CARSUS_DATA_NIST_IONIZATION_URL, verify=False)
        carsus_data = response.text
        extracted_content = []
        for line in carsus_data.split('\n'):
            if f' {max_atomic_number + 1} ' in line:
                break
            extracted_content.append(line)      
        return '\n'.join(extracted_content)
    
    else:
        logger.info("Downloading ionization energies from the NIST Atomic Spectra Database.")
        r = retry_request(url=IONIZATION_ENERGIES_URL, method="post", data=data)
        return r.text


class NISTIonizationEnergiesParser(BaseParser):
    """
        Class for parsers for the Ionization Energies Data from the NIST Atomic Spectra
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
            lvl = pd.Series(index=["term", "spin_multiplicity", "L", "parity", "J"], dtype='float64')

            try:
                lvl_tokens = level.parseString(ground_level)
            except ParseException:
                raise

            lvl["parity"] = lvl_tokens["parity"]

            try:
                lvl["J"] = lvl_tokens["J"]
            except KeyError:
                pass

            # To handle cases where the ground level J has not been understood:
            # Take as assumption J=0
            if (np.isnan(lvl["J"])):
                lvl["J"] = '0'
                logger.warning(f"Set `J=0` for ground state of species `{convert_atomic_number2symbol(row['atomic_number'])} {row['ion_charge']}`.")
            
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
        Class for ingesters for the Ionization Energies Data
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

        logger.info(f"Ingesting ionization energies from `{self.data_source.short_name}`.")

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

        logger.info(f"Ingesting ground levels from `{self.data_source.short_name}`.")

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


class NISTIonizationEnergies(BaseParser):
    """
    Attributes
    ----------
    base : pandas.Series
    version : str
    """
    def __init__(self, spectra, nist_url=False):
        input_data = download_ionization_energies(nist_url=nist_url,spectra=spectra)
        self.parser = NISTIonizationEnergiesParser(input_data=input_data)
        self._prepare_data()
        self._get_version()


    def _prepare_data(self):
        ionization_data = pd.DataFrame()
        ionization_data = self.parser.base[['atomic_number', 'ion_charge']].copy()
        ionization_data['ionization_energy'] = self.parser.base[
                'ionization_energy_str'].str.strip('[]()').astype(np.float64)
        ionization_data.set_index(['atomic_number',
                                   'ion_charge'], inplace=True)

        self.base = ionization_data.squeeze()

    def get_ground_levels(self):
        """Returns a DataFrame with the ground levels for the selected spectra

        Returns
        -------
        pd.DataFrame
            DataFrame with ground levels
        """
        levels = self.parser.prepare_ground_levels()
        levels['g'] = 2*levels['J'] + 1
        levels['g'] = levels['g'].astype(np.int)
        levels['energy'] = 0.
        levels = levels[['g', 'energy']]
        levels = levels.reset_index()

        return levels

    def _get_version(self):
        """Returns NIST Atomic Spectra Database version.
        """        
        selector = "body > div > table:nth-child(1) > tbody > \
                        tr:nth-child(1) > td:nth-child(1) > b"
         
        html = requests.get(IONIZATION_ENERGIES_VERSION_URL)
        bs = BeautifulSoup(html.text, 'html5lib')
        
        version = bs.select(selector)
        version = version[0].text.replace(u'\xa0', ' ')\
                    .replace('Version', ' ')

        self.version = version
