"""
Input module for the NIST Ionization Energies database
http://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
"""

import requests
import pandas as pd
from bs4 import BeautifulSoup
from StringIO import StringIO
from astropy import units as u
from uncertainties import ufloat_fromstr
from pyparsing import ParseException
from carsus.model import DataSource, Ion, IonizationEnergy
from carsus.io.base import BaseParser, BaseIngester
from carsus.io.nist.ionization_grammar import level

IONIZATION_ENERGIES_URL = 'http://physics.nist.gov/cgi-bin/ASD/ie.pl'


def download_ionization_energies(spectra='h-uuh', e_out=0, e_unit=1, format_=1, at_num_out=True, sp_name_out=False,
                                 ion_charge_out=True, el_name_out=False, seq_out=False, shells_out=True,
                                 conf_out=False, level_out=True, ion_conf_out=False, unc_out=True, biblio=False):
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

    data = {k: v for k, v in data.iteritems() if v is not False}

    print "Downloading ionization energies data from http://physics.nist.gov/PhysRefData/ASD/ionEnergy.html"
    r = requests.post(IONIZATION_ENERGIES_URL, data=data)
    return r.text


class NISTIonizationEnergiesParser(BaseParser):
    """
        Class for parsers for the Ionization Energies Data from the NIST Atomic Spectra Database
        Attributes
        ----------
        base_df : pandas.DataFrame
        grammar : pyparsing.ParseElement
            (default value = isotope)
        columns : list of str
            (default value = COLUMNS)
        Methods
        -------
        load(input_data)
            Parses the input data and stores the results in the `base_df` attribute
        prepare_ion_energies_df()
            Returns a new dataframe created from `base_df` that contains ionization energies data
    """

    def load(self, input_data):
        soup = BeautifulSoup(input_data, 'html5lib')
        pre_tag = soup.pre
        for a in pre_tag.find_all("a"):
            a = a.sting
        text_data = pre_tag.get_text()
        column_names = ['atomic_number', 'ion_charge', 'ground_shells', 'ground_level', 'ionization_energy_str']
        base_df = pd.read_csv(StringIO(text_data), sep='|', header=None,
                         usecols=range(5), names=column_names, skiprows=3, skipfooter=1)
        for column in ['ground_shells', 'ground_level', 'ionization_energy_str']:
                base_df[column] = base_df[column].map(lambda x: x.strip())
        self.base_df = base_df

    def prepare_ioniz_energies_df(self):
        """ Returns a new dataframe created from `base_df` that contains ionization energies data """
        ioniz_energies_df = self.base_df.copy()

        def parse_ioniz_energy_str(row):
            ioniz_energy_str = row['ionization_energy_str']
            if ioniz_energy_str == '':
                return None
            if ioniz_energy_str.startswith('('):
                method = 'theor' # theoretical
                ioniz_energy_str = ioniz_energy_str.strip('(').replace('))', ')')
            elif ioniz_energy_str.startswith('['):
                method = 'intrpl' # interpolated
                ioniz_energy_str = ioniz_energy_str.strip('[]')
            else:
                method = 'meas'  # measured
            # ToDo: Some value are given without uncertainty. How to be with them?
            ioniz_energy = ufloat_fromstr(ioniz_energy_str)
            return pd.Series([ioniz_energy.nominal_value, ioniz_energy.std_dev, method])

        ioniz_energies_df[['ionization_energy_value', 'ionization_energy_uncert',
                      'ionization_energy_method']] = ioniz_energies_df.apply(parse_ioniz_energy_str, axis=1)
        ioniz_energies_df.drop('ionization_energy_str', axis=1, inplace=True)
        ioniz_energies_df.set_index(['atomic_number', 'ion_charge'], inplace=True)

        # discard null values
        ioniz_energies_df = ioniz_energies_df[pd.notnull(ioniz_energies_df["ionization_energy_value"])]

        return ioniz_energies_df

    def prepare_ground_levels_df(self):
        """ Returns a new dataframe created from `base_df` that contains the ground levels data """

        ground_levels_df = self.base_df.loc[:,["atomic_number", "ion_charge",
                                             "ground_shells", "ground_level"]].copy()

        def parse_ground_level(row):
            ground_level = row["ground_level"]
            lvl = pd.Series(index=["term", "spin_multiplicity", "L", "parity", "J"])
            try:
                lvl_tokens = level.parseString(ground_level)
            except ParseException:
                # For some ions the ground level is not given
                pass
            else:
                lvl["J"] = lvl_tokens["J"]
                if "ls_term" in lvl_tokens:
                    lvl["term"] = "".join([str(_) for _ in lvl_tokens["ls_term"]])
                    lvl["spin_multiplicity"] = lvl_tokens["ls_term"]["mult"]
                    lvl["L"] = lvl_tokens["ls_term"]["L"]
                    lvl["parity"] = lvl_tokens["ls_term"]["parity"]
                elif "jj_term" in lvl_tokens:
                    lvl["term"] = "".join([str(_) for _ in lvl_tokens["jj_term"]])
            return lvl

        ground_levels_df[["term", "spin_multiplicity",
                          "L", "parity", "J"]] = ground_levels_df.apply(parse_ground_level, axis=1)

        ground_levels_df.rename(columns={"ground_shells": "configuration"}, inplace=True)
        ground_levels_df.set_index(['atomic_number', 'ion_charge'], inplace=True)

        return ground_levels_df


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

        Methods
        -------
        download()
            Downloads the data with the 'downloader' and loads the `parser` with it
        ingest(session)
            Persists the downloaded data into the database
        """

    def __init__(self, session, ds_short_name="nist-asd", downloader=None, parser=None):
        if parser is None:
            parser = NISTIonizationEnergiesParser()
        if downloader is None:
            downloader = download_ionization_energies
        super(NISTIonizationEnergiesIngester, self). \
            __init__(session, ds_short_name=ds_short_name, parser=parser, downloader=downloader)

    def download(self, spectra='h-uuh'):
        data = self.downloader(spectra=spectra)
        self.parser(data)

    def ingest_ionization_energies(self):
        print("Ingesting ionization energies from {}".format(self.data_source.short_name))

        ioniz_energies_df = self.parser.prepare_ioniz_energies_df()

        for index, row in ioniz_energies_df.iterrows():
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

    def ingest(self, ionization_energies=True):

        print("Ingesting data from {}".format(self.data_source.short_name))

        if ionization_energies:
            self.ingest_ionization_energies()
