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
from carsus.model import DataSource, Ion, IonizationEnergy
from carsus.io.base import BaseParser, BaseIngester

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

    def prepare_ion_energies_df(self):
        """ Returns a new dataframe created from `base_df` that contains ionization energies data """
        ion_energies_df = self.base_df.copy()

        def parse_ion_energy_str(row):
            ion_energy_str = row['ionization_energy_str']
            if ion_energy_str == '':
                return None
            if ion_energy_str.startswith('('):
                method = 'theoretical'
                ion_energy_str = ion_energy_str.strip('(').replace('))', ')')
            elif ion_energy_str.startswith('['):
                method = 'interpolated'
                ion_energy_str = ion_energy_str.strip('[]')
            else:
                method = 'measured'
            # ToDo: Some value are given without uncertainty. How to be with them?
            ion_energy = ufloat_fromstr(ion_energy_str)
            return pd.Series([ion_energy.nominal_value, ion_energy.std_dev, method])

        ion_energies_df[['ionization_energy_value', 'ionization_energy_uncert',
                      'ionization_energy_method']] = ion_energies_df.apply(parse_ion_energy_str, axis=1)
        ion_energies_df.drop('ionization_energy_str', axis=1, inplace=True)
        ion_energies_df.set_index(['atomic_number', 'ion_charge'], inplace=True)
        return ion_energies_df


class NISTIonizationEnergiesIngester(BaseIngester):
    """
        Class for ingesters for the Ionization Energies Data from the NIST Atomic Spectra Database

        Attributes
        ----------
        parser : BaseParser instance
            (default value = NISTIonizationEnergiesParser())

        downloader : function
            (default value = download_ionization_energies)

        ds_short_name : str
            (default value = "nist-asd-ionenergy")

        Methods
        -------
        download()
            Downloads the data with the 'downloader' and loads the `parser` with it

        ingest(session)
            Persists the downloaded data into the database

        """
    ds_short_name = "nist-asd-ionenergy"

    def __init__(self, downloader=download_ionization_energies, parser=None):
        if parser is None:
            parser = NISTIonizationEnergiesParser()
        super(NISTIonizationEnergiesIngester, self). \
            __init__(parser=parser,
                     downloader=downloader)

    def download(self, spectra='h-uuh'):
        data = self.downloader(spectra=spectra)
        self.parser(data)

    def ingest(self, session):
        """ *Only* ingests ions and ionization energies *for now* """
        print "Ingesting ionization energies data"
        ion_energies_df = self.parser.prepare_ion_energies_df()
        data_source = DataSource.as_unique(session, short_name=self.ds_short_name)

        #  Select existing ions and ionization energies
        ion_energies_subq = session.query(IonizationEnergy).\
            filter(IonizationEnergy.data_source == data_source).subquery()
        exist_ions = session.query(Ion).\
            outerjoin(ion_energies_subq, Ion.ionization_energies).all()
        exist_ions_indices = list()

        for ion in exist_ions:

            #  Locate the corresponding row in the dataframe and update ion
            index = (ion.atomic_number, ion.ion_charge)
            row = ion_energies_df.loc[index]
            ion.ground_shells = row['ground_shells']
            ion.ground_level = row['ground_level']

            # If the ionization energy value for this ion is present in the dataframe
            # than update it as well
            if pd.notnull(row['ionization_energy_value']):

                # Create a new instance
                if not ion.ionization_energies:
                    ion.ionization_energies = [
                        IonizationEnergy(quantity=row['ionization_energy_value']*u.eV,
                                         data_source=data_source,
                                         std_dev=row['ionization_energy_uncert'],
                                         method=row['ionization_energy_method'])
                    ]
                # Or update the old one
                else:
                    ion_energy = ion.ionization_energies[0]  # there can be at most 1 quantity here
                    ion_energy.quantity = row['ionization_energy_value']*u.eV
                    ion_energy.std_dev = row['ionization_energy_uncert']
                    ion_energy.method = row['ionization_energy_method']

            exist_ions_indices.append(index)

        nonexist_ion_energies_df = ion_energies_df.iloc[
            ~ion_energies_df.index.isin(exist_ions_indices)
        ]

        # Create new instances for ions that were not present in the database
        for index, row in nonexist_ion_energies_df.iterrows():
            atomic_number, ion_charge = index
            new_ion = Ion(atomic_number=atomic_number, ion_charge=ion_charge,
                          ground_shells=row['ground_shells'],
                          ground_level=row['ground_level'])
            if pd.notnull(row['ionization_energy_value']):
                new_ion.ionization_energies = [
                    IonizationEnergy(quantity=row['ionization_energy_value']*u.eV,
                                     data_source=data_source,
                                     std_dev=row['ionization_energy_uncert'],
                                     method=row['ionization_energy_method'])
                ]
            session.add(new_ion)