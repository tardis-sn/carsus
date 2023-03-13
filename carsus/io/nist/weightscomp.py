"""
Input module for the NIST Atomic Weights and Isotopic Compositions database
http://www.nist.gov/pml/data/comp.cfm
"""

import logging
import requests
import pandas as pd

from bs4 import BeautifulSoup
from astropy import units as u
from carsus.base import basic_atomic_data_fname
from carsus.model import AtomWeight
from carsus.util import parse_selected_atoms
from carsus.io.base import BasePyparser, BaseIngester, BaseParser
from carsus.io.util import to_nom_val_and_std_dev
from carsus.util.helpers import ATOMIC_SYMBOLS_DATA
from carsus.io.nist.weightscomp_grammar import isotope, COLUMNS, ATOM_NUM_COL, MASS_NUM_COL,\
    AM_VAL_COL, AM_SD_COL, INTERVAL, STABLE_MASS_NUM, ATOM_WEIGHT_COLS, AW_STABLE_MASS_NUM_COL,\
    AW_TYPE_COL, AW_VAL_COL, AW_SD_COL, AW_LWR_BND_COL, AW_UPR_BND_COL

requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

WEIGHTSCOMP_URL = "http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
WEIGHTSCOMP_VERSION_URL = "https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-version-history"


def download_weightscomp(ascii='ascii2', isotype='some'):
    """
    Downloader function for the NIST Atomic Weights and Isotopic Compositions database

    Makes a GET request to download data; then extracts preformatted text

    Parameters
    ----------
    ascii: str
        GET request parameter, refer to the NIST docs
        (default: 'ascii')
    isotype: str
        GET request parameter, refer to the NIST docs
        (default: 'some')

    Returns
    -------
    str
        Preformatted text data

    """
    logger.info("Downloading data from the NIST Atomic Weights and Isotopic Compositions Database.")
    r = requests.get(WEIGHTSCOMP_URL, params={'ascii': ascii, 'isotype': isotype}, verify=False)
    soup = BeautifulSoup(r.text, 'html5lib')
    pre_text_data = soup.pre.get_text()
    pre_text_data = pre_text_data.replace(u'\xa0', u' ')  # replace non-breaking spaces with spaces
    return pre_text_data


class NISTWeightsCompPyparser(BasePyparser):
    """
    Class for parsers for the NIST Atomic Weights and Isotopic Compositions database

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

    prepare_atomic_dataframe()
        Returns a new dataframe created from the `base` and containing data *only* related to atoms.

    prepare_isotopic_dataframe()
        Returns a new dataframe created from the `base` and containing data *only* related to isotopes

    """

    def __init__(self, grammar=isotope, columns=COLUMNS, input_data=None):
        super(NISTWeightsCompPyparser, self).\
            __init__(grammar=grammar,
                     columns=columns,
                     input_data=input_data)

    def load(self, input_data):
        super(NISTWeightsCompPyparser, self).load(input_data)
        self.base.set_index([ATOM_NUM_COL, MASS_NUM_COL], inplace=True)  # set multiindex "atomic_number","mass_number"

    def _prepare_atomic_weights(self, atomic):
        grouped = atomic.groupby([AW_TYPE_COL])
        interval_gr = grouped.get_group(INTERVAL).copy()
        stable_mass_num_gr = grouped.get_group(STABLE_MASS_NUM).copy()

        def atomic_weight_interval_to_nom_val_and_std(row):
            nom_val, std_dev = to_nom_val_and_std_dev([row[AW_LWR_BND_COL], row[AW_UPR_BND_COL]])
            return pd.Series([nom_val, std_dev])

        interval_gr[[AW_VAL_COL, AW_SD_COL]] = interval_gr.\
            apply(atomic_weight_interval_to_nom_val_and_std, axis=1)

        def atomic_weight_find_stable_atom_mass(row):
            stable_isotope = self.base.loc[row.name, int(row[AW_STABLE_MASS_NUM_COL])]
            return stable_isotope[[AM_VAL_COL, AM_SD_COL]]

        stable_mass_num_gr[[AW_VAL_COL, AW_SD_COL]] = stable_mass_num_gr.\
            apply(atomic_weight_find_stable_atom_mass, axis=1)

        atomic.update(interval_gr)
        atomic.update(stable_mass_num_gr)
        return atomic.drop([AW_TYPE_COL, AW_LWR_BND_COL, AW_UPR_BND_COL, AW_STABLE_MASS_NUM_COL], axis=1)

    def prepare_atomic_dataframe(self):
        """ Returns a new dataframe created from `base` and containing data *only* related to atoms """
        atomic = self.base[ATOM_WEIGHT_COLS].reset_index(level=MASS_NUM_COL, drop=True)
        atomic = atomic[~atomic.index.duplicated()]
        atomic = self._prepare_atomic_weights(atomic)
        atomic = atomic[pd.notnull(atomic[AW_VAL_COL])]
        return atomic

    def prepare_isotope_dataframe(self):
        """ Returns a new dataframe created from `base` and containing data *only* related to isotopes """
        pass


class NISTWeightsCompIngester(BaseIngester):
    """
    Class for ingesters for the NIST Atomic Weights and Isotopic Compositions database

    Attributes
    ----------
    session: SQLAlchemy session

    data_source: DataSource instance
        The data source of the ingester

    parser : BaseParser instance
        (default value = NISTWeightsCompPyparser())

    downloader : function
        (default value = download_weightscomp)

    Methods
    -------
    download()
        Downloads the data with the 'downloader' and loads the `parser` with it

    ingest(session)
        Persists the downloaded data into the database

    """

    def __init__(self, session, ds_short_name="nist", parser=None, downloader=None):
        if parser is None:
            parser = NISTWeightsCompPyparser()
        if downloader is None:
            downloader = download_weightscomp
        super(NISTWeightsCompIngester, self).\
            __init__(session, ds_short_name, parser=parser, downloader=downloader)

    def download(self):
        data = self.downloader()
        self.parser(data)

    def ingest_atomic_weights(self, atomic_weights=None):

        if atomic_weights is None:
            atomic_weights = self.parser.prepare_atomic_dataframe()

        logger.info("Ingesting atomic weights from `{}`.".format(self.data_source.short_name))

        for atomic_number, row in atomic_weights.iterrows():
            weight = AtomWeight(atomic_number=atomic_number,
                                     data_source=self.data_source,
                                     quantity=row[AW_VAL_COL] * u.u,
                                     uncert=row[AW_SD_COL])
            self.session.add(weight)

    def ingest(self, atomic_weights=True):
        """ *Only* ingests atomic weights *for now* """

        if self.parser.base is None:
            self.download()

        if atomic_weights:
            self.ingest_atomic_weights()
            self.session.flush()


class NISTWeightsComp(BaseParser):
    """
    Attributes
    ----------
    base : pandas.DataFrame
    version : str
    """
    def __init__(self, atoms='H-Pu'):
        input_data = download_weightscomp()
        self.parser = NISTWeightsCompPyparser(input_data=input_data)
        self._prepare_data(atoms)
        self._get_version()

    def _prepare_data(self, atoms):
        atomic_numbers = parse_selected_atoms(atoms)
        atom_data_list = []

        for atomic_number in atomic_numbers:
            basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
            basic_atomic_data = basic_atomic_data.loc[atomic_number-1]

            atom_masses = self.parser.prepare_atomic_dataframe()
            atom_masses = atom_masses.drop(columns='atomic_weight_std_dev')
            atom_masses = atom_masses.rename(
                            columns={'atomic_weight_nominal_value': 'mass'})

            data = atom_masses.loc[[(atomic_number)]]
            data['symbol'] = basic_atomic_data['symbol']
            data['name'] = basic_atomic_data['name']

            atom_data_list.append(data)

        atom_data = pd.concat(atom_data_list)
        self.base = atom_data[['symbol', 'name', 'mass']]

    def _get_version(self):
        """
        Returns NIST Atomic Weights and Isotopic Components Database version.
        """
        selector = "td"
        html = requests.get(WEIGHTSCOMP_VERSION_URL, verify=False)
        bs = BeautifulSoup(html.text, 'html5lib')

        version = bs.select(selector)
        version = version[0].text.split()[1] 
        self.version = version
