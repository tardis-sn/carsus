"""
Input module for the NIST Atomic Weights and Isotopic Compositions database
http://www.nist.gov/pml/data/comp.cfm
"""

import logging

import pandas as pd
import requests
from bs4 import BeautifulSoup

from carsus.base import basic_atomic_data_fname
from carsus.io.base import BaseParser, BasePyparser
from carsus.io.nist.weightscomp_grammar import (
    AM_SD_COL,
    AM_VAL_COL,
    ATOM_NUM_COL,
    ATOM_WEIGHT_COLS,
    AW_LWR_BND_COL,
    AW_SD_COL,
    AW_STABLE_MASS_NUM_COL,
    AW_TYPE_COL,
    AW_UPR_BND_COL,
    AW_VAL_COL,
    COLUMNS,
    INTERVAL,
    MASS_NUM_COL,
    STABLE_MASS_NUM,
    isotope,
)
from carsus.io.util import retry_request, to_nom_val_and_std_dev
from carsus.util import parse_selected_atoms

logger = logging.getLogger(__name__)

WEIGHTSCOMP_URL = "http://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl"
WEIGHTSCOMP_VERSION_URL = (
    "https://www.nist.gov/pml/atomic-weights-and-isotopic-compositions-version-history"
)
CARSUS_DATA_NIST_WEIGHTS_URL = "https://raw.githubusercontent.com/tardis-sn/carsus-data-nist/main/html_files/weights.html"


def download_weightscomp(nist_url=False, ascii="ascii2", isotype="some"):
    """
    Downloader function for the NIST Atomic Weights and Isotopic Compositions database

    Makes a GET request to download data; then extracts preformatted text

    Parameters
    ----------
    nist_url: bool
        If False, downloads data from the carsus-dat-nist repository,
        else, downloads data from the NIST Atomic Weights and Isotopic Compositions Database.
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

    if not nist_url:
        logger.info("Downloading data from the carsus-dat-nist repository")
        response = requests.get(CARSUS_DATA_NIST_WEIGHTS_URL, verify=False)
        data = response.text
        return data
    else:
        logger.info(
            "Downloading data from the NIST Atomic Weights and Isotopic Compositions Database."
        )
        r = retry_request(
            WEIGHTSCOMP_URL, method="get", params={"ascii": ascii, "isotype": isotype}
        )
        soup = BeautifulSoup(r.text, "html5lib")
        pre_text_data = soup.pre.get_text()
        pre_text_data = pre_text_data.replace(
            "\xa0", " "
        )  # replace non-breaking spaces with spaces
        return pre_text_data


class NISTWeightsCompPyparser(BasePyparser):
    """
    Class for parsers for the NIST Atomic Weights and Isotopic Compositions

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
        super(NISTWeightsCompPyparser, self).__init__(
            grammar=grammar, columns=columns, input_data=input_data
        )

    def load(self, input_data):
        super(NISTWeightsCompPyparser, self).load(input_data)
        self.base.set_index(
            [ATOM_NUM_COL, MASS_NUM_COL], inplace=True
        )  # set multiindex "atomic_number","mass_number"

    def _prepare_atomic_weights(self, atomic):
        grouped = atomic.groupby([AW_TYPE_COL])
        interval_gr = grouped.get_group(INTERVAL).copy()
        stable_mass_num_gr = grouped.get_group(STABLE_MASS_NUM).copy()

        def atomic_weight_interval_to_nom_val_and_std(row):
            nom_val, std_dev = to_nom_val_and_std_dev(
                [row[AW_LWR_BND_COL], row[AW_UPR_BND_COL]]
            )
            return pd.Series([nom_val, std_dev])

        interval_gr[[AW_VAL_COL, AW_SD_COL]] = interval_gr.apply(
            atomic_weight_interval_to_nom_val_and_std, axis=1
        )

        def atomic_weight_find_stable_atom_mass(row):
            stable_isotope = self.base.loc[row.name, int(row[AW_STABLE_MASS_NUM_COL])]
            return stable_isotope[[AM_VAL_COL, AM_SD_COL]]

        stable_mass_num_gr[[AW_VAL_COL, AW_SD_COL]] = stable_mass_num_gr.apply(
            atomic_weight_find_stable_atom_mass, axis=1
        )

        atomic.update(interval_gr)
        atomic.update(stable_mass_num_gr)
        return atomic.drop(
            [AW_TYPE_COL, AW_LWR_BND_COL, AW_UPR_BND_COL, AW_STABLE_MASS_NUM_COL],
            axis=1,
        )

    def prepare_atomic_dataframe(self):
        """Returns a new dataframe created from `base` and containing data *only* related to atoms"""
        atomic = self.base[ATOM_WEIGHT_COLS].reset_index(level=MASS_NUM_COL, drop=True)
        atomic = atomic[~atomic.index.duplicated()]
        atomic = self._prepare_atomic_weights(atomic)
        atomic = atomic[pd.notnull(atomic[AW_VAL_COL])]
        return atomic

    def prepare_isotope_dataframe(self):
        """Returns a new dataframe created from `base` and containing data *only* related to isotopes"""
        pass


class NISTWeightsComp(BaseParser):
    """
    Attributes
    ----------
    base : pandas.DataFrame
    version : str
    """

    def __init__(self, atoms="H-Pu", nist_url=False):
        input_data = download_weightscomp(nist_url)
        self.parser = NISTWeightsCompPyparser(input_data=input_data)
        self._prepare_data(atoms)
        self._get_version()

    def _prepare_data(self, atoms):
        atomic_numbers = parse_selected_atoms(atoms)

        if atomic_numbers[-1] > 94:
            raise ValueError(
                "Atoms with atomic number greater than 94 are not stable, and cannot be handled by the NIST parser."
                )

        atom_data_list = []

        for atomic_number in atomic_numbers:
            basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
            basic_atomic_data = basic_atomic_data.loc[atomic_number - 1]

            atom_masses = self.parser.prepare_atomic_dataframe()
            atom_masses = atom_masses.drop(columns="atomic_weight_std_dev")
            atom_masses = atom_masses.rename(
                columns={"atomic_weight_nominal_value": "mass"}
            )

            data = atom_masses.loc[[(atomic_number)]]
            data["symbol"] = basic_atomic_data["symbol"]
            data["name"] = basic_atomic_data["name"]

            atom_data_list.append(data)

        atom_data = pd.concat(atom_data_list)
        self.base = atom_data[["symbol", "name", "mass"]]

    def _get_version(self):
        """
        Returns NIST Atomic Weights and Isotopic Components Database version.
        """
        selector = "td"
        html = requests.get(WEIGHTSCOMP_VERSION_URL)
        bs = BeautifulSoup(html.text, "html5lib")

        version = bs.select(selector)
        version = version[0].text.split()[1]
        self.version = version
