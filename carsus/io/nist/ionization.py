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
from bs4 import BeautifulSoup
from pyparsing import ParseException

import carsus
from carsus.io.base import BaseParser
from carsus.io.nist.ionization_grammar import level
from carsus.io.util import retry_request
from carsus.util import convert_atomic_number2symbol
from uncertainties import ufloat_fromstr

IONIZATION_ENERGIES_URL = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
IONIZATION_ENERGIES_VERSION_URL = (
    "https://physics.nist.gov/PhysRefData/ASD/Html/verhist.shtml"
)

CARSUS_DATA_NIST_IONIZATION_URL = "https://raw.githubusercontent.com/tardis-sn/carsus-data-nist/main/html_files/ionization_energies.html"

logger = logging.getLogger(__name__)


def download_ionization_energies(
    spectra="h-uuh",
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
    biblio=False,
    nist_url=False,
):
    """
    Downloads ionization energies data from the NIST Atomic Spectra Database
    Parameters
    ----------
    nist_url: bool
        If False, downloads data from the carsus-dat-nist repository,
        else, downloads data from the NIST Atomic Weights and Isotopic Compositions Database.
    spectra: str
        (default value = 'h-uuh')
    Returns
    -------
    str
        Preformatted text data
    """
    data = {
        "spectra": spectra,
        "units": e_unit,
        "format": format_,
        "at_num_out": at_num_out,
        "sp_name_out": sp_name_out,
        "ion_charge_out": ion_charge_out,
        "el_name_out": el_name_out,
        "seq_out": seq_out,
        "shells_out": shells_out,
        "conf_out": conf_out,
        "level_out": level_out,
        "ion_conf_out": ion_conf_out,
        "e_out": e_out,
        "unc_out": unc_out,
        "biblio": biblio,
    }

    data = {k: v for k, v in data.items() if v is not False}
    data = {k: "on" if v is True else v for k, v in data.items()}

    if not nist_url:
        logger.info("Downloading ionization energies from the carsus-data-nist repo.")
        if spectra == "h-uuh":
            response = requests.get(CARSUS_DATA_NIST_IONIZATION_URL, verify=False)
            return response.text
        else:
            basic_atomic_data_fname = os.path.join(
                carsus.__path__[0], "data", "basic_atomic_data.csv"
            )
            basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
            atomic_number_mapping = dict(
                zip(basic_atomic_data["symbol"], basic_atomic_data["atomic_number"])
            )
            atomic_numbers = [
                atomic_number_mapping.get(name) for name in spectra.split("-")
            ]

            if None in atomic_numbers:
                raise ValueError("Invalid atomic name")

            max_atomic_number = max(atomic_numbers)
            response = requests.get(CARSUS_DATA_NIST_IONIZATION_URL, verify=False)
            carsus_data = response.text
            extracted_content = []
            for line in carsus_data.split("\n"):
                if f" {max_atomic_number + 1} " in line:
                    break
                extracted_content.append(line)
            return "\n".join(extracted_content)

    else:
        logger.info(
            "Downloading ionization energies from the NIST Atomic Spectra Database."
        )
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
        soup = BeautifulSoup(input_data, "html5lib")
        pre_tag = soup.pre
        for a in pre_tag.find_all("a"):
            a = a.sting
        text_data = pre_tag.get_text()
        processed_text_data = ""
        for line in text_data.split("\n")[2:]:
            if line.startswith("----"):
                continue
            if line.startswith("If"):
                break
            if line.startswith("Notes"):
                break
            line.strip("|")
            processed_text_data += line + "\n"
        column_names = [
            "atomic_number",
            "ion_charge",
            "ground_shells",
            "ground_level",
            "ionization_energy_str",
        ]
        base = pd.read_csv(
            StringIO(processed_text_data),
            sep="|",
            header=None,
            usecols=range(5),
            names=column_names,
        )
        for column in ["ground_shells", "ground_level", "ionization_energy_str"]:
            base[column] = base[column].map(lambda x: x.strip())
        self.base = base

    def prepare_ioniz_energies(self):
        """Returns a new dataframe created from `base` that contains ionization energies data"""
        ioniz_energies = self.base.copy()

        def parse_ioniz_energy_str(row):
            ioniz_energy_str = row["ionization_energy_str"]
            if ioniz_energy_str == "":
                return None
            if ioniz_energy_str.startswith("("):
                method = "theor"  # theoretical
                ioniz_energy_str = ioniz_energy_str[
                    1:-1
                ]  # .strip('()') wasn't working for '(217.7185766(10))'
                # .replace('))', ')') - not clear why that exists
            elif ioniz_energy_str.startswith("["):
                method = "intrpl"  # interpolated
                ioniz_energy_str = ioniz_energy_str.strip("[]")
            else:
                method = "meas"  # measured
            # ToDo: Some value are given without uncertainty. How to be with them?
            ioniz_energy = ufloat_fromstr(ioniz_energy_str)
            return pd.Series([ioniz_energy.nominal_value, ioniz_energy.std_dev, method])

        ioniz_energies[
            [
                "ionization_energy_value",
                "ionization_energy_uncert",
                "ionization_energy_method",
            ]
        ] = ioniz_energies.apply(parse_ioniz_energy_str, axis=1)
        ioniz_energies.drop("ionization_energy_str", axis=1, inplace=True)
        ioniz_energies.set_index(["atomic_number", "ion_charge"], inplace=True)

        # discard null values
        ioniz_energies = ioniz_energies[
            pd.notnull(ioniz_energies["ionization_energy_value"])
        ]

        return ioniz_energies

    def prepare_ground_levels(self):
        """Returns a new dataframe created from `base` that contains the ground levels data"""

        ground_levels = self.base.loc[
            :, ["atomic_number", "ion_charge", "ground_shells", "ground_level"]
        ].copy()

        def parse_ground_level(row):
            ground_level = row["ground_level"]
            lvl = pd.Series(
                index=["term", "spin_multiplicity", "L", "parity", "J"], dtype="float64"
            )

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
            if np.isnan(lvl["J"]):
                lvl["J"] = "0"
                logger.warning(
                    f"Set `J=0` for ground state of species `{convert_atomic_number2symbol(row['atomic_number'])} {row['ion_charge']}`."
                )

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

        ground_levels[
            ["term", "spin_multiplicity", "L", "parity", "J"]
        ] = ground_levels.apply(parse_ground_level, axis=1)

        ground_levels.rename(columns={"ground_shells": "configuration"}, inplace=True)
        ground_levels.set_index(["atomic_number", "ion_charge"], inplace=True)

        return ground_levels


class NISTIonizationEnergies(BaseParser):
    """
    Attributes
    ----------
    base : pandas.Series
    version : str
    """

    def __init__(self, spectra="h-uuh", nist_url=False):
        input_data = download_ionization_energies(spectra=spectra, nist_url=nist_url)
        self.parser = NISTIonizationEnergiesParser(input_data=input_data)
        self._prepare_data()
        self._get_version()

    def _prepare_data(self):
        ionization_data = pd.DataFrame()
        ionization_data = self.parser.base[["atomic_number", "ion_charge"]].copy()
        ionization_data["ionization_energy"] = (
            self.parser.base["ionization_energy_str"]
            .str.strip("[]()")
            .astype(np.float64)
        )
        ionization_data.set_index(["atomic_number", "ion_charge"], inplace=True)

        self.base = ionization_data.squeeze()

    def get_ground_levels(self):
        """Returns a DataFrame with the ground levels for the selected spectra

        Returns
        -------
        pd.DataFrame
            DataFrame with ground levels
        """
        levels = self.parser.prepare_ground_levels()
        levels["g"] = 2 * levels["J"] + 1
        levels["g"] = levels["g"].astype(np.int)
        levels["energy"] = 0.0
        levels = levels[["g", "energy"]]
        levels = levels.reset_index()

        return levels

    def _get_version(self):
        """Returns NIST Atomic Spectra Database version."""
        selector = "body > div > table:nth-child(1) > tbody > \
                        tr:nth-child(1) > td:nth-child(1) > b"

        html = requests.get(IONIZATION_ENERGIES_VERSION_URL)
        bs = BeautifulSoup(html.text, "html5lib")

        version = bs.select(selector)
        version = version[0].text.replace("\xa0", " ").replace("Version", " ")

        self.version = version
