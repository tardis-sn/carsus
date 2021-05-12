import urllib.request, urllib.error, urllib.parse
import os
import re
import logging
from astropy import units as u

logger = logging.getLogger(__name__)

import bs4
import pandas as pd

import pathlib

# getting the data_path

from nuclear.config import get_data_dir
from nuclear.io.nndc.parsers import decay_radiation_parsers, uncertainty_parser

TARDISNUCLEAR_DATA_DIR = pathlib.Path(get_data_dir())
import datetime
from pyne import nucname
from uncertainties import ufloat_fromstr

NNDC_DECAY_RADIATION_BASE_URL = (
    "http://www.nndc.bnl.gov/nudat2/" "decaysearchdirect.jsp?nuc={nucname}&unc=nds"
)

NNDC_ARTIFICIAL_DATASET_TAG = "NNDC_DATASET_SPLITTER"


def _get_nuclear_database_path():
    if not TARDISNUCLEAR_DATA_DIR.exists():
        os.mkdir(TARDISNUCLEAR_DATA_DIR)
    return TARDISNUCLEAR_DATA_DIR / "decay_radiation.h5"


def _sanitize_isotope_string(isotope_string):
    """
    Checks if the string given is a valid isotope_string
    
    Parameters
    ----------
    isotope_string: str

    Returns
    -------
    sanitized_isotope_string: str
    """
    try:
        sanitized_isotope_string = nucname.name(isotope_string)
    except RuntimeError:
        raise ValueError(f"{isotope_string} not a valid isotope string")
    else:
        return sanitized_isotope_string


def construct_decay_radiation_url(isotope_string):
    """
    Construct the URL for downloading the decay_radiation
    Returns
    -------

    """
    isotope_string = _sanitize_isotope_string(isotope_string)

    return NNDC_DECAY_RADIATION_BASE_URL.format(nucname=isotope_string.upper())


def download_raw_decay_radiation(isotope_string):
    """
    Download the dataset from NNDC. Splitup the page into different dataset
    types. Return a list of dictionaries that contains a dataset for list entry
    Parameters
    ----------
    isotope_string: str

    Returns
    -------
    datasets: dict
    """
    nndc_data_url = construct_decay_radiation_url(isotope_string)
    logger.info(f"Downloading data from {nndc_data_url}")
    nuclear_bs = bs4.BeautifulSoup(
        urllib.request.urlopen(nndc_data_url), features="lxml"
    )

    for utag in nuclear_bs.find_all("u"):
        utag.insert_before(NNDC_ARTIFICIAL_DATASET_TAG)

    split_raw_dataset = str(nuclear_bs).split(NNDC_ARTIFICIAL_DATASET_TAG)

    datasets = []
    cur_dataset = {}
    for data_portion in split_raw_dataset[1:]:
        data_portion = bs4.BeautifulSoup(data_portion, features="lxml")
        data_type = data_portion.find("u").text
        if data_type.startswith("Result"):
            continue
        if data_type.startswith("Dataset"):
            if len(cur_dataset) > 0:
                datasets.append(cur_dataset)
            cur_dataset = {}
        cur_dataset[data_type] = str(data_portion)
    cur_dataset["download-timestamp"] = str(datetime.datetime.utcnow())
    datasets.append(cur_dataset)
    return datasets


DATASET_PATTERN = re.compile(r"Dataset\s#(\d):")


def parse_decay_radiation_dataset(decay_rad_dataset_dict):
    """
    Parse decay radiation dataset to dataframes and compile them

    Parameters
    ----------
    decay_rad_dataset_dict

    Returns
    -------

    """

    meta = {
        "energy_column_unit": u.keV,
        "end_point_energy_column_unit": u.keV,
        "intensity_column_unit": u.percent,
        "intensity_unc_column_unit": u.keV,
    }
    dataset = []
    for data_type, data_portion in decay_rad_dataset_dict.items():
        if DATASET_PATTERN.match(data_type):
            data_set_id = int(DATASET_PATTERN.match(data_type).group(1))
            logger.info(f"Importing new Dataset {data_set_id}")
            dataset = []
            if len(dataset) != 0:
                raise ValueError(
                    "Trying to import multiple Datasets for one "
                    "isotope. Currently, this is not supported - "
                    "please file an issue on GitHub."
                )
        elif data_type in decay_radiation_parsers:
            parser = decay_radiation_parsers[data_type]
            dataset.append(parser.parse(data_portion))
        elif data_type == "Authors":
            author_bs = bs4.BeautifulSoup(data_portion, "lxml")
            meta["authors"] = author_bs.find("body").text.split(":")[1].strip()
        elif data_type == "Citation":
            citation_bs = bs4.BeautifulSoup(data_portion, "lxml")
            citation_data = citation_bs.find("body").text.split(":")[1].split("Parent")[0]
            meta["citation"] = citation_data[:-1]
            decay_table = data_portion.split('</p>', 1)[1]
            decay_table = decay_table.split('<p></p>')[0]
            decay_table_df = pd.read_html(decay_table)[0]
            decay_table_df.columns = decay_table_df.iloc[0]
            decay_table_df = decay_table_df.drop(decay_table_df.index[0])
            for column in decay_table_df:
                if column != "DecayScheme" and column != "ENSDFfile":
                    if column == "Parent T1/2" or column == "GS-GS Q-value (keV)":
                        value = decay_table_df[column].values[0]
                        unit = ''.join(x for x in value if x.isalpha())
                        if unit == '':
                            unit = '\xa0'
                            value = value.replace('\xa0', ' \xa0 ')
                        nominal, unc = uncertainty_parser(value, split_unc_symbol=unit)
                        if column  == "GS-GS Q-value (keV)":
                            unit = column[-4:-1]
                            column = column[:-12]
                        meta[column+' value'] = str(nominal) + ' ' + unit
                        meta[column+' unc'] = str(unc) + ' ' + unit
                    else:
                        meta[column] = decay_table_df[column].values[0]
        else:
            logger.warning(f"Data Type {data_type} not known and not parsed")
    full_dataset = pd.concat(dataset)
    full_dataset["download-timestamp"] = decay_rad_dataset_dict["download-timestamp"]
    return full_dataset, meta


def download_decay_radiation(isotope_string):
    """
    Download and parse decay radiation from NNDC

    Parameters
    ----------
    isotope_string

    Returns
    -------

    """
    isotope_string = _sanitize_isotope_string(isotope_string)
    raw_datasets = download_raw_decay_radiation(isotope_string)
    decay_radiation, meta = parse_decay_radiation_dataset(raw_datasets[0])
    meta = pd.Series(meta).to_frame().reset_index()
    meta.columns = ["key", "value"]
    meta["isotope"] = isotope_string
    meta["value"] = meta["value"].astype(str)
    decay_radiation["isotope"] = isotope_string
    return decay_radiation.set_index("isotope"), meta.set_index("isotope")


def update_decay_radiation_from_ejecta(ejecta, force_update=False):
    """
    Check if all isotopes of a given ejecta are in the database and
    download if necessary.

    :param ejecta:
    :param force_update:
    :return:
    """

    for isotope in ejecta.get_all_children_nuc_name():
        print("Working on isotope", isotope)
        try:
            store_decay_radiation(isotope, force_update=force_update)
        except IOError as e:
            print(str(e))
            print("skipping")


def store_decay_radiation(isotope_string, force_update=False):
    """

    Parameters
    ----------
    isotope_string
    force_update

    Returns
    -------

    """

    isotope_string = _sanitize_isotope_string(isotope_string)
    db_fname = _get_nuclear_database_path()

    if not db_fname.exists():
        data_exists = False
    else:
        decay_radiation_db = pd.read_hdf(db_fname, "decay_radiation")
        if isotope_string in decay_radiation_db.index:
            data_exists = True
        else:
            data_exists = False

    if data_exists and not force_update:
        raise IOError(
            f"{isotope_string} is already in the database "
            "(force_update to overwrite)"
        )

    new_decay_radiation, new_meta = download_decay_radiation(isotope_string)

    if data_exists:
        decay_radiation = pd.read_hdf(db_fname, "decay_radiation")
        meta = pd.read_hdf(db_fname, "metadata")

        decay_radiation.drop(isotope_string, axis=0, inplace=True)
        meta.drop(isotope_string, axis=0, inplace=True)

        decay_radiation = decay_radiation.append(new_decay_radiation)
        meta = meta.append(new_meta)
    else:
        decay_radiation = new_decay_radiation
        meta = new_meta

    with pd.HDFStore(db_fname, mode="w") as decay_radiation_db:
        decay_radiation_db["metadata"] = meta
        decay_radiation_db["decay_radiation"] = decay_radiation


def get_decay_radiation_database():
    """
    Loads and returns the nuclear decay radiation database and returns the
    database and metadata

    Returns
    -------
    decay_radiation_db: pandas.DataFrame
    meta: pandas.DataFrame
    """

    decay_radiation_db = pd.read_hdf(_get_nuclear_database_path(), "decay_radiation")
    meta = pd.read_hdf(_get_nuclear_database_path(), "metadata")

    return decay_radiation_db, meta

