import urllib.request, urllib.error, urllib.parse
import os
import logging

logger = logging.getLogger(__name__)

import bs4
import pandas as pd

#getting the data_path

from nuclear.config import get_data_dir
from nuclear.io.nndc.parsers import decay_radiation_parsers
TARDISNUCLEAR_DATA_DIR = get_data_dir()

from pyne import nucname

NNDC_DECAY_RADIATION_BASE_URL = 'http://www.nndc.bnl.gov/nudat2/' \
                                'decaysearchdirect.jsp?nuc={nucname}&unc=nds'

NNDC_ARTIFICIAL_DATASET_TAG = 'NNDC_DATASET_SPLITTER'
def _get_nuclear_database_path():
    if not os.path.exists(TARDISNUCLEAR_DATA_DIR):
        os.mkdir(TARDISNUCLEAR_DATA_DIR)
    return os.path.join(TARDISNUCLEAR_DATA_DIR, 'decay_radiation.h5')

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
        raise ValueError(f'{isotope_string} not a valid isotope string')
    else:
        return sanitized_isotope_string

def construct_decay_radiation_url(isotope_string):
    """
    Construct the URL for downloading the decay_radiation
    Returns
    -------

    """
    isotope_string = _sanitize_isotope_string(isotope_string)

    return NNDC_DECAY_RADIATION_BASE_URL.format(
        nucname=isotope_string.upper())


def download_raw_decay_radiation(isotope_string):
    """
    Download the dataset from NNDC. Splitup the page into different dataset
    types. Return a list of dictionaries that contains a dataset for list entry
    Parameters
    ----------
    isotope_string

    Returns
    -------

    """
    nndc_data_url = construct_decay_radiation_url(isotope_string)
    logger.info(f'Downloading data from {nndc_data_url}')
    nuclear_bs = bs4.BeautifulSoup(urllib.request.urlopen(nndc_data_url),
                                   features='lxml')

    for utag in nuclear_bs.find_all('u'):
        utag.insert_before(NNDC_ARTIFICIAL_DATASET_TAG)

    split_raw_dataset = str(nuclear_bs).split(NNDC_ARTIFICIAL_DATASET_TAG)

    datasets = []
    cur_dataset = {}
    for data_portion in split_raw_dataset[1:]:
        data_portion = bs4.BeautifulSoup(data_portion, features='lxml')
        data_type = data_portion.find('u').text
        if data_type.startswith('Result'):
            continue
        if data_type.startswith('Dataset'):
            if len(cur_dataset) > 0:
                datasets.append(cur_dataset)
            cur_dataset = {}
        cur_dataset[data_type] = str(data_portion)

    datasets.append(cur_dataset)

    return datasets


def parse_decay_radiation_dataset(decay_rad_dataset_dict):
    parsed_dataset = {}
    for data_type, data_portion in decay_rad_dataset_dict.items():
        if data_type in decay_radiation_parsers:
            parser = decay_radiation_parsers[data_type]
            parsed_dataset[data_type] = parser.parse(data_portion)
        else:
            logger.warning(f"Data Type {data_type} not known and not parsed")

    return parsed_dataset





def download_decay_radiation(isotope_string):

    nndc_data_url = construct_decay_radiation_url(isotope_string)
    logger.info(f'Downloading data from {nndc_data_url}')
    nuclear_bs = bs4.BeautifulSoup(urllib.request.urlopen(nndc_data_url))

    data_list = nuclear_bs.find_all('u')
    if data_list == []:
        raise ValueError(f'{isotope_string} is stable and does not have decay '
                         'radiation')

    data_sets = []
    current_data_set = {}
    for item in data_list:
        data_name = item.get_text()
        if data_name.startswith('Dataset'):
            if current_data_set != {}:
                data_sets.append(current_data_set)
            current_data_set = {}


        if data_name in decay_radiation_parsers:
            next_table = item.find_next('table')
            data_result = decay_radiation_parsers[data_name].parse(next_table)
            current_data_set.update(data_result)

        elif data_name.startswith('Author'):
            pass
        elif data_name.startswith('Citation'):
            pass
        else:
            print("Data \"{0}\" is not recognized".format(item.get_text()))

    if current_data_set != {}:
            data_sets.append(current_data_set)
    return data_sets

def store_decay_radiation_from_ejecta(ejecta, force_update=False):
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
    isotope_string = _sanitize_isotope_string(isotope_string)
    fname = _get_nuclear_database_path()
    with pd.HDFStore(fname, mode='a') as ds:
        if isotope_string in ds and not force_update:
            raise IOError('{0} is already in the database '
                          '(force_update to overwrite)'.format(isotope_string))
        try:
            data_set_list = download_decay_radiation(isotope_string)
        except ValueError:
            print("{0} is stable - making empty dataset".format(isotope_string))
            ds['{0}'.format(isotope_string)] = pd.DataFrame()
        else:
            for i, data_set in enumerate(data_set_list):
                for key, value in list(data_set.items()):
                    group_str = '{0}/data_set{1}/{2}'.format(isotope_string, i,
                                                             key)
                    print("Writing group", group_str)
                    ds[group_str] = value
        ds.flush()
        ds.close()



def get_decay_radiation(isotope_string, data_set_idx=0):
    isotope_string = _sanitize_isotope_string(isotope_string)
    fname = _get_nuclear_database_path()

    if not os.path.exists(fname):
        if (not input('{0} not in database - download [Y/n]'.format(
                isotope_string)).lower() == 'n'):
            store_decay_radiation(isotope_string)
        else:
            raise ValueError('{0} not in database'.format(
                        isotope_string))


    with pd.HDFStore(fname, mode='r') as ds:
        current_keys = [key for key in list(ds.keys())
                            if key.startswith('/{0}/data_set{1:d}'.format(
                isotope_string, data_set_idx))]
        if len(current_keys) == 0:
            if '/{0}'.format(isotope_string) in list(ds.keys()):
                logger.debug('{0} is stable - no decay radiation available'.format(
                    isotope_string))
                return {}
            else:
                if (not input(
                        '{0} not in database - download [Y/n]'.format(
                            isotope_string)).lower() == 'n'):
                    ds.close()
                    store_decay_radiation(isotope_string)
                else:
                    raise ValueError('{0} not in database'.format(
                        isotope_string))
        data_set = {}
        for key in current_keys:
            data_set[key.split('/')[-1]] = ds[key]
    return data_set





