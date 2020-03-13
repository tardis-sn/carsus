import os
import requests
import tarfile
import re
import pandas as pd
import numpy as np

URL = 'http://kookaburra.phyast.pitt.edu/hillier/cmfgen_files/atomic_data_15nov16.tar.gz'
FILE_NAME = "atomic_data.tar.gz"
ATOMIC_DATA = 'atomic/SIL/II/16sep15/si2_osc_kurucz'
SAVE_PATH = '/home/chinmay/Downloads'


class AtomicDataParser():

    """
    A parser class to extract data from si2_osc_kurucz to an hdf file.

    Methods
    -------
    download_data :
        Checks if the cmfgen tarfile is downloaded, if not,
        the tar file is downloaded to the specified path.

    load_file :
        If the required file is not already extracted,
        the required file 'si2_osc_kurucz' is extracted.

    make_hdf :
        Extracts and parses the data from ascii file to Dataframes.
        These Dataframes are then added to a hdf file.

        Parameters
        ----------
        file_path : string optional
            The path to the si2_ocs_kurucz data file.

        save_path : string optional
            Path where the hdf file is to be saved.
            Default : current working directory

    Example
    -------
    parser = AtomicDataParser()
    parser.download_data()
    atomic_data = parser.load_file()
    parser.make_hdf(atomic_data,path/where/hdf/is/saved)

    """

    def download_data(self):
        chunk_size = 128
        if(os.path.exists(os.path.join(SAVE_PATH,ATOMIC_DATA)) or os.path.exists(os.path.join(SAVE_PATH,FILE_NAME))):
            pass
        else:
            r = requests.get(URL,stream=True)
            with open(SAVE_PATH,'wb') as fd:
                for chunk in r.iter_content(chunk_size = chunk_size):
                    fd.write(chunk)

    def load_file(self):
        if (os.path.exists(os.path.join(SAVE_PATH,ATOMIC_DATA))):
            pass
        else:
            tar_file = tarfile.open(os.path.join(SAVE_PATH,FILE_NAME))
            tar_file.extract(ATOMIC_DATA,SAVE_PATH)
            tar_file.close()
        atomic_data = os.path.join(SAVE_PATH,ATOMIC_DATA)
        return atomic_data


    def make_hdf(self,file_path = os.path.join(SAVE_PATH,ATOMIC_DATA),save_path = os.getcwd()):

        with open(file_path) as file:

    #       Defining columns for Dataframes
            COLUMNS_ENERGY_LEVELS = ['Configuration', 'g', 'E(cm^-1)', '10^15 Hz', 'eV', 'Lam(A)', 'ID', 'ARAD', 'C4', 'C6']
            COLUMNS_OSCILLATOR_STRENGTHS = ['State A', 'State B', 'f', 'A', 'Lam(A)', 'i','j', 'Lam(obs)', '% Acc']

    #       Defining meta_data for h5 file
            meta_data = {}

    #       defining regex patterns for dataframe rows
            META_PATTERN = re.compile(r'^(\d{1,2}-[a-zA-Z]+-\d{4}|\d+\s|\d+\.\d+)\s*!([a-zA-Z\s]+[a-zA-Z]$)')
            PATTERN1 = re.compile(r"\d[a-z][^\s-]+")
            PATTERN2 = re.compile(r"[+-]?\d+\d+[eE][-+]?\d+|\d+\.\d+|\d+-\s+\d+|\s[+-]?\d+\s")

            energy_levels = []
            oscillator_strengths = []

            flag = 0

            line = file.readline()
            while(line):
                if(bool(re.match("\*+",line))):
                    flag = flag+1
                    line = file.readline()
                    while(bool(re.match("\*+",line)) is False):
                        line = file.readline()
                    flag = flag+1

                if(flag<=2):
                    while(bool(META_PATTERN.match(line))):
                        pair = META_PATTERN.search(line)
                        meta_data[pair.group(2)] = pair.group(1)
                        line = file.readline()
                    while(PATTERN1.match(line)):
                        ls = PATTERN1.findall(line)
                        ls.extend(PATTERN2.findall(line))
                        energy_levels.append(ls)
                        line = file.readline()
                else:
                    while(PATTERN1.match(line)):
                        ls = PATTERN1.findall(line)
                        ls.extend(PATTERN2.findall(line))
                        oscillator_strengths.append(ls)
                        line = file.readline()

                line = file.readline()

    #   converting row elements into proper data types
        for osc in oscillator_strengths:
            temp = osc[-1]
            osc.pop()
            ls = temp.split('-')
            osc.append(int(ls[0].strip()))
            osc.append(int(ls[1].strip()))
            for i in range(2,len(osc)-2):
                osc[i] = float(osc[i])
            osc.extend([np.nan,np.nan])

        for e_level in energy_levels:
            for i in range(1,len(e_level)):
                e_level[i] = float(e_level[i].strip())
            e_level[6] = int(e_level[6])

    #   forming data frames from lists of lists
        df1 = pd.DataFrame(energy_levels,columns = COLUMNS_ENERGY_LEVELS)
        df2 = pd.DataFrame(oscillator_strengths,columns = COLUMNS_OSCILLATOR_STRENGTHS)

        if(os.path.isdir(save_path) is not True):
            print("Invalid save_path. Defaulting to cwd.")
            save_path = os.getcwd()

        df1.to_hdf(os.path.join(save_path,'si2_osc_kurucz.h5'),key='energy_levels')
        df2.to_hdf(os.path.join(save_path,'si2_osc_kurucz.h5'),key = 'oscillator_strengths')
