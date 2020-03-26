"""
=====================
TARDIS test_UpdateCMFGEN module
=====================
created on Mar 18, 2020
"""

import pytest
from carsus.Update_CMFGEN import UPDATE_CMFGEN
from bs4 import BeautifulSoup
import requests
import pandas as pd
from os import path
import os
import numpy as np
import urllib.request, shutil
import tarfile
import re
import h5py
@pytest.fixture()
def update_class():
    
    """
    This function tests the update function of class UPDATE_CMFGEN, which in turn
    checks the get_links, file_With_Extension and download_data functions. Not 
    only it checks these functions, but also returns this object of class UPDATE_CMFGEN
    on which other testing can be done.
    """
    
    a = UPDATE_CMFGEN(".Testing")
    url = "https://carsusatomicdatatest.herokuapp.com/test.html"
    a.update(url)
    return a


def testconstructor():
    
    """
    This test tests the constructor of class UPDATE_CMFGEN.
    """
    
    a = UPDATE_CMFGEN()
    assert a.hidden_folder == ".CMFGEN"
    
    b = UPDATE_CMFGEN(".test")
    assert b.hidden_folder == ".test"
    
    c = UPDATE_CMFGEN("test")
    assert c.hidden_folder == ".CMFGEN"
   
    
def testisSame(update_class):
    
    """
    This function tests the is_same function by comparing the log file. This 
    inturn checks that the log file produced by the getlinks function is correct 
    or not.
    """
    assert update_class.is_same(".Testing/CMFGEN.txt", ".Testing/CARSUSFolder@1/CARSUS/CMFGEN.txt")
    

@pytest.mark.HDF5
def test_createdHDF5(update_class):
    
    """
    This function tests whether all HDF5 files are created at correct positions. 
    Also files of incorrect extensions should not be created.
    """
    
    assert os.path.exists(".Testing/HDF5CARSUSFolder@1/CARSUS/CMFGEN.hdf5")
    assert os.path.exists(".Testing/HDF5CARSUSFolder@1/CARSUS/file2.hdf5")
    assert os.path.exists(".Testing/HDF5TARDISFolder@1/TARDIS/Folder1/Subfolder/file.hdf5")
    assert os.path.exists(".Testing/HDF5TARDISFolder@1/TARDIS/Folder1/file.hdf5")
    assert os.path.exists(".Testing/HDF5TARDISFolder@1/TARDIS/Folder2/si2_osc_kurucz.hdf5")    
    
    
@pytest.mark.HDF5
def test_HDF5():

    """
    This function checks that data is perfectly written in HDF5 file.
    It also checks that the data itself is correct or not from physics point 
    of view by taking example of si2_osc_kurucz file.
    """
    a = UPDATE_CMFGEN(".Testing")
    url = "https://carsusatomicdatatest.herokuapp.com/test.html"
    a.update(url)
    a.process_file(".Testing/TARDISFolder@1/TARDIS/Folder2/si2_osc_kurucz")
    
    first_table = pd.read_hdf(".Testing/HDF5TARDISFolder@1/TARDIS/Folder2/si2_osc_kurucz.hdf5", "0")
    second_table = pd.read_hdf(".Testing/HDF5TARDISFolder@1/TARDIS/Folder2/si2_osc_kurucz.hdf5", "1")
    
    assert len(list(first_table.columns)) == 10
    assert len(list(second_table.columns)) == 9
    
    assert first_table.shape[0] == 157
    assert second_table.shape[0] == 4196
    
    #checking if all values of g(statistical weight of the energy level) is greater than zero
    eval_dataframe = first_table[first_table["g"].astype("float64")>0.0]
    assert eval_dataframe.shape == first_table.shape
    
