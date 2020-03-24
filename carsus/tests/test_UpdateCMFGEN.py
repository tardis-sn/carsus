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
    

