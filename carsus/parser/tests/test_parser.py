import numpy as np
import pandas as pd
from carsus.parser.base import AtomicDataParser
import pytest
import tempfile
import os

def test_parser_output():

    temp = AtomicDataParser()

    temp.make_hdf("data/si2_osc_kurucz")

#   check if the h5 file is formed
    assert os.path.exists(os.path.join(os.getcwd(),"si2_osc_kurucz.h5"))

#   extract dataframes from h5 file
    df1 = pd.read_hdf("si2_osc_kurucz.h5",key="energy_levels")
    df2 = pd.read_hdf("si2_osc_kurucz.h5",key="oscillator_strengths")

#   delete the hdf file as it's no longer required
    os.remove("si2_osc_kurucz.h5")

#   check the shapes of DataFrames
    assert df1.shape == (157,10)
    assert df2.shape == (4196,9)
