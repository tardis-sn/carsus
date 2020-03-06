"""
=====================
TARDIS Atomic_Data test module
=====================
created on Mar 3, 2020
"""

import numpy as np
import pandas as pd
from carsus.io.output.atomic_data import Atomic_Data
import pytest


def test_atomicdata():
    
    """
    Creating Atomic_Data object and checking various members
    """
    
    atomic_data = Atomic_Data()
    first_table = atomic_data.first_table
    second_table = atomic_data.second_table
    
    assert len(list(first_table.columns)) == 11
    assert len(list(second_table.columns)) == 8
    
    assert first_table.shape[0] == 157
    assert second_table.shape[0] == 4196
    
