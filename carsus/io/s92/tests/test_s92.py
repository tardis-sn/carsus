import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import os

from carsus.io.s92.base import S92Reader

@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "data"

@pytest.fixture
def test_s92_file(test_data_dir):
    """Create a sample s92 test file if it doesn't exist."""
    test_file = test_data_dir / "s92_test.dat"
    
    if not test_data_dir.exists():
        os.makedirs(test_data_dir)
        
    if not test_file.exists():
        # Create a simple test file with a few rows of mock data
        with open(test_file, 'w') as f:
            f.write("# Test s92 file\n")
            f.write("1 0\n")  # H I
            f.write("13.6   1.0   5.475E-18   32.88   2.99   0.0   0.0\n")
            f.write("2 0\n")  # He I
            f.write("24.6   0.25   9.492E-18   1.469   3.188   2.039   0.0\n")
            
    return test_file

def test_s92_reader_init(test_s92_file):
    """Test S92Reader initialization."""
    reader = S92Reader(file_path=test_s92_file)
    assert reader.file_path == test_s92_file

def test_s92_reader_parse(test_s92_file):
    """Test parsing the S92 file."""
    reader = S92Reader(file_path=test_s92_file)
    data = reader.dataset
    
    assert isinstance(data, pd.DataFrame)
    assert not data.empty
    assert "atomic_number" in data.columns
    assert "ion_charge" in data.columns
    
    # Check that we have the expected data
    assert len(data) == 2
    assert data["atomic_number"].iloc[0] == 1  # Hydrogen
    assert data["ion_charge"].iloc[0] == 0     # Neutral
    assert data["atomic_number"].iloc[1] == 2  # Helium
    
def test_calculate_cross_section(test_s92_file):
    """Test cross-section calculation."""
    reader = S92Reader(file_path=test_s92_file)
    
    # Test single value
    sigma = reader.calculate_cross_section(20.0, 1, 0)
    assert isinstance(sigma, (float, np.float64))
    assert sigma > 0
    
    # Test array input
    energies = np.array([10.0, 15.0, 20.0, 30.0])
    sigmas = reader.calculate_cross_section(energies, 1, 0)
    assert len(sigmas) == len(energies)
    assert sigmas[0] == 0.0  # Below threshold
    assert sigmas[1] > 0.0   # Above threshold
