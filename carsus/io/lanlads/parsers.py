from pathlib import Path
import pandas as pd
import logging
import warnings

from astropy import units as u
from carsus.util.helpers import SYMBOL2ATOMIC_NUMBER
import re

logger = logging.getLogger(__name__)

def read_atomic_levels(fname, atomic_number, ion_charge):
    levels_df = pd.read_csv(fname, sep=r"\s+", skiprows=[0,1], usecols=[0, 4, 5], names=['level_index', 'j', 'energy'])    
    # Add atomic_number, ion_charge as new columns
    levels_df['atomic_number'] = atomic_number
    levels_df['ion_charge'] = ion_charge
    # levels_df['energy'] = (levels_df['energy'].values * u.eV).to(u.erg).value  
    levels_df['level_index'] = levels_df['level_index'] - 1
    # Set a multi-index using atomic_number, ion_charge, and level_index
    levels_df = levels_df.set_index(['atomic_number', 'ion_charge', 'level_index'], append=False)
    return levels_df

def read_atomic_lines(fname, atomic_number, ion_charge):
    """
    Reads in a plotgfl file with a multi-index comprised of:
    atomic_number, ion_charge, level_index_lower, level_index_upper
    and returns a DataFrame with only the columns:
    wavelengths, gf
    """
    lines_df = pd.read_csv(fname, sep=r"\s+", 
                skiprows=[0,1], usecols=[0, 2, 9, 10], names=['wavelength', 'gf', 'level_index_lower', 'level_index_upper'])
    lines_df['atomic_number'] = atomic_number
    lines_df['ion_charge'] = ion_charge
    # Set a multi-index using atomic_number, ion_charge, level_index_lower, and level_index_upper
    lines_df['level_index_lower'] = lines_df['level_index_lower'] - 1
    lines_df['level_index_upper'] = lines_df['level_index_upper'] - 1
    return lines_df.set_index(['atomic_number', 'ion_charge', 'level_index_lower', 'level_index_upper'])

def read_lanl_available_ions(lanl_data_dir):
    """
    Read the available ions from a LANL data directory.
    Parameters
    ----------
    lanl_data_dir : str or pathlib.Path
        Path to the LANL data directory containing element subdirectories and their files.
    Returns
    -------
    list of tuple of (int, int)
        Returns a list of (atomic_number, ion_charge) tuples indicating the available ions
        found under the specified LANL data directory.
    Notes
    -----
    - Each element subdirectory is expected to be named with a valid element symbol.
    - For each subdirectory, the function searches for files named using the pattern
      'levels_*', extracts the corresponding ion charge, and verifies the existence of
      a matching lines file ('plotgfl_*').
    - The function may issue warnings for directories or files that do not follow
      the expected naming conventions.
    """
    lanl_data_dir = Path(lanl_data_dir)
    ion_fname_dict = {}
    for element_dir in lanl_data_dir.iterdir():
        atomic_name = element_dir.name
        atomic_number = SYMBOL2ATOMIC_NUMBER.get(atomic_name.capitalize(), None)
        if atomic_number is None:
            warnings.warn(f"Element directory is not an element symbol {element_dir.name.capitalize()} - skipping")
            continue
        
        for ion_levels_fname in list(element_dir.glob('levels_*')):
            levels_fname_match = re.match(r'levels_\w+(\d+)_n\d+', ion_levels_fname.name)
            if levels_fname_match:
                ion_charge = int(levels_fname_match.group(1)) - 1
            else:
                warnings.warn(f"Levels file {ion_levels_fname} does not match the expected pattern - skipping")
                continue
            
            ion_lines_fname = element_dir / ion_levels_fname.name.replace('levels', 'plotgfl')
            if not ion_lines_fname.exists():
                warnings.warn(f"Lines file {ion_lines_fname} does not exist - skipping")
                continue
            ion_tuple = (atomic_number, ion_charge)
            ion_fname_dict[ion_tuple] = (ion_levels_fname, ion_lines_fname)
        return ion_fname_dict