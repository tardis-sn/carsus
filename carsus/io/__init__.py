import os
from carsus.io.kurucz import GFALLReader
from carsus.io.atom_data_compare import AtomDataCompare
from carsus.io.hdf import save_to_hdf, read_hdf_with_metadata  # Add these imports

if "XUVTOP" in os.environ:
    from carsus.io.chianti_ import ChiantiIonReader

__all__ = [
    'GFALLReader',
    'AtomDataCompare',
    'save_to_hdf',          # Add to __all__
    'read_hdf_with_metadata'  # Add to __all__
]

# Conditionally add ChiantiIonReader to __all__ if environment variable exists
if "XUVTOP" in os.environ:
    __all__.append('ChiantiIonReader')
