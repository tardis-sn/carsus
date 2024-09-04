import os
from carsus.io.kurucz import GFALLReader
from carsus.io.atom_data_compare import AtomDataCompare

if "XUVTOP" in os.environ:
    from carsus.io.chianti_ import ChiantiIonReader
