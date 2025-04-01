import pandas as pd
import logging 
import numpy as np

from carsus.io.lanlads.parsers import read_atomic_levels, read_atomic_lines, read_lanl_available_ions
from carsus.util import parse_selected_elements, parse_selected_species
from carsus.util.helpers import SYMBOL2ATOMIC_NUMBER, ATOMIC_NUMBER2SYMBOL

logger = logging.getLogger(__name__)

class LANLADSReader:

    lines_loggf_threshold = -3
    levels_metastable_loggf_threshold = -3
            


    def __init__(self, lanl_data_dir, ions_symbols, priority=10):
        self.lanl_data_dir = lanl_data_dir
        self.requested_ions_tuple = set(parse_selected_species(ions_symbols))
        self.available_ions = read_lanl_available_ions(lanl_data_dir)
        
        if not self.requested_ions_tuple.issubset(self.available_ions.keys()):
            missing_ions = self.requested_ions_tuple - self.available_ions_tuple
            raise ValueError(f"Some requested ions are not available in the LANL data directory: {missing_ions}")

        self.priority = priority
        
        self.read_levels_lines()
        
    def read_levels_lines(self):
        logger.info(f"Reading levels and lines LANL data for ions: {self.requested_ions_tuple}")
        
        levels_list = []
        lines_list = []
        for ion in self.requested_ions_tuple:
            levels_fname, lines_fname = self.available_ions[ion]
            
            levels_df = read_atomic_levels(levels_fname, ion[0], ion[1])
            lines_df = read_atomic_lines(lines_fname, ion[0], ion[1])
            lines_df = lines_df[np.log10(lines_df["gf"]) > self.lines_loggf_threshold]
            levels_list.append(levels_df)
            lines_list.append(lines_df)

        levels = pd.concat(levels_list, sort=True)
        levels["priority"] = self.priority
        self.levels = levels
        self.lines = pd.concat(lines_list, sort=True)
     
