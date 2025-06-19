from dataclasses import dataclass
from typing import Dict, Any
import pandas as pd
import re

@dataclass
class JLODKData:
    """Dataclass to hold parsed JLODK atomic data"""
    meta: Dict[str, Any]
    levels: pd.DataFrame
    lines: pd.DataFrame

def parse_jlodk_grasp_file(file_path: str) -> JLODKData:
    """
    Parse a JLODK atomic data file and return structured data.
    
    Parameters
    ----------
    file_path : str
        Path to the JLODK file to parse
        
    Returns
    -------
    JLODKData
        Dataclass containing metadata, levels, and lines dataframes
    """
    # Compile patterns for sequential matching
    compiled_patterns = [
        (re.compile(r'# (\d+) (\d+)'), ('atomic_number', 'ion_number'), (int, int)),           # line 3  
        (re.compile(r'# (\d+) (\d+)'), ('number_of_levels', 'number_of_lines'), (int, int)),    # line 4
        (re.compile(r'# IP = ([\d.]+)'), ('ionization_potential',), (float,)),                  # line 6
        (re.compile(r'# Energy levels'), ('energy_levels_header',), (None,))                    # line 7
    ]

    current_pattern_index = 0
    meta_data = {}
    
    with open(file_path, "r") as fh:
        line_number = 0
        for line_number, line in enumerate(fh):
            # Use the compiled patterns to match sequentially
            current_pattern, current_pattern_names, current_pattern_dtypes = compiled_patterns[current_pattern_index]
            if current_pattern.match(line.strip()):
                if current_pattern_names[0] == 'energy_levels_header':
                    break
                for meta_data_name, meta_data_value, meta_data_dtype in zip(current_pattern_names, current_pattern.match(line.strip()).groups(), current_pattern_dtypes):
                    meta_data[meta_data_name] = meta_data_dtype(meta_data_value)
                current_pattern_index += 1
    
    data_line_start = line_number + 1
    
    # Parse levels data
    levels_df = pd.read_csv(
        file_path, 
        skiprows=data_line_start, 
        nrows=meta_data['number_of_levels'], 
        sep=r'\s+', 
        names=["level_id", "g", "parity", "energy", "configuration", "ls_configuration"]
    )
    
    # Parse lines data
    lines_df = pd.read_csv(
        file_path, 
        skiprows=meta_data['number_of_levels'] + data_line_start + 1, 
        nrows=meta_data['number_of_lines'], 
        sep=r'\s+',
        names=["level_id_lower", "level_id_upper", "wavelength", "gua", "log_gf"]
    )
    
    return JLODKData(meta=meta_data, levels=levels_df, lines=lines_df)