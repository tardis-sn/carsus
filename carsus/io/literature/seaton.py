"""
Reader for Seaton opacity data (Seaton 1992).

This module provides functionality to read continuum opacity data from 
the Seaton archive (VI/80 at CDS).
"""

import pandas as pd
import requests
import io
import gzip


def get_seaton_opacity_df(file_name):
    """
    Load and parse a Seaton opacity table from the CDS archive.
    
    Parameters
    ----------
    file_name : str
        The name of the gzip-compressed file to load from the Seaton 
        archive (e.g., 's92.201.gz').
    
    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns:
        - logT : float
            Log10 of temperature in Kelvin
        - logNe : float
            Log10 of electron density
        - kappa_planck : float
            Planck-weighted mean opacity
        - kappa_rosseland : float
            Rosseland-weighted mean opacity
    """
    url = f"https://cdsarc.cds.unistra.fr/ftp/VI/80/{file_name}"
    response = requests.get(url)

    with gzip.GzipFile(fileobj=io.BytesIO(response.content)) as f:
        lines = [line.decode('utf-8').strip() for line in f.readlines() if line.strip()]

    data_rows = []
    current_log_t = None

    # Standard Seaton format: logT = Index / 40
    for line in lines[1:]:  # Skip metadata line
        parts = line.split()
        if not parts:
            continue

        # Detect Block Header: (e.g., 140 14 68 2)
        # These define the constant log(T) for the following rows
        if len(parts) == 4 and float(parts[0]) >= 140 and '.' not in parts[0]:
            current_log_t = float(parts[0]) / 40.0
            continue

        # Parse Data Rows: [Index] [logNe] [Planck_Opacity] [Rosseland_Opacity]
        if current_log_t is not None and len(parts) >= 4:
            try:
                data_rows.append({
                    'logT': current_log_t,
                    'logNe': float(parts[1]),
                    'kappa_planck': float(parts[2]),
                    'kappa_rosseland': float(parts[3])
                })
            except ValueError:
                continue

    return pd.DataFrame(data_rows)
