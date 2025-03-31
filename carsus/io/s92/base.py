import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re
import gzip
import urllib.request
from urllib.error import HTTPError, URLError

from carsus.io.base import BaseParser

logger = logging.getLogger(__name__)

S92_FTP_URL = "ftp://cfa-ftp.harvard.edu/pub/spbase/s92.201.gz"

class S92Reader:
    """
    Reader for the Shull & Van Steenberg 1992 data files containing photoionization data.
    
    This reader processes the s92.201 data file containing photoionization cross sections
    from Shull & Van Steenberg 1992, which provides data for atoms and ions.
    
    Parameters
    ----------
    file_path : str or Path, optional
        Path to the s92.201 file. If not provided, will download from the default source.
    url : str, optional
        URL to download the file from if not provided locally.
    
    Attributes
    ----------
    file_path : Path
        Path to the s92.201 file.
    url : str
        URL for downloading the data file.
    dataset : pandas.DataFrame
        The parsed photoionization data from the s92.201 file.
    """
    
    def __init__(self, file_path=None, url=None):
        self.file_path = Path(file_path) if file_path else None
        self.url = url or S92_FTP_URL
        self._dataset = None
        
        if self.file_path is None:
            self.download_file()
    
    def download_file(self, output_path=None):
        """
        Download the s92.201.gz file from the URL.
        
        Parameters
        ----------
        output_path : str or Path, optional
            Path to save the downloaded file. If not provided, a temporary path will be used.
        
        Returns
        -------
        Path
            Path to the downloaded file.
        """
        import tempfile
        
        if output_path is None:
            # Use a temporary directory instead of hardcoding to user's home
            temp_dir = tempfile.gettempdir()
            output_path = Path(temp_dir) / "s92.201"
        else:
            output_path = Path(output_path)
        
        # Create parent directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Downloading S92 data from {self.url}")
            urllib.request.urlretrieve(self.url, output_path.with_suffix('.gz'))
            
            # If downloaded as gz, decompress it
            if self.url.endswith('.gz'):
                with gzip.open(output_path.with_suffix('.gz'), 'rb') as f_in:
                    with open(output_path, 'wb') as f_out:
                        f_out.write(f_in.read())
                output_path.with_suffix('.gz').unlink()  # Remove the .gz file
            
            self.file_path = output_path
            logger.info(f"Downloaded S92 data to {output_path}")
            return output_path
            
        except (HTTPError, URLError) as e:
            logger.error(f"Failed to download file from {self.url}: {e}")
            raise
    
    @property
    def dataset(self):
        """
        Parse and return the S92 photoionization data.
        
        Returns
        -------
        pandas.DataFrame
            The parsed photoionization data from the s92.201 file.
        """
        if self._dataset is None:
            self._dataset = self.parse_file()
        return self._dataset
    
    def parse_file(self):
        """
        Parse the s92.201 file into a pandas DataFrame.
        
        The s92.201 file contains photoionization cross-section data from
        Shull & Van Steenberg 1992. This method parses the file structure and
        extracts the tabular data into a DataFrame.
        
        Returns
        -------
        pandas.DataFrame
            The parsed photoionization data with columns for atomic number,
            ion charge, threshold energy, and fitted parameters.
        """
        if self.file_path is None:
            raise ValueError("File path is not set. Call download_file first.")
        
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        data_rows = []
        header_pattern = re.compile(r'^(\s*\d+\s+\d+)')
        
        # Process each line
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                i += 1
                continue
            
            # Check if this is a header line (contains atom/ion identifiers)
            header_match = header_pattern.match(line)
            if header_match:
                # Parse the header line with atom/ion info
                parts = line.split()
                if len(parts) >= 2:
                    atomic_number = int(parts[0])
                    ion_charge = int(parts[1])
                    
                    # The next line(s) contain the data values
                    i += 1
                    if i < len(lines):
                        data_line = lines[i].strip()
                        values = data_line.split()
                        
                        if len(values) >= 7:  # Expected number of columns
                            data_row = {
                                'atomic_number': atomic_number,
                                'ion_charge': ion_charge,
                                'threshold_energy_eV': float(values[0]),
                                'E_0': float(values[1]),
                                'sigma_0': float(values[2]),
                                'ya': float(values[3]),
                                'P': float(values[4]),
                                'yw': float(values[5]),
                                'y0': float(values[6]) if len(values) > 6 else np.nan,
                                'y1': float(values[7]) if len(values) > 7 else np.nan
                            }
                            data_rows.append(data_row)
            i += 1
        
        if not data_rows:
            logger.warning(f"No data found in {self.file_path}")
            return pd.DataFrame()
        
        df = pd.DataFrame(data_rows)
        return df
    
    def to_hdf(self, output_path):
        """
        Save the parsed photoionization data to an HDF5 file.
        
        Parameters
        ----------
        output_path : str or Path
            Path to save the HDF5 file.
        
        Returns
        -------
        None
        """
        output_path = Path(output_path)
        self.dataset.to_hdf(output_path, key='s92_photoionization')
        logger.info(f"Saved S92 photoionization data to {output_path}")

    def calculate_cross_section(self, energy, atomic_number, ion_charge):
        """
        Calculate photoionization cross-section at a given energy for specific atom/ion.
        
        Uses the Verner & Yakovlev (1995) fitting formula to calculate the
        photoionization cross-section at the requested energy.
        
        Parameters
        ----------
        energy : float or array-like
            Energy in eV at which to calculate the cross-section
        atomic_number : int
            Atomic number of the element
        ion_charge : int
            Ion charge state
            
        Returns
        -------
        float or array-like
            Photoionization cross-section in cm²
        """
        # Get parameters for this element and ion
        params = self.dataset[
            (self.dataset['atomic_number'] == atomic_number) & 
            (self.dataset['ion_charge'] == ion_charge)
        ]
        
        if params.empty:
            logger.error(f"No parameters found for Z={atomic_number}, ion charge={ion_charge}")
            return np.zeros_like(energy) if hasattr(energy, 'shape') else 0.0
        
        # Extract parameters
        E_0 = params['E_0'].values[0]
        sigma_0 = params['sigma_0'].values[0]
        ya = params['ya'].values[0]
        P = params['P'].values[0]
        yw = params['yw'].values[0]
        threshold = params['threshold_energy_eV'].values[0]
        
        # Calculate cross-section using Verner & Yakovlev formula
        x = energy / E_0
        y = x**2
        sigma = sigma_0 * ((x - 1)**2 + yw**2) * y**(0.5*P - 5.5) * (1 + np.sqrt(y/ya))**(-P)
        
        # Zero below threshold
        if hasattr(energy, 'shape'):
            sigma[energy < threshold] = 0.0
        elif energy < threshold:
            sigma = 0.0
            
        return sigma
