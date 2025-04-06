"""
carsus/io.py

Modified output functions with metadata support.
"""
from typing import Dict, Optional
import pandas as pd
import h5py
from astropy import units as u

from carsus.metadata.metadata import MetadataHandler

def save_to_hdf(df: pd.DataFrame, 
               path: str,
               data_source: str,
               description: Optional[str] = None,
               metadata: Optional[Dict] = None,
               **kwargs) -> None:
    """
    Save DataFrame to HDF5 file with metadata.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Data to save
    path : str
        Path to HDF5 file
    data_source : str
        Source of the atomic data
    description : str, optional
        Description of the dataset
    metadata : dict, optional
        Additional metadata to include
    **kwargs
        Additional arguments passed to pandas.DataFrame.to_hdf
    """
    # Initialize metadata handler
    meta_handler = MetadataHandler(data_source=data_source, 
                                  description=description)
    
    # Add default units for known columns
    default_units = {
        'wavelength': 'angstrom',
        'frequency': 'Hz',
        'energy': 'eV',
        'gf': None,  # dimensionless
        'A': '1/s',
        'mass': 'u'
    }
    
    for col in df.columns:
        if col in default_units and default_units[col] is not None:
            meta_handler.add_units(col, default_units[col])
    
    # Add additional metadata if provided
    if metadata is not None:
        if 'references' in metadata:
            for ref in metadata['references']:
                meta_handler.add_reference(**ref)
        if 'units' in metadata:
            for col, unit in metadata['units'].items():
                meta_handler.add_units(col, unit)
    
    # Save DataFrame to HDF5
    df.to_hdf(path, key='atomic_data', **kwargs)
    
    # Add metadata to the HDF5 file
    meta_handler.add_to_hdf(path)

def read_hdf_with_metadata(path: str) -> Dict:
    """
    Read HDF5 file with metadata.
    
    Parameters
    ----------
    path : str
        Path to HDF5 file
        
    Returns
    -------
    dict
        Dictionary with 'data' (DataFrame) and 'metadata' keys
    """
    with h5py.File(path, 'r') as f:
        # Read atomic data
        df = pd.read_hdf(path, key='atomic_data')
        
        # Read metadata if present
        metadata = {}
        if 'metadata' in f:
            metadata_group = f['metadata']
            
            # Read attributes
            metadata.update(dict(metadata_group.attrs))
            
            # Read units if present
            if 'units' in metadata_group:
                metadata['units'] = dict(metadata_group['units'].attrs)
                
            # Read references if present
            if 'references' in metadata_group:
                metadata['references'] = []
                for ref_name in metadata_group['references']:
                    ref_group = metadata_group['references'][ref_name]
                    metadata['references'].append(dict(ref_group.attrs))
                    
            # Read git info if present
            if 'git_info' in metadata_group:
                metadata['git_info'] = dict(metadata_group['git_info'].attrs)
    
    return {
        'data': df,
        'metadata': metadata
    }