"""
Module for handling metadata in Carsus atomic data outputs.
"""
import os
import datetime
import subprocess
from typing import Dict, Optional, Union
import pandas as pd
import h5py
from astropy import units as u

class MetadataHandler:
    """
    Handles creation and management of metadata for Carsus outputs.
    """
    
    def __init__(self, 
                 data_source: str,
                 description: str = None,
                 creator: str = "Carsus"):
        """
        Initialize the metadata handler.
        
        Parameters
        ----------
        data_source : str
            Source of the atomic data (e.g., "NIST", "Kurucz")
        description : str, optional
            Description of the dataset
        creator : str, optional
            Name of the creator (default: "Carsus")
        """
        self.metadata = {
            'creation_date': datetime.datetime.now().isoformat(),
            'creator': creator,
            'data_source': data_source,
            'description': description,
            'units': {},
            'references': [],
            'git_info': self._get_git_info()
        }
        
    def _get_git_info(self) -> Dict[str, Optional[str]]:
        """Get git repository information if available."""
        try:
            repo = subprocess.check_output(
                ['git', 'rev-parse', '--show-toplevel'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'],
                stderr=subprocess.DEVNULL
            ).decode('utf-8').strip()
            
            return {
                'repository': os.path.basename(repo),
                'commit_hash': commit_hash,
                'commit_date': subprocess.check_output(
                    ['git', 'show', '-s', '--format=%ci', 'HEAD'],
                    stderr=subprocess.DEVNULL
                ).decode('utf-8').strip()
            }
        except (subprocess.CalledProcessError, FileNotFoundError):
            return {
                'repository': None,
                'commit_hash': None,
                'commit_date': None
            }
    
    def add_units(self, column: str, unit: Union[str, u.Unit]) -> None:
        """
        Add unit information for a specific column.
        
        Parameters
        ----------
        column : str
            Name of the column/dataset
        unit : str or astropy.Unit
            Physical unit for the column
        """
        if isinstance(unit, str):
            unit = u.Unit(unit)
        self.metadata['units'][column] = str(unit)
    
    def add_reference(self, 
                     doi: str = None, 
                     bibcode: str = None, 
                     url: str = None,
                     description: str = None) -> None:
        """
        Add a reference to the metadata.
        
        Parameters
        ----------
        doi : str, optional
            Digital Object Identifier
        bibcode : str, optional
            ADS bibcode
        url : str, optional
            URL of the reference
        description : str, optional
            Description of the reference
        """
        ref = {}
        if doi is not None:
            ref['doi'] = doi
            ref['url'] = f"https://doi.org/{doi}"
        if bibcode is not None:
            ref['bibcode'] = bibcode
        if url is not None:
            ref['url'] = url
        if description is not None:
            ref['description'] = description
            
        self.metadata['references'].append(ref)
    
    def to_dict(self) -> dict:
        """Return metadata as a dictionary."""
        return self.metadata
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metadata to a pandas DataFrame."""
        # Flatten the dictionary for DataFrame representation
        flat_metadata = {
            'creation_date': self.metadata['creation_date'],
            'creator': self.metadata['creator'],
            'data_source': self.metadata['data_source'],
            'description': self.metadata['description'],
            'git_repository': self.metadata['git_info']['repository'],
            'git_commit_hash': self.metadata['git_info']['commit_hash'],
            'git_commit_date': self.metadata['git_info']['commit_date']
        }
        
        # Add references
        for i, ref in enumerate(self.metadata['references']):
            for key, value in ref.items():
                flat_metadata[f'reference_{i+1}_{key}'] = value
                
        return pd.DataFrame.from_dict(flat_metadata, orient='index', columns=['value'])
    
    def add_to_hdf(self, hdf_file: Union[str, h5py.File], 
                  group: str = 'metadata') -> None:
        """
        Add metadata to an HDF5 file.
        
        Parameters
        ----------
        hdf_file : str or h5py.File
            HDF5 file path or file object
        group : str, optional
            HDF5 group to store metadata (default: 'metadata')
        """
        if isinstance(hdf_file, str):
            with h5py.File(hdf_file, 'a') as f:
                self._write_metadata(f, group)
        else:
            self._write_metadata(hdf_file, group)
    
    def _write_metadata(self, hdf: h5py.File, group: str) -> None:
        """Write metadata to HDF5 group."""
        if group in hdf:
            del hdf[group]
            
        meta_group = hdf.create_group(group)
        
        # Store basic metadata as attributes
        for key, value in self.metadata.items():
            if key in ['units', 'references', 'git_info']:
                continue
            if value is not None:
                meta_group.attrs[key] = value
                
        # Store units
        if self.metadata['units']:
            units_group = meta_group.create_group('units')
            for col, unit in self.metadata['units'].items():
                units_group.attrs[col] = unit
                
        # Store references
        if self.metadata['references']:
            ref_group = meta_group.create_group('references')
            for i, ref in enumerate(self.metadata['references']):
                ref_subgroup = ref_group.create_group(f'reference_{i+1}')
                for key, value in ref.items():
                    ref_subgroup.attrs[key] = value
                    
        # Store git info
        if any(self.metadata['git_info'].values()):
            git_group = meta_group.create_group('git_info')
            for key, value in self.metadata['git_info'].items():
                if value is not None:
                    git_group.attrs[key] = value