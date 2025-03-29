import os
import tempfile
import pytest
import pandas as pd
import h5py
from astropy import units as u

from carsus.metadata import MetadataHandler
from carsus.io import save_to_hdf, read_hdf_with_metadata

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'wavelength': [6562.8, 4861.3],
        'frequency': [4.57e14, 6.17e14],
        'A': [4.41e7, 8.42e6]
    })

def test_metadata_handler_init():
    meta = MetadataHandler(data_source='NIST', description='Test data')
    assert meta.metadata['data_source'] == 'NIST'
    assert meta.metadata['description'] == 'Test data'
    assert 'creation_date' in meta.metadata

def test_add_units():
    meta = MetadataHandler(data_source='test')
    meta.add_units('wavelength', 'angstrom')
    assert meta.metadata['units']['wavelength'] == 'Angstrom'
    
    meta.add_units('frequency', u.Hz)
    assert meta.metadata['units']['frequency'] == 'Hz'

def test_add_reference():
    meta = MetadataHandler(data_source='test')
    meta.add_reference(doi='10.1234/example', description='Test paper')
    assert len(meta.metadata['references']) == 1
    assert meta.metadata['references'][0]['doi'] == '10.1234/example'

def test_save_and_read_metadata(sample_data):
    with tempfile.NamedTemporaryFile(suffix='.h5') as tmp:
        # Save with metadata
        save_to_hdf(
            df=sample_data,
            path=tmp.name,
            data_source='test',
            description='test data',
            metadata={
                'references': [{
                    'doi': '10.1234/test',
                    'description': 'Test reference'
                }]
            }
        )
        
        # Read and verify
        result = read_hdf_with_metadata(tmp.name)
        
        assert 'data' in result
        assert 'metadata' in result
        assert result['metadata']['data_source'] == 'test'
        assert len(result['metadata']['references']) == 1
        assert result['metadata']['references'][0]['doi'] == '10.1234/test'
        assert 'units' in result['metadata']
        assert result['metadata']['units']['wavelength'] == 'Angstrom'

def test_git_info_in_metadata():
    meta = MetadataHandler(data_source='test')
    # can't predict the git info, but we can check the structure
    assert 'repository' in meta.metadata['git_info']
    assert 'commit_hash' in meta.metadata['git_info']
    assert 'commit_date' in meta.metadata['git_info']