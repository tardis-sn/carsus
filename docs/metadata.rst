Metadata in Carsus
==================

Carsus now includes comprehensive metadata with all atomic data outputs. This metadata includes:

- Data source information
- Creation date and creator
- Physical units for all columns
- References (DOIs, bibcodes, URLs)
- Git repository information (commit hash, date)

Using Metadata
--------------

When saving data::

    from carsus.io import save_to_hdf
    
    save_to_hdf(
        df=your_dataframe,
        path='output.h5',
        data_source='NIST',
        description='My atomic dataset',
        metadata={
            'references': [
                {'doi': '10.1234/example', 'description': 'Important paper'}
            ],
            'units': {
                'custom_column': 'erg'
            }
        }
    )

When reading data::

    from carsus.io import read_hdf_with_metadata
    
    data = read_hdf_with_metadata('output.h5')
    print(data['data'])  # The atomic data
    print(data['metadata'])  # All metadata

Metadata Structure
-----------------

The metadata is stored in the HDF5 file with this structure::

    /metadata
        attributes: creation_date, creator, data_source, description
        /units
            attributes: column1=unit1, column2=unit2, ...
        /references
            /reference_1
                attributes: doi=..., description=...
            /reference_2
                attributes: bibcode=..., url=...
        /git_info
            attributes: repository=..., commit_hash=..., commit_date=...