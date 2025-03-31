.. _io:

*******************
Input/Output
*******************

.. _atomic_tables:

Atomic Tables
============

.. _databases:

Databases
=========

.. _nist:

NIST
----

.. _kurucz:

Kurucz
------

.. _chianti:

CHIANTI
-------

.. _zeta:

Zeta
----

.. _s92:

Shull & Van Steenberg (1992)
----------------------------

The S92Reader class provides functionality to read and process photoionization data 
from Shull & Van Steenberg (1992). This data provides photoionization cross-sections 
for atoms and ions.

Basic Usage
^^^^^^^^^^

.. code-block:: python

    from carsus.io.s92 import S92Reader
    
    # Initialize the reader with a file path
    reader = S92Reader(file_path="path/to/s92.201")
    
    # Or let it download the file automatically
    reader = S92Reader()
    
    # Get the parsed data
    data = reader.dataset
    
    # Calculate cross-section for a specific atom/ion at given energy
    energy = 30.0  # eV
    cross_section = reader.calculate_cross_section(energy, atomic_number=1, ion_charge=0)
    
    # Save data to HDF5
    reader.to_hdf("s92_data.h5")

Integration with TARDISAtomData
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can include S92 photoionization data in your TARDIS atomic dataset:

.. code-block:: python

    from carsus.io.output.base import TARDISAtomData
    from carsus.io.output.config import output_config
    
    # Create TARDISAtomData with S92 photoionization enabled
    atom_data = TARDISAtomData(
        atomic_weights,
        ionization_energies,
        gfall_reader,
        zeta_data,
        chianti_reader=chianti_reader,
        cmfgen_reader=cmfgen_reader,
        s92_photoionization=True,  # Enable S92 data
        s92_file_path="/path/to/s92.201"  # Optional: specify file path
    )
    
    # Write to HDF5 file with photoionization data
    atom_data.to_hdf("atom_data.h5")
