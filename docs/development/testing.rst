*************
Running Tests
*************

Carsus's tests are based on the 
`AstroPy Package Template <https://docs.astropy.org/projects/package-template/en/latest/index.html>`_ 
and `pytest <https://pytest.org/en/latest>`_. Then, running simple tests on your machine is 
straightforward:

.. code ::

    $ pytest carsus

==============
Optional Flags
==============

A set of flags can be appended to the above command to run different kinds of tests:

- `--remote-data`
    Run tests marked with the ``@pytest.mark.remote_data`` decorator. Requires an internet connection.

- `--refdata=/path/to/carsus-refdata`
    Run tests marked with the ``@with_refdata`` decorator. Requires the
    `tardis-sn/carsus-refdata <https://github.com/tardis-sn/carsus-refdata>`_ repository.
  
- `--cov=carsus --cov-report=xml --cov-report=html`
    Get code coverage results using the `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ plugin.

- `--arraydiff-generate-path=carsus-refdata/arraydiff`
    Generate reference files for tests marked with ``@pytest.mark.array_compare`` decorator and save them in the 
    refdata folder.

- `--arraydiff --arraydiff-reference-path=carsus-refdata/arraydiff`
    Run tests marked with ``@pytest.mark.array_compare`` decorator. 
    The tests would look for reference files in the refdata folder which can be generated using the above option.

