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

- `--runslow`
    Run tests marked with the ``@slow`` decorator.

- `--test-db=/path/to/carsus-db/test_databases/test.db`
    Run test for the Carsus SQL output module. Requires data from the
    `tardis-sn/carsus-db <https://github.com/tardis-sn/carsus-refdata>`_ repository.

- `--refdata=/path/to/carsus-refdata`
    Run tests marked with the ``@with_refdata`` decorator. Requires the
    `tardis-sn/carsus-refdata <https://github.com/tardis-sn/carsus-refdata>`_ repository.

- `--ignore-glob="**/*.py" --nb-test-files --nb-exec-timeout 600`
    Run notebook tests using the `pytest-notebook <https://pytest-notebook.readthedocs.io/en/latest/>`_ plugin.
  
- `--cov=carsus --cov-report=xml --cov-report=html`
    Get code coverage results using the `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_ plugin.
