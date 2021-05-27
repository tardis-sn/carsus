*************
Running Tests
*************

.. code ::

    pytest carsus <flags>

**Flags:**

- `pytest carsus --remote-data`

.. code ::

    pytest carsus --runslow 

.. code ::

    pytest carsus --test-db=$BUILD_DIR/carsus-db/test_databases/test.db 

.. code ::

    pytest carsus --refdata=$BUILD_DIR/carsus-refdata 

.. code ::

    pytest carsus --cov=carsus --cov-report=xml --cov-report=html
