************
Installation
************

=============
Prerequisites
=============

#. Requires a valid Anaconda `or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_ installation.
#. Download and extract the `Chianti atomic database <https://www.chiantidatabase.org/chianti_download.html>`_ and set the following environment variable in your shell configuration file:

.. code ::

    export XUVTOP=/path/to/chianti/root

====================
Clone the repository
====================

.. code ::

    $ git clone https://github.com/tardis-sn/carsus.git


=====================
Setup the environment
=====================

.. code ::

    $ cd carsus
    $ conda env create -f carsus_env3.yml


===================
Install the package
===================

.. code ::

    $ conda activate carsus
    $ python setup.py install


You are ready! Follow the `Quickstart for Carsus <quickstart.html>`_ guide to continue.
