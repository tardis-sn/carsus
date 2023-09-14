************
Installation
************

=============
Prerequisites
=============

#. Requires a valid Anaconda `or Miniconda <https://docs.conda.io/projects/conda/en/latest/user-guide/install/download.html>`_ installation.
#. *(optional)*. Download and extract the `Chianti Atomic Database <https://download.chiantidatabase.org/>`_ **v9.0.1** and set the following environment variable in your shell configuration file:

    .. code ::

        export XUVTOP=/path/to/chianti/root

#. *(optional)*. Download and extract the `CMFGEN Atomic Data <http://kookaburra.phyast.pitt.edu/hillier/web/CMFGEN.htm>`_.  

====================
Clone the Repository
====================

.. code ::

    $ git clone https://github.com/tardis-sn/carsus.git


=====================
Setup the Environment
=====================

.. code ::

    $ cd carsus


If you're using GNU/Linux or Intel-based Mac installation then directly create the environment using the below command:

.. code ::

    $ conda env create -f carsus_env3.yml


However, if you're using M1-based Mac (Apple Silicon), then force conda to install Intel-based packages:

.. code ::

    $ CONDA_SUBDIR=osx-64 conda env create -f carsus_env3.yml


===================
Install the Package
===================

.. code ::

    $ conda activate carsus
    $ pip install -e .


You are ready! Follow the `Quickstart for Carsus <quickstart.html>`_ guide to continue.
