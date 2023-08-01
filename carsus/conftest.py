"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

import os

from astropy.version import version as astropy_version

# For Astropy 3.0 and later, we can use the standalone pytest plugin
if astropy_version < '3.0':
    from astropy.tests.pytest_plugins import *  # noqa
    del pytest_report_header
    ASTROPY_HEADER = True
else:
    try:
        from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS
        ASTROPY_HEADER = True
    except ImportError:
        ASTROPY_HEADER = False


def pytest_configure(config):
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration

    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the tests.
        PYTEST_HEADER_MODULES.pop('Pandas', None)
        PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'

        from . import __version__
        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__

# Uncomment the last two lines in this block to treat all DeprecationWarnings as
# exceptions. For Astropy v2.0 or later, there are 2 additional keywords,
# as follow (although default should work for most cases).
# To ignore some packages that produce deprecation warnings on import
# (in addition to 'compiler', 'scipy', 'pygments', 'ipykernel', and
# 'setuptools'), add:
#     modules_to_ignore_on_import=['module_1', 'module_2']
# To ignore some specific deprecation warning messages for Python version
# MAJOR.MINOR or later, add:
#     warnings_to_ignore_by_pyver={(MAJOR, MINOR): ['Message to ignore']}
# from astropy.tests.helper import enable_deprecations_as_exceptions  # noqa
# enable_deprecations_as_exceptions()


import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus import init_db


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="include running slow tests during run")
    parser.addoption("--test-db", dest='test-db', default=None,
                     help="filename for the testing database")
    parser.addoption("--refdata", dest='refdata', default=None,
                     help="carsus-refdata folder location")
    
@pytest.fixture
def withtestdb(request):
    return request.config.getoption("--test-db")

@pytest.fixture
def withrefdata(request):
    return request.config.getoption("--refdata")

@pytest.fixture
def runslow(request):
    return request.config.getoption("--runslow")


@pytest.fixture
def memory_session():
    session = init_db('sqlite://')
    session.commit()
    return session


@pytest.fixture(scope="session")
def data_dir():
    return os.path.join(os.path.dirname(__file__), 'tests', 'data')


@pytest.fixture(scope="session")
def test_db_fname(request):
    test_db_fname = request.config.getoption("--test-db")
    if test_db_fname is None:
        pytest.skip('--testing database was not specified')
    else:
        return os.path.expandvars(os.path.expanduser(test_db_fname))


@pytest.fixture(scope="session")
def test_db_url(test_db_fname):
    return "sqlite:///" + test_db_fname


@pytest.fixture(scope="session")
def gfall_fname(data_dir):
    return os.path.join(data_dir, 'gftest.all')  # Be III, B IV, N VI


@pytest.fixture(scope="session")
def gfall_http(data_dir):
    url = "https://raw.githubusercontent.com/tardis-sn/carsus/"
    url += "master/carsus/tests/data/gftest.all"
    return url


@pytest.fixture(scope="session")
def nndc_dirname(data_dir):
    return os.path.join(data_dir, "nndc")  # Mn-52, Ni-56


@pytest.fixture(scope="session")
def test_engine(test_db_url):
    return create_engine(test_db_url)


@pytest.fixture
def test_session(test_engine, request):

    # engine.echo=True
    # connect to the database
    connection = test_engine.connect()

    # begin a non-ORM transaction
    trans = connection.begin()

    # bind an individual Session to the connection
    session = Session(bind=connection)

    def fin():
        session.close()
        # rollback - everything that happened with the
        # Session above (including calls to commit())
        # is rolled back.
        trans.rollback()
        # return connection to the Engine
        connection.close()

    request.addfinalizer(fin)

    return session


@pytest.fixture(scope="session")
def refdata_path(request):
    refdata_path = request.config.getoption("--refdata")
    if refdata_path is None:
        pytest.skip('--refdata folder path was not specified')
    else:
        return os.path.expandvars(os.path.expanduser(refdata_path))
