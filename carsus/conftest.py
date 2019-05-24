# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.plugins.config import *
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus import init_db

## exceptions
# enable_deprecations_as_exceptions()

## Uncomment and customize the following lines to add/remove entries
## from the list of packages for which version numbers are displayed
## when running the tests
# try:
#     PYTEST_HEADER_MODULES['Astropy'] = 'astropy'
#     PYTEST_HEADER_MODULES['scikit-image'] = 'skimage'
#     del PYTEST_HEADER_MODULES['h5py']
# except NameError:  # needed to support Astropy < 1.0
#     pass

## Uncomment the following lines to display the version number of the
## package rather than the version number of Astropy in the top line when
## running the tests.
# import os
#
## This is to figure out the affiliated package version, rather than
## using Astropy's
# from . import version
#
# try:
#     packagename = os.path.basename(os.path.dirname(__file__))
#     TESTED_VERSIONS[packagename] = version.version
# except NameError:   # Needed to support Astropy <= 1.0.0
#     pass

def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true",
                     help="include running slow tests during run")
    parser.addoption("--test-db", dest='test-db', default=None,
                     help="filename for the testing database")


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
