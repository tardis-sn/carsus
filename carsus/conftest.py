# this contains imports plugins that configure py.test for astropy tests.
# by importing them here in conftest.py they are discoverable by py.test
# no matter how it is invoked within the source tree.

from astropy.tests.pytest_plugins import *
from carsus import init_db
from sqlalchemy.orm import Session

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

# data_dir = os.path.join(os.path.dirname(__file__), 'tests', 'data')
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)
#
# test_db_url = 'sqlite:///' + os.path.join(data_dir, 'test.db')


@pytest.fixture(scope="session")
def test_engine():
    session = init_db("sqlite://")
    session.commit()
    session.close()
    return session.get_bind()


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
