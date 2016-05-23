
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.model import Base, Atom, DataSource, AtomicWeight, Ion, IonizationEnergy
from astropy import units as u

data_dir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

foo_db_url = 'sqlite:///' + os.path.join(data_dir, 'foo.db')


@pytest.fixture
def memory_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    return session


@pytest.fixture(scope="session")
def foo_engine():
    engine = create_engine(foo_db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = Session(bind=engine)

    # atoms
    H = Atom(atomic_number=1, symbol='H')
    O = Atom(atomic_number=8, symbol='O')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')

    # atomic weights
    H.quantities = [
        AtomicWeight(quantity=1.00784*u.u, data_source=nist, std_dev=4e-3),
        AtomicWeight(quantity=1.00811*u.u, data_source=ku, std_dev=4e-3),
    ]

    # ions
    H_0 = Ion(atom=H, ion_charge=0, ground_shells="1s", ground_level="2S<1/2>")
    O_0 = Ion(atom=O, ion_charge=0, ground_shells="1s2.2s2.2p4", ground_level="3P<2>")
    O_1 = Ion(atom=O, ion_charge=1, ground_shells="1s2.2s2.2p3", ground_level="4S*<3/2>")

    # ionization energies
    H_0.ionization_energies = [
        IonizationEnergy(quantity=13.598434*u.eV, data_source=nist, std_dev=4e-5)
    ]
    O_0.ionization_energies = [
        IonizationEnergy(quantity=13.6180540*u.eV, data_source=nist, std_dev=6e-6)
    ]
    O_1.ionization_energies = [
        IonizationEnergy(quantity=35.121110*u.eV, data_source=nist, std_dev=2e-5),
        IonizationEnergy(quantity=35.121313*u.eV, data_source=ku, std_dev=3e-5)
    ]

    session.add_all([H, O, nist, ku, H_0, O_0, O_1])
    session.commit()
    session.close()
    return engine


@pytest.fixture
def foo_session(foo_engine, request):
    # connect to the database
    connection = foo_engine.connect()

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


@pytest.fixture
def H(foo_session):
    return foo_session.query(Atom).get(1)


@pytest.fixture
def nist(foo_session):
    return DataSource.as_unique(foo_session, short_name="nist")

@pytest.fixture
def O_1(foo_session):
    return Ion.as_unique(foo_session, atomic_number=8, ion_charge=1)
