
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.model import Base, Atom, DataSource, AtomicWeight,\
    Ion,  Level, LevelEnergy, LevelData, LevelChiantiData
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
    Si = Atom(atomic_number=14, symbol='Si')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='kurucz')
    ch = DataSource(short_name='chianti')

    # atomic weights
    H.quantities = [
        AtomicWeight(quantity=1.00784*u.u, data_source=nist, std_dev=4e-3),
        AtomicWeight(quantity=1.00811*u.u, data_source=ku, std_dev=4e-3),
    ]

    # ions
    H_1 = Ion(atom=H, ion_charge=0)
    Si_1 = Ion(atom=Si, ion_charge=0)
    Si_2 = Ion(atom=Si, ion_charge=1)


    # levels
    Si_2_lvl0 = Level(ion=Si_2, L="P", J=0.5, parity=1, spin_mult=2, configuration="3s2.3p")
    Si_2_lvl1 = Level(ion=Si_2, L="P", J=1.5, parity=1, spin_mult=2, configuration="3s2.3p")
    Si_2_lvl2 = Level(ion=Si_2, L="P", J=0.5, parity=1, spin_mult=4, configuration="3s.3p2")

    # Chianti level data:
    ch_si_2_lvl0 = LevelChiantiData(
        level=Si_2_lvl0, data_source=ch, ch_index=1, energies=[
            LevelEnergy(quantity=0*u.eV, method='m'),
            LevelEnergy(quantity=0*u.eV, method='th')
        ]
    )

    ch_si_2_lvl1 = LevelChiantiData(
        level=Si_2_lvl1, data_source=ch, ch_index=2, energies=[
            LevelEnergy(quantity=287.24*u.eV, method='m'),
            LevelEnergy(quantity=287.512*u.eV, method='th')
        ]
    )

    session.add_all([H, Si, Si_1, Si_2,
                     Si_2_lvl0, Si_2_lvl1, Si_2_lvl2,
                     ch_si_2_lvl0, ch_si_2_lvl1,
                     nist, ku, ch])
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
    return foo_session.query(Atom).filter(Atom.atomic_number==1).one()


@pytest.fixture
def nist(foo_session):
    return DataSource.as_unique(foo_session, short_name="nist")
