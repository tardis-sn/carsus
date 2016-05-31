
import pytest
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from carsus.model import Base, Atom, DataSource, AtomicWeight, Ion, \
    LevelEnergy, ChiantiLevel, Line, LineWavelength, LineAValue, LineGFValue, \
    ECollision, ECollisionTemp, ECollisionStrength, \
    ECollisionGFValue, ECollisionTempStrength, ECollisionEnergy
from astropy import units as u

# data_dir = os.path.join(os.path.dirname(__file__), 'data')
# if not os.path.exists(data_dir):
#     os.makedirs(data_dir)
#
# foo_db_url = 'sqlite:///' + os.path.join(data_dir, 'foo.db')


@pytest.fixture
def memory_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = Session(bind=engine)
    return session


@pytest.fixture(scope="session")
def foo_engine():
    engine = create_engine("sqlite://")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = Session(bind=engine)

    # atoms
    H = Atom(atomic_number=1, symbol='H')
    Ne = Atom(atomic_number=10, symbol='Ne')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')
    ch = DataSource(short_name='ch_v8.0.2')

    # atomic weights
    H.quantities = [
        AtomicWeight(quantity=1.00784*u.u, data_source=nist, std_dev=4e-3),
        AtomicWeight(quantity=1.00811*u.u, data_source=ku, std_dev=4e-3),
    ]

    session.add_all([H, Ne, nist, ku, ch])
    session.commit()

    # ions
    ne_0 = Ion(atom=Ne, ion_charge=1)
    ne_1 = Ion(atom=Ne, ion_charge=2)

    # levels
    ne_1_lvl0 = ChiantiLevel(ion=ne_1, data_source=ch,
                             configuration="2s2.2p5", term="2P1.5", L="P", J=1.5,
                             spin_multiplicity=2, parity=1, ch_index=1,
                             energies=[
                                 LevelEnergy(quantity=0, method="m"),
                                 LevelEnergy(quantity=0, method="th")
                             ])

    ne_1_lvl1 = ChiantiLevel(ion=ne_1, data_source=ch,
                             configuration="2s2.2p5", term="2P0.5", L="P", J=0.5,
                             spin_multiplicity=2, parity=1, ch_index=2,
                             energies=[
                                 LevelEnergy(quantity=780.4*u.Unit("cm-1"), method="m"),
                                 LevelEnergy(quantity=780.0*u.Unit("cm-1"), method="th")
                             ])
    # 2s2p6 2S0.5
    ne_1_lvl2 = ChiantiLevel(ion=ne_1, data_source=ch,
                             configuration="2s2p6", term="2S0.5", L="S", J=0.5,
                             spin_multiplicity=2, parity=0, ch_index=3,
                             energies=[
                                 LevelEnergy(quantity=217047.594*u.Unit("cm-1"), method="m"),
                                 LevelEnergy(quantity=217048*u.Unit("cm-1"), method="th")
                             ])

    # lines
    ne_1_line0 = Line(
        source_level=ne_1_lvl0,
        target_level=ne_1_lvl1,
        data_source=ch,
        wavelengths=[
            LineWavelength(quantity=1*u.AA)
        ],
        a_values=[
            LineAValue(quantity=48*u.Unit("s-1"))
        ],
        gf_values=[
            LineGFValue(quantity=23)
        ]
    )

    ne_1_line1 = Line(
        source_level=ne_1_lvl0,
        target_level=ne_1_lvl2,
        data_source=ch)

    lw1 = LineWavelength(quantity=10*u.AA,
                         line=ne_1_line1)

    # electron collisions
    ne_1_col = ECollision(
        source_level=ne_1_lvl0,
        target_level=ne_1_lvl1,
        data_source=ch,
        bt92_ttype=2,
        bt92_cups=11.16,
        energies=[
            ECollisionEnergy(quantity=0.007108*u.rydberg)
        ],
        temp_strengths_tuple=[
            (ECollisionTemp(quantity=0.0), ECollisionStrength(quantity=0.255)),
            (ECollisionTemp(quantity=0.07394), ECollisionStrength(quantity=0.266))
        ]
    )

    session.add_all([ne_1, ne_0,
                     ne_1_lvl0, ne_1_lvl1, ne_1_lvl2,
                     ne_1_line0, ne_1_lvl1, ne_1_col])
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


