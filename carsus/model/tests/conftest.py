
import pytest
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from astropy import units as u
from carsus.model import Base, Atom, DataSource, AtomWeight,\
    Ion, IonizationEnergy, Level, LevelEnergy, Line, LineAValue, LineWavelength, LineGFValue,\
    ECollision, ECollisionEnergy, ECollisionTempStrength

data_dir = os.path.join(os.path.dirname(__file__), 'data')
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

foo_db_url = 'sqlite:///' + os.path.join(data_dir, 'foo.db')


@pytest.fixture
def memory_session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return Session(bind=engine)


@pytest.fixture(scope="session")
def foo_engine():
    engine = create_engine(foo_db_url)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    session = Session(bind=engine)

    # atoms
    h = Atom(atomic_number=1, symbol='H')
    ne = Atom(atomic_number=10, symbol='Ne')

    # data sources
    nist = DataSource(short_name='nist')
    ku = DataSource(short_name='ku')
    ch = DataSource(short_name="chianti")

    # atomic weights
    h.weights = [
        AtomWeight(quantity=1.00784*u.u, data_source=nist, uncert=4e-3),
        AtomWeight(quantity=1.00811*u.u, data_source=ku, uncert=4e-3),
    ]

    # ions
    h0 = Ion(
        atomic_number=1,
        ion_charge=0,
        ionization_energies=[
            IonizationEnergy(quantity=13.5984*u.eV, data_source=nist, method="th")
        ])

    ne1 = Ion(
        atomic_number=10,
        ion_charge=1,
        ionization_energies=[
            IonizationEnergy(quantity=40.96296*u.eV, data_source=nist, method="th"),
            IonizationEnergy(quantity=40.97*u.eV, data_source=nist, method="m")
        ]
    )

    ne2 = Ion(
        atomic_number=10,
        ion_charge=2,
        ionization_energies=[
            IonizationEnergy(quantity=48.839 * u.eV, data_source=nist, method="th")
        ]
    )

    # levels
    ne2_lvl0_ku = Level(
        ion=ne2, data_source=ku, level_index=0,
        configuration="2s2.2p5", term="2P1.5", L="P", J=1.5, spin_multiplicity=2, parity=1,
        energies=[
            LevelEnergy(quantity=0, data_source=ku, method="m"),
            LevelEnergy(quantity=0, data_source=ku, method="th")
        ])

    ne2_lvl1_ku = Level(
        ion=ne2, data_source=ku, level_index=1,
        configuration="2s2.2p5", term="2P0.5", L="P", J=0.5, spin_multiplicity=2, parity=1,
        energies=[
            LevelEnergy(quantity=780.4*u.Unit("cm-1"), data_source=ku, method="m"),
            LevelEnergy(quantity=780.0*u.Unit("cm-1"), data_source=ku, method="th")
        ])

    ne2_lvl2_ku = Level(
        ion=ne2, data_source=ku, level_index=2,
        configuration="2s2.2p5", term="2D2.5", L="D", J=2.5, spin_multiplicity=2, parity=0,
        energies=[
            LevelEnergy(quantity=1366.3*u.Unit("cm-1"), data_source=ku, method="m")
        ])

    ne2_lvl1_ch = Level(
        ion=ne2, data_source=ch, level_index=1,
        configuration="2s2.2p5", term="2P0.5", L="P", J=0.5, spin_multiplicity=2, parity=1,
        energies=[
            LevelEnergy(quantity=780.2*u.Unit("cm-1"), data_source=ch, method="m")
        ])

    # lines
    ne2_line0_ku = Line(
        lower_level=ne2_lvl0_ku,
        upper_level=ne2_lvl1_ku,
        data_source=ku,
        wavelengths=[
            LineWavelength(quantity=183.571*u.AA, data_source=ku)
        ],
        a_values=[
            LineAValue(quantity=5.971e-03*u.Unit("s-1"), data_source=ku)
        ],
        gf_values=[
            LineGFValue(quantity=8.792e-01, data_source=ku)
        ]
    )

    ne2_line1_ku = Line(
        lower_level=ne2_lvl0_ku,
        upper_level=ne2_lvl2_ku,
        data_source=ku,
        wavelengths=[
            LineWavelength(quantity=18.4210*u.nm, medium=1, data_source=ku)
        ],
        a_values=[
            LineAValue(quantity=5.587e-03*u.Unit("s-1"), data_source=ku)
        ],
        gf_values=[
            LineGFValue(quantity=8.238e-01, data_source=ku)
        ]
    )

    # electron collisions
    ne2_e_col0_ku = ECollision(
        lower_level=ne2_lvl0_ku,
        upper_level=ne2_lvl1_ku,
        data_source=ku,
        bt92_ttype=2,
        bt92_cups=11.16,
        energies=[
            ECollisionEnergy(quantity=0.007108*u.rydberg, data_source=ku)
        ],
        temp_strengths=[
            (ECollisionTempStrength(temp=0.0, strength=0.255)),
            (ECollisionTempStrength(temp=0.07394, strength=0.266))
        ]
    )

    session.add_all([h, ne, nist, ku, ch, h0, ne1,
                     ne2_lvl1_ch, ne2_lvl0_ku, ne2_lvl1_ku, ne2_lvl2_ku,
                     ne2_line0_ku, ne2_line1_ku, ne2_e_col0_ku])
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


