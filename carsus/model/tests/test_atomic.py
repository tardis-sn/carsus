import pytest
from carsus.model import Atom, AtomicWeight, DataSource, Ion, Level, LevelEnergy, LevelData
from astropy import units as u
from astropy.units import UnitsError, UnitConversionError
from sqlalchemy import and_
from sqlalchemy.exc import IntegrityError
from numpy.testing import assert_allclose, assert_almost_equal

def test_atom_count(foo_session):
    assert foo_session.query(Atom).count() == 2


def test_atom_query_atom(foo_session):
    H = foo_session.query(Atom).\
        filter(Atom.symbol=="H").one()
    assert H.atomic_number == 1


def test_atom_unique_constraint(foo_session):
    fake = Atom(atomic_number=1, symbol="Fk")
    foo_session.add(fake)
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_data_source_as_unique(memory_session):
    nist = DataSource.as_unique(memory_session, short_name="nist",
                                name="National Institute of Standards and Technology")
    nist2 = DataSource.as_unique(memory_session, short_name="nist")
    assert nist is nist2


def test_data_sources_count(foo_session):
    assert foo_session.query(DataSource).count() == 3


def test_data_sources_unique_constraint(foo_session):
    duplicate = DataSource(short_name="nist")
    foo_session.add(duplicate)
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_atomic_weights_values(foo_session):
    q = foo_session.query(AtomicWeight.quantity.value).\
        filter(AtomicWeight.atomic_number==1).\
        order_by(AtomicWeight.quantity.value)
    values = [_[0] for _ in q]
    assert_allclose(values, [1.00784, 1.00811])


def test_atomic_weights_atom_relationship(foo_session, nist):
    q = foo_session.query(Atom, AtomicWeight).\
        join(Atom.quantities.of_type(AtomicWeight)).\
        filter(AtomicWeight.data_source == nist).first()

    assert_almost_equal(q.AtomicWeight.quantity.value, 1.00784)


def test_atomic_weights_convert_to(foo_session):
    q = foo_session.query(AtomicWeight.quantity.to(u.solMass).value)
    values = [_[0] for _ in q]
    assert_allclose(values, [8.41364096027349e-58, 8.415894971881755e-58])


def test_atomic_weights_add_same_units(foo_session):
    q = foo_session.query((AtomicWeight.quantity + 1*u.u).value)
    values = [_[0] for _ in q]
    assert_allclose(values, [2.00784, 2.00811])


def test_atomic_weights_unique_constraint(foo_session, H, nist):
    H.quantities.append(
        AtomicWeight(data_source=nist, quantity=666*u.u)
    )
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_atom_merge_new_quantity(foo_session, H):
    cr = DataSource.as_unique(foo_session, short_name="cr")
    H.merge_quantity(foo_session, AtomicWeight(data_source=cr, quantity=1.00754*u.u, std_dev=3e-5,))
    foo_session.commit()
    res = foo_session.query(AtomicWeight).filter(and_(AtomicWeight.atom==H,
                                                    AtomicWeight.data_source==cr)).one()
    assert_almost_equal(res.quantity.value, 1.00754)


def test_atom_merge_existing_quantity(foo_session, H, nist):
    H.merge_quantity(foo_session,  AtomicWeight(data_source=nist, quantity=1.00654*u.u, std_dev=4e-5,))
    foo_session.commit()
    res = foo_session.query(AtomicWeight).filter(and_(AtomicWeight.atom==H,
                                                    AtomicWeight.data_source==nist)).one()
    assert_almost_equal(res.quantity.value, 1.00654)


def test_atomic_quantity_convert_to(foo_session, H, nist):
    res = foo_session.query(AtomicWeight.quantity.to(u.ng).value).filter(and_(AtomicWeight.atom==H,
                                                    AtomicWeight.data_source==nist)).scalar()
    assert_almost_equal(res, 1.6735573234079996e-15)


def test_atomic_quantity_compare(foo_session):
    res = foo_session.query(AtomicWeight).filter(AtomicWeight.quantity > 1.008*u.u).count()
    assert res == 1


def test_atomic_quantity_compare_diff_units(foo_session):
    res = foo_session.query(AtomicWeight).\
        filter(AtomicWeight.quantity > 1.6738230095999998e-27*u.kg).count()
    assert res == 1


def test_atomic_quantity_compare_uncompatible_units(foo_session):
    with pytest.raises(UnitConversionError):
        res = foo_session.query(AtomicWeight).filter(AtomicWeight.quantity > 1.008*u.m)


def test_atomic_quantity_compare_with_zero(foo_session):
    res = foo_session.query(AtomicWeight).filter(AtomicWeight.quantity > 0).count()
    assert res == 2


def test_levels_count(foo_session):
    res = foo_session.query(Level).count()
    assert res == 3


def test_levels_count(foo_session):
    res = foo_session.query(Level).count()
    assert res == 3


def test_levels_as_unique_existing(foo_session):
    si_2 = Ion.as_unique(foo_session, atomic_number=14, ion_charge=1)
    si_2_lvl0_1 = foo_session.query(Level).filter(and_(
                                Level.ion == si_2,
                                Level.configuration == "3s2.3p",
                                Level.term == "2P*",
                                Level.J == 0.5)).one()
    si_2_lvl0_2 = Level.as_unique(foo_session, ion=si_2, configuration="3s2.3p", term="2P*", J=0.5)
    assert si_2_lvl0_1 is si_2_lvl0_2


def test_levels_as_unique_new(foo_session):
    si_2 = Ion.as_unique(foo_session, atomic_number=14, ion_charge=1)
    si_2_lvl3 = Level.as_unique(foo_session, ion=si_2, configuration="3s.3p2", term="2P*", J=1.5)
    res = foo_session.query(Level).count()
    import pdb; pdb.set_trace()
    assert res == 4
