import pytest
from carsus.model import Atom, AtomicWeight, DataSource, Ion, IonizationEnergy
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
    assert foo_session.query(DataSource).count() == 2


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
    values = [val for val, in q]
    assert_allclose(values, [8.41364096027349e-58, 8.415894971881755e-58])


def test_atomic_weights_add_same_units(foo_session):
    q = foo_session.query((AtomicWeight.quantity + 1*u.u).value)
    values = [val for val, in q]
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


def test_ion_count(foo_session):
    assert foo_session.query(Ion).count() == 3


def test_ion_as_unique(foo_session):
    H_0 = Ion.as_unique(foo_session, atomic_number=1, ion_charge=0)
    H_02 = foo_session.query(Ion).filter(and_(Ion.atomic_number==1,
                                              Ion.ion_charge==0)).one()
    assert H_0 is H_02


def test_ion_unique_constraint(foo_session):
    duplicate = Ion(atomic_number=8, ion_charge=1)
    foo_session.add(duplicate)
    with pytest.raises(IntegrityError):
        foo_session.commit()


def test_ionization_energies_values(foo_session, O_1):
    q = foo_session.query(IonizationEnergy.quantity.value).\
        filter(IonizationEnergy.ion == O_1).\
        order_by(IonizationEnergy.quantity.value)
    values = [val for val, in q]
    assert_allclose(values, [35.121110, 35.121313])


def test_ionization_energies_convert_to(foo_session, O_1):
    q = foo_session.query(IonizationEnergy.quantity.to(u.Unit("cm-1"), equivalencies=u.spectral()).value). \
        filter(IonizationEnergy.ion == O_1). \
        order_by(IonizationEnergy.quantity.value)
    values = [val for val, in q]
    assert_allclose(values, [283270.868429, 283272.505735])


def test_ionization_energies_unique_constraint(foo_session, O_1, nist):
    O_1.ionization_energies.append(
        IonizationEnergy(quantity=35.1212323*u.eV, data_source=nist, std_dev=2e-5)
    )
    with pytest.raises(IntegrityError):
        foo_session.commit()
