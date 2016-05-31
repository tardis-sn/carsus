from .meta import Base, UniqueMixin, DBQuantity, QuantityMixin

from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.associationproxy import association_proxy
from sqlalchemy import Column, Integer, String, Float, ForeignKey, UniqueConstraint, and_
from astropy import units as u


class Atom(Base):
    __tablename__ = "atom"
    atomic_number = Column(Integer, primary_key=True)
    symbol = Column(String(5), nullable=False)
    name = Column(String(150))
    group = Column(Integer)
    period = Column(Integer)
    quantities = relationship("AtomicQuantity",
                    backref='atom',
                    cascade='all, delete-orphan')

    ions = relationship("Ion", back_populates="atom")

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)

    def merge_quantity(self, session, source_qty):
        """ Updates an existing quantity or creates a new one"""
        qty_cls = source_qty.__class__
        try:
            target_qty = session.query(qty_cls).\
                         filter(and_(qty_cls.atom==self,
                                     qty_cls.data_source==source_qty.data_source)).one()
            target_qty.quantity = source_qty.quantity
            target_qty.std_dev = source_qty.std_dev

        except NoResultFound:

            self.quantities.append(source_qty)


class AtomicQuantity(Base):
    __tablename__ = "atomic_quantity"

    id = Column(Integer, primary_key=True)
    type = Column(String(20))
    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.id'), nullable=False)

    _value = Column(Float, nullable=False)
    unit = u.Unit("")

    # Public interface for value is via the Quantity object
    @hybrid_property
    def quantity(self):
        return DBQuantity(self._value, self.unit)

    @quantity.setter
    def quantity(self, qty):
        self._value = qty.to(self.unit).value

    std_dev = Column(Float)

    data_source = relationship("DataSource")

    __table_args__ = (UniqueConstraint('type', 'atomic_number', 'data_source_id'),)
    __mapper_args__ = {
        'polymorphic_on':type,
        'polymorphic_identity':'atomic_quantity',
        'with_polymorphic' : '*'
    }

    def __repr__(self):
        return "<Quantity: {0}, value: {1}>".format(self.type, self._value)


class AtomicWeight(AtomicQuantity):
    unit = u.u

    __mapper_args__ = {
        'polymorphic_identity':'atomic_weight'
    }


class Ion(UniqueMixin, Base):
    __tablename__ = "ion"

    @classmethod
    def unique_hash(cls, atomic_number, ion_charge, *args, **kwargs):
        return "i{0}_{1}".format(atomic_number, ion_charge)

    @classmethod
    def unique_filter(cls, query, atomic_number, ion_charge, *args, **kwargs):
        return query.filter(and_(Ion.atomic_number == atomic_number,
                                 Ion.ion_charge == ion_charge))

    id = Column(Integer, primary_key=True)
    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), nullable=False)
    ion_charge = Column(Integer, nullable=False)

    levels = relationship("Level", back_populates='ion')
    atom = relationship("Atom", back_populates='ions')

    __table_args__ = (UniqueConstraint('atomic_number', 'ion_charge'),)

    def __repr__(self):
        return "<Ion z={0} +{1}>".format(self.atomic_number, self.ion_charge)


class Level(Base):
    __tablename__ = "level"

    id = Column(Integer, primary_key=True)
    ion_id = Column(Integer, ForeignKey('ion.id'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.id'))
    type = Column(String(20))
    configuration = Column(String(50))
    L = Column(String(2))  # total orbital angular momentum
    J = Column(Float)  # total angular momentum
    spin_multiplicity = Column(Integer)  # 2*S + 1, where S is total spin
    parity = Column(Integer)  # 0 - even, 1 - odd
    # ToDo I think that term column can be derived from L, S, parity and configuration
    term = Column(String(20))

    energies = relationship("LevelEnergy", back_populates="level")
    ion = relationship("Ion", back_populates="levels")
    data_source = relationship("DataSource", backref="levels")

    __table_args__ = (UniqueConstraint('id', 'ion_id', 'data_source_id'),)

    __mapper_args__ = {
        'polymorphic_identity': 'level',
        'polymorphic_on': type
    }


class ChiantiLevel(Level):
    __tablename__ = "chianti_level"

    id = Column(Integer, ForeignKey('level.id'), primary_key=True)
    ch_index = Column(Integer)
    ch_label = Column(String(10))

    __mapper_args__ = {
        'polymorphic_identity': 'chianti'
    }


class LevelEnergy(QuantityMixin, Base):
    __tablename__ = "level_energy"

    level_id = Column(Integer, ForeignKey('level.id'), nullable=False)
    unit = u.eV
    equivalencies = u.spectral()

    level = relationship("Level", back_populates="energies")


class Transition(Base):
    __tablename__ = "transition"

    id = Column(Integer, primary_key=True)
    type = Column(String(50))

    source_level_id = Column(Integer, ForeignKey('level.id'), nullable=False)
    target_level_id = Column(Integer, ForeignKey('level.id'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.id'), nullable=False)

    source_level = relationship("Level",
                                primaryjoin=(Level.id == source_level_id))
    target_level = relationship("Level",
                                primaryjoin=(Level.id == target_level_id))


    data_source = relationship("DataSource", backref="transitions")

    __mapper_args__ = {
        'polymorphic_identity': 'transition',
        'polymorphic_on': type
    }


class Line(Transition):
    __tablename__ = "line"

    id = Column(Integer, ForeignKey('transition.id'), primary_key=True)

    wavelengths = relationship("LineWavelength", backref="line")
    a_values = relationship("LineAValue", backref="line")
    gf_values = relationship("LineGFValue", backref="line")

    __mapper_args__ = {
        'polymorphic_identity': 'line'
    }


class LineQuantity(QuantityMixin, Base):
    __tablename__ = "line_quantity"

    line_id = Column(Integer, ForeignKey("line.id"))
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'line_qty',
        'polymorphic_on': type
    }


class LineWavelength(LineQuantity):

    unit = u.Angstrom

    __mapper_args__ = {
        'polymorphic_identity': 'wavelength'
    }


class LineAValue(LineQuantity):

    unit = u.Unit("s-1")

    __mapper_args__ = {
        'polymorphic_identity': 'a_value'
    }


class LineGFValue(LineQuantity):

    unit = u.dimensionless_unscaled

    __mapper_args__ = {
        'polymorphic_identity': 'gf_value'
    }


class ECollision(Transition):
    __tablename__ = "e_collision"

    id = Column(Integer, ForeignKey('transition.id'), primary_key=True)
    bt92_ttype = Column(Integer)  # BT92 Transition type
    bt92_cups = Column(Float)  # BT92 Scaling parameter

    energies = relationship("ECollisionEnergy",
                            backref="e_collision")
    gf_values = relationship("ECollisionGFValue",
                            backref="e_collision")
    temp_strengths = relationship("ECollisionTempStrength",
                            backref="e_collision")

    temp_strengths_tuple = association_proxy("temp_strengths", "as_tuple")

    __mapper_args__ = {
        'polymorphic_identity': 'e_collision'
    }


class ECollisionQuantity(QuantityMixin, Base):
    __tablename__ = "e_collision_qty"

    e_collision_id = Column(Integer, ForeignKey("e_collision.id"))
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'e_collision_qty',
        'polymorphic_on': type
    }


class ECollisionEnergy(ECollisionQuantity):

    unit = u.eV

    __mapper_args__ = {
        'polymorphic_identity': 'energy'
    }


class ECollisionTemp(ECollisionQuantity):

    unit = u.dimensionless_unscaled

    __mapper_args__ = {
        'polymorphic_identity': 'temp'
    }


class ECollisionStrength(ECollisionQuantity):  # Scaled effective collision strengths

    unit = u.dimensionless_unscaled

    __mapper_args__ = {
        'polymorphic_identity': 'strength'
    }


class ECollisionGFValue(ECollisionQuantity):

    unit = u.dimensionless_unscaled

    __mapper_args__ = {
        'polymorphic_identity': 'gf_value'
    }


class ECollisionTempStrength(Base):
    __tablename__ = "e_collision_temp_strength"

    temp_id = Column(Integer, ForeignKey("e_collision_qty.id"), primary_key=True)
    strength_id = Column(Integer, ForeignKey("e_collision_qty.id"), primary_key=True)
    e_collision_id = Column(Integer, ForeignKey("e_collision.id"))

    temp = relationship("ECollisionTemp", foreign_keys=[temp_id])
    strength = relationship("ECollisionStrength", foreign_keys=[strength_id])

    def __init__(self, tup):
        self.temp, self.strength = tup

    @property
    def as_tuple(self):
        return self.temp, self.strength


class DataSource(UniqueMixin, Base):
    __tablename__ = "data_source"

    @classmethod
    def unique_hash(cls, short_name, *args, **kwargs):
        return short_name

    @classmethod
    def unique_filter(cls, query, short_name, *args, **kwargs):
        return query.filter(DataSource.short_name == short_name)

    id = Column(Integer, primary_key=True)
    short_name = Column(String(20), unique=True, nullable=False)
    name = Column(String(120))
    description = Column(String(800))
    data_source_quality = Column(Integer)

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)