from .meta import Base, UniqueMixin, QuantityMixin, DataSourceMixin

from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey,\
    UniqueConstraint, ForeignKeyConstraint, and_, cast
from sqlalchemy.ext.associationproxy import association_proxy
from astropy import units as u


class Atom(Base):
    __tablename__ = "atom"
    atomic_number = Column(Integer, primary_key=True)
    symbol = Column(String(5), nullable=False)
    name = Column(String(150))
    group = Column(Integer)
    period = Column(Integer)

    weights = relationship("AtomWeight", back_populates='atom')
    ions = relationship("Ion", back_populates='atom')

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)


class AtomQuantity(QuantityMixin, Base):
    __tablename__ = "atom_quantity"

    atom_qty_id = Column(Integer, primary_key=True)
    atomic_number= Column(Integer, ForeignKey("atom.atomic_number"), nullable=False)
    type = Column(String(20))

    # __table_args__ = (UniqueConstraint('data_source_id', 'atomic_number', 'type', 'method'),)
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class AtomWeight(AtomQuantity):

    unit = u.u
    atom = relationship("Atom", back_populates='weights')

    __mapper_args__ = {
        'polymorphic_identity': 'weight'
    }


class Ion(UniqueMixin, Base):
    __tablename__ = "ion"

    @classmethod
    def unique_hash(cls, atomic_number, ion_charge, *args, **kwargs):
        return "ion:{0},{1}".format(atomic_number, ion_charge)

    @classmethod
    def unique_filter(cls, query, atomic_number, ion_charge, *args, **kwargs):
        return query.filter(and_(Ion.atomic_number == atomic_number,
                                 Ion.ion_charge == ion_charge))

    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), primary_key=True)
    ion_charge = Column(Integer, primary_key=True)

    ionization_energies = relationship("IonizationEnergy",
                                       back_populates='ion')
    levels = relationship("Level", back_populates="ion")
    ground_levels = relationship("Level", primaryjoin="and_(Ion.atomic_number==Level.atomic_number,"
                                                           "Ion.ion_charge==Level.ion_charge,"
                                                           "Level.ground)")
    atom = relationship("Atom", back_populates='ions')

    def __repr__(self):
        return "<Ion Z={0} +{1}>".format(self.atomic_number, self.ion_charge)


class IonQuantity(QuantityMixin, Base):
    __tablename__ = "ion_quantity"

    ion_qty_id = Column(Integer, primary_key=True)
    atomic_number= Column(Integer, nullable=False)
    ion_charge = Column(Integer, nullable=False)
    type = Column(String(20))

    __table_args__ = (ForeignKeyConstraint(['atomic_number', 'ion_charge'],
                                           ['ion.atomic_number', 'ion.ion_charge']),)
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class IonizationEnergy(IonQuantity):

    unit = u.eV
    ion = relationship("Ion", back_populates='ionization_energies')

    __mapper_args__ = {
        'polymorphic_identity': 'weight'
    }


class Level(Base):
    __tablename__ = "level"

    level_id = Column(Integer, primary_key=True)

    # Ion CFK
    atomic_number = Column(Integer, nullable=False)
    ion_charge = Column(Integer, nullable=False)

    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)
    level_index = Column(Integer)  # Index of this level from its data source
    configuration = Column(String(50))
    L = Column(String(2))  # total orbital angular momentum
    J = Column(Float)  # total angular momentum
    spin_multiplicity = Column(Integer)  # 2*S+1, where S is total spin
    parity = Column(Integer)  # 0 - even, 1 - odd
    # ToDo I think that term column can be derived from L, S, parity and configuration
    term = Column(String(20))
    ground = Column(Boolean, default=False)

    energies = relationship("LevelEnergy", back_populates="level")
    ion = relationship("Ion", back_populates="levels")
    data_source = relationship("DataSource", back_populates="levels")

    @hybrid_property
    def g(self):
        return int(2 * self.J + 1)

    @g.expression
    def g(cls):
        return cast(2 * cls.J + 1, Integer)

    __table_args__ = (ForeignKeyConstraint(['atomic_number', 'ion_charge'],
                                           ['ion.atomic_number', 'ion.ion_charge']),)


class LevelQuantity(QuantityMixin, Base):
    __tablename__ = "level_quantity"

    level_qty_id = Column(Integer, primary_key=True)
    level_id = Column(Integer, ForeignKey('level.level_id'), nullable=False)
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class LevelEnergy(LevelQuantity):

    unit = u.eV
    equivalencies = u.spectral()

    level = relationship("Level", back_populates="energies")

    __mapper_args__ = {
        'polymorphic_identity': 'energy'
    }


class Transition(Base):
    __tablename__ = "transition"

    transition_id = Column(Integer, primary_key=True)
    type = Column(String(50))

    lower_level_id = Column(Integer, ForeignKey('level.level_id'), nullable=False)
    upper_level_id = Column(Integer, ForeignKey('level.level_id'), nullable=False)
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)

    lower_level = relationship("Level", foreign_keys=[lower_level_id])
    upper_level = relationship("Level", foreign_keys=[upper_level_id])

    data_source = relationship("DataSource", back_populates="transitions")

    __mapper_args__ = {
        'polymorphic_identity': 'transition',
        'polymorphic_on': type
    }


class Line(Transition):
    __tablename__ = "line"

    line_id = Column(Integer, ForeignKey('transition.transition_id'), primary_key=True)

    wavelengths = relationship("LineWavelength", back_populates="line")
    a_values = relationship("LineAValue", back_populates="line")
    gf_values = relationship("LineGFValue", back_populates="line")

    __mapper_args__ = {
        'polymorphic_identity': 'line'
    }


class LineQuantity(QuantityMixin, Base):
    __tablename__ = "line_quantity"

    line_qty_id = Column(Integer, primary_key=True)
    line_id = Column(Integer, ForeignKey("line.line_id"))
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'line_qty',
        'polymorphic_on': type
    }


class LineWavelength(LineQuantity):

    unit = u.Angstrom

    line = relationship("Line", back_populates="wavelengths")

    __mapper_args__ = {
        'polymorphic_identity': 'wavelength'
    }


class LineAValue(LineQuantity):

    unit = u.Unit("s-1")

    line = relationship("Line", back_populates="a_values")

    __mapper_args__ = {
        'polymorphic_identity': 'a_value'
    }


class LineGFValue(LineQuantity):

    unit = u.dimensionless_unscaled

    line = relationship("Line", back_populates="gf_values")

    __mapper_args__ = {
        'polymorphic_identity': 'gf_value'
    }


class ECollision(Transition):
    __tablename__ = "e_collision"

    e_col_id = Column(Integer, ForeignKey('transition.transition_id'), primary_key=True)
    bt92_ttype = Column(Integer)  # BT92 Transition type
    bt92_cups = Column(Float)  # BT92 Scaling parameter

    energies = relationship("ECollisionEnergy", back_populates="e_collision")
    gf_values = relationship("ECollisionGFValue", back_populates="e_collision")
    temp_strengths = relationship("ECollisionTempStrength", back_populates="e_collision")

    temp_strengths_tuple = association_proxy("temp_strengths", "as_tuple")

    __mapper_args__ = {
        'polymorphic_identity': 'e_collision'
    }


class ECollisionQuantity(QuantityMixin, Base):
    __tablename__ = "e_collision_qty"

    e_col_qty_id = Column(Integer, primary_key=Transition)
    e_col_id = Column(Integer, ForeignKey("e_collision.e_col_id"))
    type = Column(String(20))

    __mapper_args__ = {
        'polymorphic_identity': 'e_collision_qty',
        'polymorphic_on': type
    }


class ECollisionEnergy(ECollisionQuantity):

    unit = u.eV

    e_collision = relationship("ECollision", back_populates="energies")

    __mapper_args__ = {
        'polymorphic_identity': 'energy'
    }


class ECollisionGFValue(ECollisionQuantity):

    unit = u.dimensionless_unscaled

    e_collision = relationship("ECollision", back_populates="gf_values")

    __mapper_args__ = {
        'polymorphic_identity': 'gf_value'
    }


class ECollisionTempStrength(Base):
    __tablename__ = "e_collision_temp_strength"

    e_col_temp_strength_id = Column(Integer, primary_key=True)
    temp = Column(Float)
    strength = Column(Float)
    e_col_id = Column(Integer, ForeignKey("e_collision.e_col_id"))

    e_collision = relationship("ECollision", back_populates="temp_strengths")

    @property
    def as_tuple(self):
        return self.temp, self.strength

    def __repr__(self):
        return "<Temp: {}, Strength: {}>".format(self.temp, self.strength)


class DataSource(UniqueMixin, Base):
    __tablename__ = "data_source"

    @classmethod
    def unique_hash(cls, short_name, *args, **kwargs):
        return short_name

    @classmethod
    def unique_filter(cls, query, short_name, *args, **kwargs):
        return query.filter(DataSource.short_name == short_name)

    data_source_id = Column(Integer, primary_key=True)
    short_name = Column(String(20), unique=True, nullable=False)
    name = Column(String(120))
    description = Column(String(800))
    data_source_quality = Column(Integer)

    levels = relationship("Level", back_populates="data_source")
    transitions = relationship("Transition", back_populates="data_source")

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)