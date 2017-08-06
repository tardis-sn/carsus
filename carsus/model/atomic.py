'''
DataBase Models in carsus
==========================

Introduction
~~~~~~~~~~~~~

`Carsus` uses sqlalchemy to link database rows to python objects. This makes it
very easy to query for a filtered subset of data.

classes define objects. each class is mapped to a database table and instances
of a class are mapped to rows of that table.  all classes inherit from Base
which is a declarative_base from sqlalchemy.  Each class has a "Primary Key"
which has to be unique for each object. Typically this is an integer starting
at 1. The primary key should be called 'id'.  Attributes of instances are
declared as special class attributes. To map an attribute to a database column
it has to be a Column object.  It is possible to define relationships with the
relationship function. This links two instances of an object together based on
one column, usually the primary key.

each class own column (share columns inheritance?)

operations mostly done on the database and not python -> fast

There are several types of classes describing the atomic data for us. First, we
have general models, like `Atom` and `Ion`. These are universal and independent
of the source of the data. They serve as anchors for datasource dependent
objects to be linked against.  Next is Data that is not universal but comes
from a source, for example IonizationEnergy. We are using data which is
provided by NIST but there are other sources aswell. To easily allow the same
data from different sources in the database, we link them to a source. This is
very important because when we extract the data, we always have to specify the
source of the data we want to extract.

In carsus we create This file contains all models linked to database tables.
'''

from sqlalchemy.orm import relationship
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, Float, ForeignKey,\
    ForeignKeyConstraint, and_, cast
from sqlalchemy.ext.associationproxy import association_proxy
from astropy import units as u
from carsus.model.meta import Base, UniqueMixin, QuantityMixin

# constants to differentiate the medium a wavelength is specified in.
# TODO: Maybe write as an enum
MEDIUM_VACUUM = 0
MEDIUM_AIR = 1

__all__ = [
        'Atom',
        'AtomQuantity',
        'AtomWeight',

        'Ion',
        'IonQuantity',
        'IonizationEnergy',

        'Level',
        'LevelQuantity',
        'LevelEnergy',

        'Transition',
        'Line',
        'LineQuantity',
        'LineWavelength',
        'LineAValue',
        'LineGFValue',

        'ECollision',
        'ECollisionQuantity',
        'ECollisionEnergy',
        'ECollisionGFValue',
        'ECollisionTempStrength',

        'DataSource',
        ]


class Atom(Base):
    '''
    Model describing a simple Atom.
    '''

    __tablename__ = "atom"
    atomic_number = Column(Integer, primary_key=True)
    '''Atomic number of the Atom'''

    symbol = Column(String(5), nullable=False)
    name = Column(String(150))
    group = Column(Integer)
    period = Column(Integer)

    weights = relationship("AtomWeight", back_populates='atom')
    ions = relationship("Ion", back_populates='atom')

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)


class AtomQuantity(QuantityMixin, Base):
    '''
    Base class for all quantities of an :py:class:`~carsus.atomic.Atom`. Mixes in the QuantityMixin to
    expose the auantity interface.
    '''

    __tablename__ = "atom_quantity"

    #: Primary Key
    atom_qty_id = Column(Integer, primary_key=True)

    #: ForeignKey linking a AtomQuantity to an Atom
    atomic_number = Column(
            Integer,
            ForeignKey("atom.atomic_number"), nullable=False)
    type = Column(String(20))

    # __table_args__ = (UniqueConstraint('data_source_id', 'atomic_number', 'type', 'method'),)
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class AtomWeight(AtomQuantity):
    '''
    Weight of an Atom in atomic units ['u'].
    '''

    unit = u.u
    atom = relationship("Atom", back_populates='weights')

    __mapper_args__ = {
        'polymorphic_identity': 'weight'
    }


class Ion(UniqueMixin, Base):
    '''
    Model describing an Ion. Inherits the UniqueMixin to guarantee no
    duplicates.
    '''
    __tablename__ = "ion"

    @classmethod
    def unique_hash(cls, atomic_number, ion_charge, *args, **kwargs):
        return "ion:{0},{1}".format(atomic_number, ion_charge)

    @classmethod
    def unique_filter(cls, query, atomic_number, ion_charge, *args, **kwargs):
        return query.filter(and_(Ion.atomic_number == atomic_number,
                                 Ion.ion_charge == ion_charge))

    #: ForeignKey linking an Ion to an Atom
    atomic_number = Column(
            Integer,
            ForeignKey('atom.atomic_number'), primary_key=True)
    #: Charge of the ion
    ion_charge = Column(Integer, primary_key=True)

    #: Relationship to IonizationEnergy
    ionization_energies = relationship("IonizationEnergy",
                                       back_populates='ion')
    #: Relationship to Level
    levels = relationship("Level", back_populates="ion")
    #: Relationship to Atom
    atom = relationship("Atom", back_populates='ions')

    def __repr__(self):
        return "<Ion Z={0} +{1}>".format(self.atomic_number, self.ion_charge)


class IonQuantity(QuantityMixin, Base):
    '''
    Base class for all quantities of an Ion. Mixes in the QuantityMixin to
    expose the auantity interface.
    '''
    __tablename__ = "ion_quantity"

    #: Primary Key
    ion_qty_id = Column(Integer, primary_key=True)
    #: ForeignKeyConstraint linking to an Ion
    atomic_number= Column(Integer, nullable=False)
    #: ForeignKeyConstraint linking to an Ion
    ion_charge = Column(Integer, nullable=False)
    type = Column(String(20))

    __table_args__ = (ForeignKeyConstraint(['atomic_number', 'ion_charge'],
                                           ['ion.atomic_number', 'ion.ion_charge']),)
    __mapper_args__ = {
        'polymorphic_on': type,
        'polymorphic_identity': 'qty'
    }


class IonizationEnergy(IonQuantity):
    '''
    Ionization energy of an Ion in electron volt [eV].
    '''

    unit = u.eV
    ion = relationship("Ion", back_populates='ionization_energies')

    __mapper_args__ = {
        'polymorphic_identity': 'weight'
    }


class Level(Base):
    '''
    Level of an Ion.
    '''
    __tablename__ = "level"

    #: Primary Key
    level_id = Column(Integer, primary_key=True)

    # Ion CFK
    #: ForeignKeyConstraint linking to an Ion
    atomic_number = Column(Integer, nullable=False)
    #: ForeignKeyConstraint linking to an Ion
    ion_charge = Column(Integer, nullable=False)

    #: Id of the datasource of this level
    data_source_id = Column(Integer, ForeignKey('data_source.data_source_id'), nullable=False)
    #: Index of this level from its data source
    level_index = Column(Integer)
    #: Configuration of the level
    configuration = Column(String(50))
    L = Column(String(2))  #: total orbital angular momentum
    J = Column(Float)  #: total angular momentum
    #: spin_multiplicity 2*S+1, where S is total spin
    spin_multiplicity = Column(Integer)
    parity = Column(Integer)  #: Parity 0 - even, 1 - odd
    # ToDo I think that term column can be derived from L, S, parity and configuration
    term = Column(String(20))

    energies = relationship("LevelEnergy", back_populates="level")
    ion = relationship("Ion", back_populates="levels")
    data_source = relationship("DataSource", backref="levels")

    @hybrid_property
    def g(self):
        return int(2 * self.J + 1)

    @g.expression
    def g(cls):
        return cast(2 * cls.J + 1, Integer).label('g')

    __table_args__ = (ForeignKeyConstraint(['atomic_number', 'ion_charge'],
                                           ['ion.atomic_number', 'ion.ion_charge']),)


class LevelQuantity(QuantityMixin, Base):
    '''
    Base class for all quantities of a level. Mixes in the QuantityMixin to
    expose the auantity interface.
    '''
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

    data_source = relationship("DataSource", backref="transitions")

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
    '''
    Base class for all quantities of a line. Mixes in the QuantityMixin to
    expose the auantity interface.
    '''
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

    medium = Column(Integer, default=MEDIUM_VACUUM)
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
    '''
    Base class for all quantities of an electron collision. Mixes in the
    QuantityMixin to expose the auantity interface.
    '''
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

    # levels = relationship("Level", back_populates="data_source")
    # transitions = relationship("Transition", back_populates="data_source")

    def __repr__(self):
        return "<Data Source: {}>".format(self.short_name)
