'''
Introduction
============

Carsus uses `sqlalchemy` to associate (experimental) data about real objects,
such as atoms, with tables in a database. This allows us to store all data in
one big database, but we can use it as if it were simple Python objects.
Additionally, operations like filtering the data are performed on the database
instead of in Python which is a lot better for performance.

At the core of this system are the database models. These are Python classes
with special class attributes that are mapped to database columns by
`sqlalchemy`. In the database, each class has its own table and instances of the
class represent one specific row of that table. All models have to inherit
from `Base` which is defined in `carsus.model.meta`. Each model has a "primary
key" which has to be unique for each object and is used to identify it.
Typically this is an integer but it is also possible to use a combination of
multiple values to form the primary key (see `IonQuantity` for example). If the
primary key is a single integer, it should be called ``id``.

Attributes of instances are declared as instances of `sqlalchemy.Column` 
which is a special class attribute pointing to a column in a table. 
Relationships between models are defined with `sqlalchemy.orm.relationship`
linking two instances of an object together where usually one column points
to the primary key of another table. Defining the relationships is important 
so `sqlalchemy` can automatically join the models together if a join operation
is added to the query.

We have several types of models for the atomic data. First, we have general
models, like :class:`~carsus.model.atomic.Atom` and
:class:`~carsus.model.atomic.Ion`. These are universal and independent of the
source of the data. They serve as anchors for datasource dependent quantities
to be linked against. These are not universal, like for example the
:class:`~carsus.model.atomic.IonizationEnergy`, but come from sources such as
NIST. To easily allow the data from different sources for the same quantity in
the database, they are linked to a source. This is very important because when
extracting the data, we always have to specify the source of the data we want
to extract.

Classes
========
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

    # weights = relationship("AtomWeight", back_populates='atom')
    # ions = relationship("Ion", back_populates='atom')

    def __repr__(self):
        return "<Atom {0}, Z={1}>".format(self.symbol, self.atomic_number)


class AtomQuantity(QuantityMixin, Base):
    '''
    Base class for all quantities of an :class:`~carsus.model.atomic.Atom`. Mixes in the QuantityMixin to
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
    atom = relationship("Atom", backref='weights')

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
    atom = relationship("Atom", backref='ions')

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
    foo
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


class Zeta(Base):
    __tablename__ = 'zeta'

    id = Column(Integer, primary_key=True)

    # Ion FKC
    atomic_number = Column(Integer, nullable=False)
    ion_charge = Column(Integer, nullable=False)

    zeta = Column(Float)

    temp_id = Column(Integer, ForeignKey('temperature.id'), nullable=False)
    data_source_id = Column(
            Integer,
            ForeignKey('data_source.data_source_id'),
            nullable=False)

    temp = relationship('Temperature')
    data_source = relationship("DataSource", backref="zeta_data")

    __table_args__ = (ForeignKeyConstraint(['atomic_number', 'ion_charge'],
                                           ['ion.atomic_number', 'ion.ion_charge']),)


class Temperature(UniqueMixin, Base):
    __tablename__ = 'temperature'

    id = Column(Integer, primary_key=True)

    value = Column(Float)  # Temperature in Kelvin

    @classmethod
    def unique_hash(cls, value, *args, **kwargs):
        return "temperature:{0}".format(value)

    @classmethod
    def unique_filter(cls, query, value, *args, **kwargs):
        return query.filter(Temperature.value == value)

    def __repr__(self):
        return "<Temperature {0} K>".format(self.value)
