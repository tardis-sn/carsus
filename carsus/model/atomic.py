from .meta import Base, UniqueMixin, DBQuantity, Decimal10, QuantityMixin
import re
from sqlalchemy.orm import relationship
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy import Column, Integer, String, Float, ForeignKey,\
    UniqueConstraint, ForeignKeyConstraint, and_, cast, case
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
        return "ion:{0},{1}".format(atomic_number, ion_charge)

    @classmethod
    def unique_filter(cls, query, atomic_number, ion_charge, *args, **kwargs):
        return query.filter(and_(Ion.atomic_number == atomic_number,
                                 Ion.ion_charge == ion_charge))

    id = Column(Integer, primary_key=True)
    atomic_number = Column(Integer, ForeignKey('atom.atomic_number'), nullable=False)
    ion_charge = Column(Integer, nullable=False)

    # ground_level_id = Column(Integer, ForeignKey('level.id'))

    # ionization_energies = relationship("IonizationEnergy",
    #                                    backref='ion',
    #                                    cascade="all, delete-orphan")

    levels = relationship("Level",
                          backref='ion',
                          cascade="all, delete-orphan")

    atom = relationship("Atom", backref='ions')

    __table_args__ = (UniqueConstraint('atomic_number', 'ion_charge'),)

    def __repr__(self):
        return "<Ion {0} +{1}>".format(self.atom.symbol, self.ion_charge)


class Level(UniqueMixin, Base):
    __tablename__ = "level"

    id = Column(Integer, primary_key=True)
    ion_id = Column(Integer, ForeignKey('ion.id'), nullable=False)
    configuration = Column(String(50))
    L = Column(String(2))
    J = Column(Decimal10)
    parity = Column(Integer)
    spin_mult = Column(Integer)  # 2*S + 1

    @classmethod
    def unique_hash(cls, configuration, term, *args, **kwargs):
        return "lvl:{0},{1}".format(configuration, term)

    @classmethod
    def unique_filter(cls, query, ion, configuration, term, J, *args, **kwargs):
        return query.filter(and_(Level.ion == ion,
                                 Level.configuration == configuration,
                                 Level.term == term,
                                 Level.J == J))

    @hybrid_property
    def term(self):
        if self.parity == 0:
            return str(self.spin_mult) + self.L
        else:
            return str(self.spin_mult) + self.L + '*'

    @term.setter
    def term(self, value):
        value_dict = re.match(r"(?P<spin_mult>\d+)(?P<L>[A-Z]+)(?P<parity>[*])?", value).groupdict()
        self.spin_mult = value_dict["spin_mult"]
        self.L = value_dict["L"]
        self.parity = 1 if value_dict["parity"] else 0


    @term.expression
    def term(cls):
        return case([
            (cls.parity == 0, cast(cls.spin_mult, String) + cls.L),
        ], else_=(cast(cls.spin_mult, String) + cls.L + '*'))

    @hybrid_property
    def g(self):
        return int(2*self.J + 1)

    @g.expression
    def g(cls):
        return (2*cast(cls.J, Integer) + 10)/10

    __table_args__ = (
        UniqueConstraint('ion_id', 'configuration', 'L', 'J', 'spin_mult'),
    )


class LevelData(Base):
    __tablename__ = "level_data"

    id = Column(Integer, primary_key=True)
    level_id = Column(Integer, ForeignKey('level.id'))
    data_source_id = Column(Integer, ForeignKey('data_source.id'))
    type = Column(String(20))

    energies = relationship("LevelEnergy")
    level = relationship("Level", backref="level_data")
    data_source = relationship("DataSource", backref="level_data")

    __mapper_args__ = {
        'polymorphic_identity': 'data',
        'polymorphic_on': type
    }


class LevelChiantiData(LevelData):
    __tablename__ = "level_chianti_data"

    id = Column(Integer, ForeignKey('level_data.id'), primary_key=True)
    ch_index = Column(Integer)
    ch_label = Column(String(10))

    __mapper_args__ = {
        'polymorphic_identity': 'chianti'
    }


class LevelEnergy(QuantityMixin, Base):
    __tablename__ = "level_energy"

    level_id = Column(Integer)
    data_source_id = Column(Integer)

    unit = u.eV

    __table_args__ = (
        ForeignKeyConstraint(['level_id', 'data_source_id'],
            ['level_data.level_id', 'level_data.data_source_id']),)

# UniqueConstraint('level_id', 'data_source_id', 'method')

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