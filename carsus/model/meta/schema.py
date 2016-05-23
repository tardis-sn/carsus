from sqlalchemy import Column, Integer, Float, ForeignKey
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import relationship
from ..meta.types import DBQuantity


class QuantityMixin(object):

    id = Column(Integer, primary_key=True)

    _value = Column(Float, nullable=False)
    std_dev = Column(Float)

    unit = None

    # Public interface for value is via `.quantity` accessor
    @hybrid_property
    def quantity(self):
        return DBQuantity(self._value, self.unit)

    @quantity.setter
    def quantity(self, qty):
        self._value = qty.to(self.unit).value

    @declared_attr
    def data_source_id(cls):
        return Column(Integer, ForeignKey('data_source.id'), nullable=False)

    @declared_attr
    def data_source(cls):
        return relationship("DataSource")

    def __repr__(self):
        return "<Quantity: {0} {1}>".format(self._value, self.unit)