from sqlalchemy import Column, Integer, Float, String
from sqlalchemy.ext.hybrid import hybrid_property
from ..meta.types import DBQuantity
from astropy.units import dimensionless_unscaled, UnitsError, set_enabled_equivalencies


class QuantityMixin(object):

    id = Column(Integer, primary_key=True)

    _value = Column(Float, nullable=False)
    uncert = Column(Float)
    method = Column(String(15))
    reference = Column(String(50))

    unit = dimensionless_unscaled
    equivalencies = None

    # Public interface for value is via `.quantity` accessor
    @hybrid_property
    def quantity(self):
        return DBQuantity(self._value, self.unit)

    @quantity.setter
    def quantity(self, qty):
        try:
            with set_enabled_equivalencies(self.equivalencies):
                self._value = qty.to(self.unit).value
        except AttributeError:
            if self.unit is dimensionless_unscaled or qty == 0:
                self._value = qty
            else:
                raise UnitsError("Can only assign dimensionless values "
                                 "to dimensionless quantities "
                                 "(unless the value is 0)")

    def __repr__(self):
        return "<Quantity: {0} {1}>".format(self._value, self.unit)