"""Types and SQL constructs specific to carsus"""

import numpy as np

from sqlalchemy.sql.expression import ClauseElement
from sqlalchemy.orm.attributes import QueryableAttribute
from astropy.units import Quantity, Unit, dimensionless_unscaled, UnitsError
from astropy.units.quantity_helper import  UFUNC_HELPERS


class DBQuantity(Quantity):
    """
    Represents a number with some associated unit and
    is used as the public interface for `.quantity` in the database.

    Parameters:
    -----------
    value : sqlalchemy.sql.expression.ClauseElement instance,
            sqlalchemy.orm.attributes.QueryableAttibure instance,
            an object that could be passed to astropy.units.Quantity

    unit : astropy.units.UnitBase instance, str

    """
    _equivalencies = []
    __array_priority__ = 100000

    def __new__(cls, value, unit=None):

        if not isinstance(value, (QueryableAttribute, ClauseElement)):
            return Quantity.__new__(Quantity, value, unit=unit)
        unit = Unit(unit) if unit is not None else dimensionless_unscaled
        value = np.array(value)
        value = value.view(cls)
        value._unit = unit
        return value

    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.

        Parameters
        ----------
        unit : UnitBase
            The unit for which the appropriate class should be returned

        Returns
        -------
        tuple :
            - `Quantity` subclass
            - bool: True is subclasses of the given class are ok
        """
        return DBQuantity, True

    def _get_converter(self, function, other):
        other_unit = getattr(other, 'unit', None)

        converters, result_unit = UFUNC_HELPERS[function](function, self.unit, other_unit)

        # We are only interested in the converter for  `other`

        converter = converters[1]

        if converter is False:
            if other == 0:
                converter = None
            else:
                raise UnitsError("Can only apply '{0}' function to "
                                 "dimensionless quantity unless it is zero)"
                                 .format(function.__name__))
        return converter, result_unit

    #  Comparison ufuncs do not work with sqlalchemy constructs so we have to override them
    def __gt__(self, other):
        converter, _ = self._get_converter(np.greater, other)

        return (
            self.value > converter(other.value)
            if converter
            else self.value > getattr(other, 'value', other)
        )

    def __lt__(self, other):
        converter, _ = self._get_converter(np.less, other)

        return (
            self.value < converter(other.value)
            if converter
            else self.value < getattr(other, 'value', other)
        )

    def __eq__(self, other):
        converter, _ = self._get_converter(np.equal, other)

        if converter:
            res = self.value = converter(other.value)
        else:  # with no conversion, other can be non-Quantity
            res = self.value = getattr(other, 'value', other)

        return res

    #  We have to override this method and remove a type check!
    def __array_wrap__(self, obj, context=None):
        if context is None:
            # Methods like .squeeze() created a new `ndarray` and then call
            # __array_wrap__ to turn the array into self's subclass.
            return self._new_view(obj)

        # with context defined, we are continuing after a ufunc evaluation.
        if hasattr(obj, '_result_unit'):
            result_unit = obj._result_unit
            del obj._result_unit
        else:
            result_unit = None

            # We now need to re-calculate quantities for which the input
            # needed to be scaled.
        if hasattr(obj, '_converters'):

            converters = obj._converters
            del obj._converters

            # For in-place operations, input will get overwritten with
            # junk. To avoid that, we hid it in a new object in
            # __array_prepare__ and retrieve it here.
            if hasattr(obj, '_result'):
                obj = obj._result
            elif hasattr(obj, '_contiguous'):
                obj[()] = obj._contiguous
                del obj._contiguous

            # take array view to which output can be written without
            # getting back here
            obj_array = obj.view(np.ndarray)

            # Find out which ufunc was called and with which inputs
            function = context[0]
            args = context[1][:function.nin]

            # Set the inputs, rescaling as necessary
            inputs = []
            for arg, converter in zip(args, converters):
                if converter:
                    inputs.append(converter(arg.value))
                else:  # with no conversion, input can be non-Quantity.
                    inputs.append(getattr(arg, 'value', arg))

            # For output arrays that require scaling, we can reuse the
            # output array to perform the scaling in place, as long as the
            # array is not integral. Here, we set the obj_array to `None`
            # when it can not be used to store the scaled result.
            if result_unit is not None:  #
                obj_array = None

                # Re-compute the output using the ufunc
            if function.nin == 1:
                if function.nout == 1:
                    out = function(inputs[0], obj_array)
                elif context[2] == 0:
                    out, _ = function(inputs[0], obj_array, None)
                else:
                    _, out = function(inputs[0], None, obj_array)
            else:
                out = function(inputs[0], inputs[1], obj_array)

            if obj_array is None:
                obj = self._new_view(out, result_unit)

        if result_unit is None:  # return a plain array
            obj = obj.view(np.ndarray)
        else:
            obj._unit = result_unit

        return obj


    def to(self, unit, equivalencies=[]):
        """
        Returns a new `DBQuantity` object with the specified
        unit.

        """
        if equivalencies == []:
            equivalencies = self._equivalencies
        unit = Unit(unit)
        scale = self.unit.to(unit, equivalencies=equivalencies)
        new_val = np.asarray(self.value*scale)
        return self._new_view(new_val, unit)