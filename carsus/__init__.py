# Licensed under a 3-clause BSD style license - see LICENSE.rst
import sys

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
if ("astropy.units" in sys.modules) or ("astropy.constants" in sys.modules):
    import warnings

    warnings.warn(
        "Astropy is already imported externally. Astropy should be imported "
        "after Carsus so Carsus can pin its constants version."
    )
else:
    from astropy import astronomical_constants, physical_constants

    physical_constants.set("codata2010")
    astronomical_constants.set("iau2012")

from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = []

import logging
from .util.colored_logger import (
    ColoredFormatter,
    formatter_message,
)

FORMAT = "[$BOLD%(name)27s$RESET][%(levelname)18s] - %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
COLOR_FORMAT = formatter_message(FORMAT, True)


logging.captureWarnings(True)
logger = logging.getLogger('carsus')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_formatter = ColoredFormatter(COLOR_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logging.getLogger('py.warnings').addHandler(console_handler)

# Set atomic file format version
FORMAT_VERSION = "2.0"
