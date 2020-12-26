# Licensed under a 3-clause BSD style license - see LICENSE.rst

# Packages may add whatever they like to this file, but
# should keep this content at the top.
# ----------------------------------------------------------------------------
from ._astropy_init import *   # noqa
# ----------------------------------------------------------------------------

__all__ = []


import sys
import logging
from .base import init_db
from .util.colored_logger import ColoredFormatter, formatter_message

FORMAT = "[$BOLD%(name)27s$RESET][%(levelname)8s] - %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
COLOR_FORMAT = formatter_message(FORMAT, True)


logging.captureWarnings(True)
logger = logging.getLogger('carsus')
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler(sys.stdout)
console_formatter = ColoredFormatter(COLOR_FORMAT)
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)
logging.getLogger('py.warnings').addHandler(console_handler)
