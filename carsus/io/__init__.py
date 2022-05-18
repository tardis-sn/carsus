import os
import logging
from carsus.io.nist import NISTIonizationEnergiesParser, NISTIonizationEnergiesIngester,\
    NISTWeightsCompPyparser, NISTWeightsCompIngester
from carsus.io.kurucz import GFALLReader, GFALLIngester
from carsus.io.output import AtomData
from carsus.io.zeta import KnoxLongZetaIngester

logger = logging.getLogger(__name__)    

if "XUVTOP" in os.environ:
    from carsus.io.chianti_ import ChiantiIonReader, ChiantiIngester
else:
    logger.warning(
        "XUVTOP environment variable is not set in your shell configuration file. "
        "The XUVTOP environment variable is required if you want to use the Chianti submodule."
    )