import os
import logging
import carsus

logger = logging.getLogger(__name__)

basic_atomic_data_fname = os.path.join(
    carsus.__path__[0], "data", "basic_atomic_data.csv"
)
