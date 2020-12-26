import os
import logging
import pandas as pd
import carsus
from carsus.model import Atom, setup

logger = logging.getLogger(__name__)

basic_atomic_data_fname = os.path.join(carsus.__path__[0], 'data',
                                       'basic_atomic_data.csv')


def init_db(db_url='sqlite://', **kwargs):
    """
    Initializes the database.
    If the database is empty ingests basic atomic data (atomic numbers, symbols, etc.)

    Parameters
    ----------
    db_url : str
        Url to the database file. Set to 'sqlite://' to create a memory session.
        Example: 'sqlite:///carsus.db'

    kwargs
        Additional keyword arguments that can be passed to the `create_engine` function (e.g. echo=True)

    Returns
    --------
    an instance of the sqlalchemy.orm.session.Session class

    """
    logger.info("Initializing the database at {}".format(db_url))

    session = setup(db_url, **kwargs)

    if session.query(Atom).count() == 0:
        _init_empty_db(session)

    return session


def _init_empty_db(session):
    """ Ingests basic atomic data to an empty database """
    basic_atomic_data = pd.read_csv(basic_atomic_data_fname)
    logger.info("Ingesting basic atomic data.")
    for i, row in basic_atomic_data.iterrows():
        session.add(Atom(atomic_number=row['atomic_number'], name=row['name'],
                          symbol=row['symbol'], group=row['group'],
                          period=row['period']))
