"""Fundamental units like declarative_base"""

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine

Base = declarative_base()
'''
Base class for all models mapping to a database with sqlalchemy.
'''


def setup(url, **kwargs):
    """Creates a configured "Session" class and returns its instance"""

    engine = create_engine(url, **kwargs)
    Base.metadata.create_all(engine)
    return Session(bind=engine)


