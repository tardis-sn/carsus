from pandas import read_sql_query
from carsus.model import Atom, AtomWeight, DataSource


def read_basic_atom_data(session):
    """
        This function reads the atomic number, symbol, and mass from

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        basic_atom_data_df : pandas.DataFrame
           DataFrame with columns z[1], symbol, mass[u]
    """
    q_basic_atom_data = session.query(
        Atom.atomic_number.label("z"),
        Atom.symbol.label("symbol"),
        AtomWeight.quantity.value.label("mass")).\
        outerjoin(Atom.weights)

    basic_atom_data_df = read_sql_query(q_basic_atom_data.selectable, session.bind)
    return basic_atom_data_df
