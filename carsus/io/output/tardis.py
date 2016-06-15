from pandas import read_sql_query
from carsus.model import Atom, AtomWeight, Ion, IonizationEnergy


def create_basic_atom_data(session):
    """
        This function reads the atomic number, symbol, and atomic weight from the database

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        basic_atom_data_df : pandas.DataFrame
           DataFrame with columns atomic_number[1], symbol, name, weight[u]
    """
    q_basic_atom_data = session.query(
        Atom.atomic_number.label("atomic_number"),
        Atom.symbol.label("symbol"),
        Atom.name.label("name"),
        AtomWeight.quantity.value.label("weight")).\
        outerjoin(Atom.weights)

    basic_atom_data_df = read_sql_query(q_basic_atom_data.selectable, session.bind)
    return basic_atom_data_df


def create_ionization_data(session):
    """
        This function reads the atomic number, ion number, ionization energy from the database

        Parameters
        ----------
        session : SQLAlchemy session

        Returns
        -------
        ionization_data_df : pandas.DataFrame
           DataFrame with columns atomic_number[1], ion_number[1], ionization_energy[eV]
    """
    q_ionization_data = session.query(
        Ion.atomic_number.label("atomic_number"),
        (Ion.ion_charge + 1).label("ion_number"),
        IonizationEnergy.quantity.value.label("ionization_energy").\
        outerjoin(Ion.ionization_energies)
    )

    ionization_data_df = read_sql_query(q_ionization_data.selectable, session.bind)
    return ionization_data_df