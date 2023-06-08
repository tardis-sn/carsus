import os

from carsus import init_db
from carsus.io import (
        NISTWeightsCompIngester,
        NISTIonizationEnergiesIngester,
        GFALLIngester,
        ChiantiIngester,
        KnoxLongZetaIngester
        )
import warnings

DATA_DIR = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'data')
TEST_DB_FNAME = os.path.join(DATA_DIR, 'test.db')
GFALL_FNAME = os.path.join(DATA_DIR, "gftest.all")
ZETA_FNAME = os.path.join(DATA_DIR, "knox_long_recombination_zeta.dat")


def create_test_db(
        test_db_fname=TEST_DB_FNAME,
        gfall_fname=GFALL_FNAME,
        zeta_fname=ZETA_FNAME
        ):
    """
    Create a database for testing

    Parameters
    ----------
    test_db_fname : str
        Filename for the testing database
    gfall_fname : str
        Filename for the GFALL file
    """

    test_db_f = open(test_db_fname, "w")
    test_db_f.close()

    session = init_db(f'sqlite:///{test_db_fname}')
    session.commit()

    # Ingest atomic weights
    weightscomp_ingester = NISTWeightsCompIngester(session)
    weightscomp_ingester.download()
    weightscomp_ingester.ingest()
    session.commit()

    # Ingest ionization energies
    ioniz_energies_ingester = NISTIonizationEnergiesIngester(
            session,
            spectra='h-zn'
            )
    ioniz_energies_ingester.ingest(ionization_energies=True, ground_levels=True)
    session.commit()

    # Ingest zeta data
    zeta_ingester = KnoxLongZetaIngester(
            session,
            zeta_fname
            )
    zeta_ingester.ingest()
    session.commit()

    # Ingest kurucz levels and lines
    gfall_ingester = GFALLIngester(session, gfall_fname)
    gfall_ingester.ingest(levels=True, lines=True)
    session.commit()

    # Ingest chianti levels, lines and electron collisions
    chianti_ingester = ChiantiIngester(session, ions="he_2;n_6")
    chianti_ingester.ingest(levels=True, lines=True, collisions=True)
    session.commit()

    session.close()


if __name__ == "__main__":
    warnings.warn(
            'Using this script with chianti_v8.0.6 and the gftest.dat file in'
            'the repository does not create a database where the tests pass.'
            'It looks like the data for silicon is missing from kurucz (in'
            'gftest.dat) and there seem to be some changes to the levels'
            'and lines of nitrogen in chianti.'
            )
    create_test_db()
