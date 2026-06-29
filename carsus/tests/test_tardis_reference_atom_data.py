import pandas as pd
import pytest


@pytest.mark.with_regression_data
def test_kurucz_cd23_chianti_h_he_latest_schema_contract(carsus_regression_path):
    reference_path = (
        carsus_regression_path
        / "atom_data"
        / "kurucz_cd23_chianti_H_He_latest.h5"
    )

    with pd.HDFStore(reference_path, mode="r") as store:
        assert set(store.keys()) == {
            "/atom_data",
            "/collisions_data",
            "/collisions_metadata",
            "/decay_radiation_data",
            "/ionization_data",
            "/levels_data",
            "/lines_data",
            "/macro_atom_data",
            "/macro_atom_references",
            "/metadata",
            "/zeta_data",
        }
        assert store.root._v_attrs["FORMAT_VERSION"] == "2.0"
        assert store.root._v_attrs["database_version"] == "v0.9"

        metadata = store["metadata"]
        assert ("md5sum", "levels") in metadata.index
        assert ("md5sum", "lines") in metadata.index
        assert ("md5sum", "levels_data") not in metadata.index
        assert ("md5sum", "lines_data") not in metadata.index

        atom_data = store["atom_data"]
        assert atom_data.index.name == "atomic_number"
        assert atom_data.index.min() == 1
        assert atom_data.index.max() == 30
