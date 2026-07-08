import pandas as pd

from carsus.io.output import TARDISAtomData


class ReaderStub:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def make_atom_data_for_hdf():
    atom_data = object.__new__(TARDISAtomData)

    atom_data.atomic_weights = ReaderStub(
        version="weights-version",
        base=pd.DataFrame(
            {
                "symbol": ["H"],
                "name": ["Hydrogen"],
                "mass": [1.008],
            },
            index=pd.Index([1], name="atomic_number"),
        ),
    )
    atom_data.ionization_energies_preparer = ReaderStub(
        ionization_energies=ReaderStub(version="spectra-version")
    )
    atom_data.gfall_reader = ReaderStub(version="gfall-version")
    atom_data.zeta_data = ReaderStub(
        version="zeta-version",
        base=pd.DataFrame(
            {2000.0: [1.0]},
            index=pd.MultiIndex.from_tuples(
                [(1, 0)], names=["atomic_number", "ion_charge"]
            ),
        ),
    )
    atom_data.chianti_reader = ReaderStub(version="chianti-version")
    atom_data.cmfgen_reader = None
    atom_data.vald_reader = None
    atom_data.barklem_2016_data = None
    atom_data.cross_sections_preparer = None
    atom_data.collisions_preparer = None
    atom_data.nndc_reader = ReaderStub(
        decay_data=pd.DataFrame(
            {"Z": [28], "Radiation": ["g"]},
            index=pd.Index(["Ni56"], name="Isotope"),
        )
    )

    atom_data.levels_lines_preparer = ReaderStub(
        levels_prepared=pd.DataFrame(
            {
                "energy": [0.0],
                "g": [2],
                "metastable": [False],
            },
            index=pd.MultiIndex.from_tuples(
                [(1, 0, 0)],
                names=["atomic_number", "ion_number", "level_number"],
            ),
        ),
        lines_prepared=pd.DataFrame(
            {
                "line_id": [0],
                "wavelength": [1215.67],
                "f_ul": [0.5],
                "f_lu": [1.0],
                "nu": [2.466e15],
                "B_lu": [1.0],
                "B_ul": [0.5],
                "A_ul": [6.25e8],
            },
            index=pd.MultiIndex.from_tuples(
                [(1, 0, 0, 1)],
                names=[
                    "atomic_number",
                    "ion_number",
                    "level_number_lower",
                    "level_number_upper",
                ],
            ),
        ),
    )
    atom_data.macro_atom_preparer = ReaderStub(
        macro_atom_prepared=pd.DataFrame(
            {
                "atomic_number": [1],
                "ion_number": [0],
                "source_level_number": [0],
                "destination_level_number": [1],
                "transition_type": [1],
                "transition_probability": [1.0],
                "transition_line_id": [0],
            }
        ),
        macro_atom_references_prepared=pd.DataFrame(
            {
                "count_down": [0],
                "count_up": [1],
                "count_total": [1],
            },
            index=pd.MultiIndex.from_tuples(
                [(1, 0, 0)],
                names=["atomic_number", "ion_number", "source_level_number"],
            ),
        ),
    )

    atom_data.ionization_energies_preparer.ionization_energies_prepared = pd.Series(
        [13.598434],
        index=pd.MultiIndex.from_tuples(
            [(1, 1)], names=["atomic_number", "ion_number"]
        ),
    )

    return atom_data


def test_to_hdf_writes_modern_metadata(tmp_path):
    atom_data = make_atom_data_for_hdf()
    output_path = tmp_path / "atom_data.h5"

    atom_data.to_hdf(output_path)

    with pd.HDFStore(output_path, mode="r") as store:
        keys = set(store.keys())
        assert "/lines_metadata" in keys
        assert "/decay_radiation_data" in keys

        metadata = store["metadata"]
        assert ("datasets", "nist_weights") in metadata.index
        assert ("md5sum", "levels_data") in metadata.index
        assert ("md5sum", "lines_data") in metadata.index
        assert not hasattr(store.root._v_attrs, "database_version")


def test_to_hdf_preserves_supplied_atomic_weights_subset(tmp_path):
    atom_data = make_atom_data_for_hdf()
    output_path = tmp_path / "atom_data.h5"

    atom_data.to_hdf(output_path)

    with pd.HDFStore(output_path, mode="r") as store:
        atom_data_output = store["atom_data"]
        assert list(atom_data_output.index) == [1]
        assert atom_data_output.index.name == "atomic_number"


def test_to_hdf_can_write_legacy_tardis_schema_metadata(tmp_path):
    atom_data = make_atom_data_for_hdf()
    output_path = tmp_path / "atom_data.h5"

    atom_data.to_hdf(
        output_path,
        legacy_tardis_schema=True,
        database_version="v0.9",
    )

    with pd.HDFStore(output_path, mode="r") as store:
        keys = set(store.keys())
        assert "/levels_data" in keys
        assert "/lines_data" in keys
        assert "/lines_metadata" not in keys
        assert store.root._v_attrs["database_version"] == "v0.9"

        metadata = store["metadata"]
        assert ("datasets", "nist_weights") not in metadata.index
        assert ("md5sum", "levels") in metadata.index
        assert ("md5sum", "lines") in metadata.index
        assert ("md5sum", "levels_data") not in metadata.index
        assert ("md5sum", "lines_data") not in metadata.index
