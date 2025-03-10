import pytest

from numpy.testing import assert_allclose
from carsus.io.molecules.molecules import BarklemCollet2016Reader


@pytest.fixture(scope="package")
def barklem_rdr():
    rdr = BarklemCollet2016Reader()
    rdr.parse_barklem_2016()
    return rdr


@pytest.fixture(scope="module")
def barklem_raw(barklem_rdr):
    return barklem_rdr.barklem_2016_raw


@pytest.fixture(scope="module")
def barklem_dissociation_energies(barklem_rdr):
    return barklem_rdr.dissociation_energies


@pytest.fixture(scope="module")
def barklem_ionization_energies(barklem_rdr):
    return barklem_rdr.ionization_energies


@pytest.fixture(scope="module")
def barklem_partition_functions(barklem_rdr):
    return barklem_rdr.partition_functions


@pytest.fixture(scope="module")
def barklem_equilibrium_constants(barklem_rdr):
    return barklem_rdr.equilibrium_constants


def test_checksums(barklem_rdr):
    assert barklem_rdr.dissociation_version == "18b419294f4f3fd8de5d3b6ad5135286"
    assert barklem_rdr.ionization_version == "2e4194bf80cb4cc0787a933b62cafc31"
    assert barklem_rdr.partition_version == "25ab2bb785f90f18eb1915fe429abbb9"
    assert barklem_rdr.equilibrium_version == "e3362edb9422807eceb4ff0a5f57fc2b"


def test_barklem_raw(barklem_raw):
    assert len(barklem_raw) == 8
    # sourcery skip: no-loop-in-tests
    for df in barklem_raw[:4]:
        assert not df.empty, "DataFrame is empty"


@pytest.mark.parametrize(
    "atomic_Number,element,first_ionization_energy",
    [
        (1, "H", 13.5984),
        (2, "He", 24.5874),
        (3, "Li", 5.3917),
    ],
)
def test_barklem_ionization_energies(
    barklem_ionization_energies, atomic_Number, element, first_ionization_energy
):
    assert len(barklem_ionization_energies) == 92
    # Test to see if any values have become nan in new columns
    assert ~barklem_ionization_energies.isna().values.any()
    row = barklem_ionization_energies.loc[atomic_Number]
    assert_allclose(row["IE1 [eV]"], first_ionization_energy)
    assert row["Element"] == element


@pytest.mark.parametrize(
    "molecule,ion1,adopted_energy",
    [
        ("H2", "H", 4.478007),
        ("Li2", "Li", 1.049900),
        ("B2", "B", 2.802000),
    ],
)
def test_barklem_dissociation_energies(
    barklem_dissociation_energies, molecule, ion1, adopted_energy
):
    assert len(barklem_dissociation_energies) == 291
    # Test to see if any values have become nan in new columns
    assert ~barklem_dissociation_energies.isna().values.any()

    row = barklem_dissociation_energies.loc[molecule]
    assert_allclose(row["Adopted Energy [eV]"], adopted_energy)
    assert row["Ion1"] == ion1


@pytest.mark.parametrize(
    "molecule,t05,t10000",
    [
        ("H2", -45134.0, 9.02320),
        ("Li2", -10578.3, 9.39624),
        ("B2", -28239.5, 9.77654),
    ],
)
# def test_barklem_equilbirium_constants():
def test_barklem_equilbirium_constants(
    barklem_equilibrium_constants, molecule, t05, t10000
):
    assert len(barklem_equilibrium_constants) == 291
    # Test to see if any values have become nan in new columns
    assert ~barklem_equilibrium_constants.isna().values.any()

    row = barklem_equilibrium_constants.loc[molecule]
    assert_allclose(row[0.50000], t05)
    assert_allclose(row[10000.00000], t10000)


@pytest.mark.parametrize(
    "molecule,t05,t10000",
    [
        ("H2", 0.250000, 194.871),
        ("Li2", 0.414917, 297872.000),
        ("B2", 1.878280, 83250.200),
    ],
)
def test_barklem_partion_functions(barklem_partition_functions, molecule, t05, t10000):
    assert len(barklem_partition_functions) == 291
    # Test to see if any values have become nan in new columns
    assert ~barklem_partition_functions.isna().values.any()

    row = barklem_partition_functions.loc[molecule]
    assert_allclose(row[0.50000], t05)
    assert_allclose(row[10000.00000], t10000)
