import pandas as pd
import pytest

from carsus.io.output.levels_lines import LevelsLinesPreparer
from astropy import units as u

from carsus.io.util import get_lvl_index2id


class ReaderStub:
    pass


class IonizationStub:
    def get_ground_levels(self):
        return pd.DataFrame(
            {
                "atomic_number": [6],
                "ion_charge": [0],
                "g": [1],
                "energy": [0.0],
            }
        )


def test_all_levels_data_uses_astropy_cm_inverse_to_ev_conversion():
    gfall_reader = ReaderStub()
    gfall_reader.levels = pd.DataFrame(
            {
                "energy": [1000.0, 2000.0],
                "j": [1.0, 2.0],
                "label": ["test 1", "test 2"],
                "method": ["meas", "meas"],
                "priority": [10, 10],
            },
        index=pd.MultiIndex.from_tuples(
            [(6, 0, 1), (6, 0, 2)],
            names=["atomic_number", "ion_charge", "level_index"],
        ),
    )

    preparer = LevelsLinesPreparer(
        IonizationStub(),
        gfall_reader,
        chianti_reader=None,
        cmfgen_reader=None,
        lanl_ads_reader=None,
    )

    levels = preparer.all_levels_data.reset_index()
    converted = levels.loc[levels["g"] == 3, "energy"].iloc[0]
    expected = u.Quantity([1000.0], "cm-1").to(
        "eV", equivalencies=u.spectral()
    ).value[0]

    assert converted == pytest.approx(expected)


def test_all_levels_data_drops_nist_ground_when_source_ground_exists():
    gfall_reader = ReaderStub()
    gfall_reader.levels = pd.DataFrame(
        {
            "energy": [0.0, 0.0, 1000.0],
            "j": [0.0, 1.0, 1.0],
            "label": ["ground 1", "ground 2", "excited"],
            "method": ["meas", "meas", "meas"],
            "priority": [10, 10, 10],
        },
        index=pd.MultiIndex.from_tuples(
            [(6, 0, 0), (6, 0, 1), (6, 0, 2)],
            names=["atomic_number", "ion_charge", "level_index"],
        ),
    )

    preparer = LevelsLinesPreparer(
        IonizationStub(),
        gfall_reader,
        chianti_reader=None,
        cmfgen_reader=None,
        lanl_ads_reader=None,
    )

    levels = preparer.all_levels_data
    ground_levels = levels[
        (levels["atomic_number"] == 6)
        & (levels["ion_number"] == 0)
        & (levels["energy"] == 0.0)
    ]

    assert set(ground_levels["ds_id"]) == {2}
    assert set(ground_levels["g"]) == {1, 3}


def test_get_lvl_index2id_maps_source_level_index_not_row_position():
    levels_all = pd.DataFrame(
        {
            "level_id": [100, 101],
            "atomic_number": [6, 6],
            "ion_number": [0, 0],
            "level_index": [10, 20],
        }
    )
    lines = pd.DataFrame(
        {"level_index_lower": [20], "level_index_upper": [10]},
        index=pd.MultiIndex.from_tuples(
            [(6, 0)],
            names=["atomic_number", "ion_number"],
        ),
    )

    matched = get_lvl_index2id(lines, levels_all)

    assert matched["lower_level_id"].tolist() == [101]
    assert matched["upper_level_id"].tolist() == [100]
