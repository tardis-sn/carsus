from enum import IntEnum, unique


@unique
class DataSourceID(IntEnum):
    """Data source identifiers used in output ``ds_id`` columns."""

    NIST = 1
    GFALL = 2
    KNOX_LONG_ZETA = 3
    CHIANTI = 4
    CMFGEN = 5
    LANL_ADS = 6
