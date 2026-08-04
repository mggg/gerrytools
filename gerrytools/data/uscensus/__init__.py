"""Census data retrieval: American Community Survey (ACS) and decennial PL94-171.

Exposes the fetch functions plus the table-definition classes that ``acs()`` and ``census()``
accept via their ``tables`` / ``table`` parameters, so callers can pass a custom table (for example
``acs(..., tables=[ACSHispByRaceTableInfo()])``) without reaching into the private modules.
"""

from ._api import CensusRateLimitError
from .acs import acs, acs_full, cvap
from .block_cvap import block_cvap_estimates
from .census import census
from .census_tables import (
    ACSAgeTableInfo,
    ACSCVAPTableInfo,
    ACSHispByRaceTableInfo,
    ACSNamedTableInfo,
    ACSRacePopTableInfo,
    ACSTableInfo,
    ACSTotPopTableInfo,
    ACSVAPTableInfo,
    PLBlockVAPTableInfo,
    PLTableInfo,
    census_column_name,
    pl_table,
)

__all__ = [
    # Fetch functions
    "acs",
    "acs_full",
    "cvap",
    "census",
    "block_cvap_estimates",
    # ACS table definitions
    "ACSTableInfo",
    "ACSNamedTableInfo",
    "ACSTotPopTableInfo",
    "ACSRacePopTableInfo",
    "ACSAgeTableInfo",
    "ACSVAPTableInfo",
    "ACSCVAPTableInfo",
    "ACSHispByRaceTableInfo",
    # Decennial PL table definitions
    "PLTableInfo",
    "PLBlockVAPTableInfo",
    "pl_table",
    "census_column_name",
    # Errors
    "CensusRateLimitError",
]
