from .binary_ensamble import ben
from .reben import (
    canonicalize_ben_file,
    relabel_json_file_by_key,
    relabel_ben_file_by_key,
    relabel_ben_file_with_map,
)
from .parse import msms_parse, smc_parse
import logging
import warnings

# There is a bug in the docker SDK package that causes this error to always be thrown
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

logger = logging.getLogger("ben")

logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler()
# handler.setLevel(logging.DEBUG) # Uncomment this line to see debug messages
handler.setLevel(logging.INFO)


logger.addHandler(handler)


__all__ = [
    "ben",
    "msms_parse",
    "smc_parse",
    "canonicalize_ben_file",
    "relabel_json_file_by_key",
    "relabel_ben_file_by_key",
    "relabel_ben_file_with_map",
]
