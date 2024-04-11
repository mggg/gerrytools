from .binary_ensemble import ben, ben_replay
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

logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG) # Uncomment this line to see debug messages

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
# handler.setLevel(logging.DEBUG) # Uncomment this line to see debug messages


logger.addHandler(handler)


__all__ = [
    "ben",
    "ben_replay",
    "msms_parse",
    "smc_parse",
    "canonicalize_ben_file",
    "relabel_json_file_by_key",
    "relabel_ben_file_by_key",
    "relabel_ben_file_with_map",
]
