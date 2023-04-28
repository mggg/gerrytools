import warnings
from typing import Dict, List, TypeVar

import maup

A = TypeVar("A")
B = TypeVar("B")


def unitmap(source, target) -> dict:
    """
    Creates a mapping from source units to target units.

    Args:
        source (tuple): 2-tuple containing a `GeoDataFrame` and an index name corresponding
            to the unique identifiers of the units, e.g. `(vtds, "GEOID20")`.
            Unique identifiers will be keys in the resulting dictionary.
        target (tuple): 2-tuple containing a `GeoDataFrame` and an index name corresponding
            to the unique identifiers of the units, e.g. `(districts, "DISTRICTN")`.
            Unique identifiers will be values in the resulting dictionary.

    Returns:
        A dictionary mapping `_from` unique identifiers to `_to` unique identifiers.
    """
    # Explode each of the tuples.
    source_shapes, source_index = source
    target_shapes, target_index = target

    # Get rid of all the unnecessary data and set the indices of each dataframe
    # to the specified indices.
    source_shapes = source_shapes[[source_index, "geometry"]].set_index(source_index)
    target_shapes = target_shapes[[target_index, "geometry"]].set_index(target_index)

    # Ensure we're in the same CRS.
    target_shapes = target_shapes.to_crs(source_shapes.crs)

    # Set a progress bar; filter out all warnings; create the mapping.
    maup.progress.enabled = True
    warnings.simplefilter("ignore", UserWarning)
    warnings.simplefilter("ignore", FutureWarning)
    mapping = maup.assign(source_shapes, target_shapes)

    # Reset the mapping's index, zip, and return.
    mapping = mapping.reset_index()
    l, r = "l", "r"
    mapping.columns = [l, r]

    return dict(zip(mapping[l], mapping[r]))


def invert(unitmap: Dict[A, B]) -> Dict[B, List[A]]:
    """
    Inverts the provided unit mapping.

    Args:
        unitmap: Dictionary taking source unique identifiers to target unique
            identifiers.

    Returns:
        A dictionary mapping target unique identifiers to _lists_ of source
        unique identifiers.
    """
    # Invert the dictionary.
    inverse: Dict[B, List[A]] = {}

    for s, t in unitmap.items():
        if inverse.get(t, None):
            inverse[t].append(s)
        else:
            inverse[t] = [s]

    return inverse
