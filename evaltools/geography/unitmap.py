
import maup
import warnings


def unitmap(_from, _to) -> dict:
    """
    Creates a mapping from source units to target units.

    Args:
        _from (tuple): 2-tuple containing a `GeoDataFrame` and an index name corresponding
            to the unique identifiers of the units, e.g. `(vtds, "GEOID20")`.
            Unique identifiers will be keys in the resulting dictionary.
        _to (tuple): 2-tuple containing a `GeoDataFrame` and an index name corresponding
            to the unique identifiers of the units, e.g. `(districts, "DISTRICTN")`.
            Unique identifiers will be values in the resulting dictionary.

    Returns:
        A dictionary mapping `_from` unique identifiers to `_to` unique identifiers.
    """
    # Explode each of the tuples.
    source_shapes, source_index = _from
    target_shapes, target_index = _to

    # Get rid of all the unnecessary data and set the indices of each dataframe
    # to the specified indices.
    source_shapes = source_shapes[[source_index, "geometry"]].set_index(source_index)
    target_shapes = target_shapes[[target_index, "geometry"]].set_index(target_index)

    # Ensure we're in the same CRS.
    target_shapes = target_shapes.to_crs(source_shapes.crs)

    # Set a progress bar; filter out all warnings; create the mapping.
    maup.progress.enabled = True
    warnings.simplefilter("ignore", UserWarning)
    mapping = maup.assign(source_shapes, target_shapes)

    # Reset the mapping's index, zip, and return.
    mapping = mapping.reset_index()
    mapping.columns = [source_index, target_index]
    
    return dict(zip(mapping[source_index], mapping[target_index]))


def invert(unitmap) -> dict:
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
    inverse = {}

    for s, t in unitmap.items():
        if inverse.get(t, None): inverse[t].append(s)
        else: inverse[t] = [s]

    return inverse