import os
from os import path


def rename(old, new):
    """
    Renames all files in the path specified by `old` to `new`; intended for use
    with directories containing shapefiles. For example, if a directory called
    `blocks20/` contains no shapefile called `blocks20.shp`, this is (a) bad
    practice and (b) prevents GeoPandas from reading the shapefile from a
    partial filepath (e.g. `gpd.read_file("blocks20/")`).

    Example:
        Basic usage.

            from gerrytools.utils import rename

            old = "./data/geometries/to-be-renamed"
            new = "blocks20"
            rename(old, new)

    Args:
        old (str): _Directory_ where files to be renamed are located.
        new (str): New name to be applied to the directory and all files in it.
    """
    for file in os.listdir(old):
        # Check whether the file has an extension.
        if "." in file:
            extension = file[file.index(".") :]
            os.rename(path.join(old, file), path.join(old, new + extension))

    # Rename the root directory.
    split = old.split("/")
    _base = split[:-1] if split[-1] != "" else split[:-2]
    base = "/".join(_base)
    new_root = path.join(base, new)
    os.rename(old, new_root)
