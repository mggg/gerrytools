"""Record GerryChain runs as self-describing BENDL files.

``RecordedChain`` wraps ``gerrychain.MarkovChain``: iterate it as usual and every step's
assignment streams into a BENDL bundle alongside the dual graph and optional metadata, published
atomically on clean completion. A clean run exposes a ``RecordedRun`` reader on
``chain.recording`` for reading the file back.
"""

from io import BytesIO
from typing import Any, cast

import geopandas as gpd
import pandas as pd
from binary_ensemble import BendlDecoder, BendlEncoder

from .recorded_chain import (
    GraphOrder,
    GraphOrderName,
    RecordedChain,
    RecordedRun,
    RunIterator,
    Variant,
)


def read_parquet_asset(
    decoder: BendlDecoder,
    name: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """Read an embedded Parquet asset into a DataFrame.

    Args:
        decoder (BendlDecoder): Decoder for the bundle containing the asset.
        name (str): Asset name in the bundle.
        **kwargs (Any): Additional arguments forwarded to :func:`pandas.read_parquet`.

    Returns:
        pandas.DataFrame: The decoded Parquet data.
    """
    return pd.read_parquet(BytesIO(decoder.read_asset_bytes(name)), **kwargs)


def read_geoparquet_asset(
    decoder: BendlDecoder,
    name: str,
    **kwargs: Any,
) -> gpd.GeoDataFrame:
    """Read an embedded GeoParquet asset into a GeoDataFrame.

    Args:
        decoder (BendlDecoder): Decoder for the bundle containing the asset.
        name (str): Asset name in the bundle.
        **kwargs (Any): Additional arguments forwarded to :func:`geopandas.read_parquet`.

    Returns:
        geopandas.GeoDataFrame: The decoded GeoParquet data.
    """
    # GeoPandas accepts binary file-like objects, but its current type stub omits them.
    return gpd.read_parquet(cast(Any, BytesIO(decoder.read_asset_bytes(name))), **kwargs)


__all__ = [
    "BendlDecoder",
    "BendlEncoder",
    "GraphOrder",
    "GraphOrderName",
    "RecordedChain",
    "RecordedRun",
    "RunIterator",
    "read_geoparquet_asset",
    "read_parquet_asset",
    "Variant",
]
