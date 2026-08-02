from collections.abc import Hashable, Mapping
from typing import Any, cast

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _resolve_positions(
    graph: nx.Graph,
    *,
    pos: Mapping[Hashable, tuple[float, float]] | None,
    x: str,
    y: str,
) -> dict[Hashable, tuple[float, float]]:
    """Resolve node positions from an explicit mapping or per-node coordinate attributes.

    Raises:
        ValueError: If a node lacks the ``x``/``y`` coordinate attributes and no ``pos``
            mapping was given.
    """
    if pos is not None:
        return dict(pos)

    positions: dict[Hashable, tuple[float, float]] = {}
    for node, properties in graph.nodes(data=True):
        if x not in properties or y not in properties:
            raise ValueError(
                f"Node {node!r} has no {x!r}/{y!r} coordinate attributes. Pass pos=... with "
                "explicit positions, or set x=/y= to the node attributes holding coordinates."
            )
        positions[node] = (float(properties[x]), float(properties[y]))
    return positions


def draw_graph(
    graph: nx.Graph,
    *,
    ax: Axes | None = None,
    pos: Mapping[Hashable, tuple[float, float]] | None = None,
    x: str = "INTPTLON20",
    y: str = "INTPTLAT20",
    node_size: float = 1,
    **kwargs: object,
) -> Axes:
    """Draw a gerrychain dual graph onto one axes.

    Args:
        graph (Graph): The dual graph to draw.
        ax (Axes, optional): `matplotlib.axes.Axes` object. If not passed, a fresh
            figure and axes are created.
        pos (Mapping | None, optional): Explicit node-to-``(x, y)`` positions. When given,
            the coordinate attribute keys are ignored. Defaults to None.
        x (str, optional): Vertex property used as the horizontal (E-W) coordinate. Defaults to
            ``"INTPTLON20"``.
        y (str, optional): Vertex property used as the vertical (N-S) coordinate. Defaults to
            ``"INTPTLAT20"``.
        node_size (float, optional): Vertex size passed to NetworkX. Defaults to 1.
        **kwargs (object): Additional keyword arguments passed to ``networkx.draw``.

    Returns:
        Axes: The axes the graph was drawn onto.
    """
    positions = _resolve_positions(graph, pos=pos, x=x, y=y)
    if ax is None:
        _, ax = plt.subplots()
    # networkx.draw's stub types each forwarded kwarg; this pass-through surface is dynamic.
    nx.draw(graph, ax=ax, pos=positions, node_size=node_size, **cast("dict[str, Any]", kwargs))
    return ax


def draw_graph_components(
    graph: nx.Graph,
    *,
    pos: Mapping[Hashable, tuple[float, float]] | None = None,
    x: str = "INTPTLON20",
    y: str = "INTPTLAT20",
    node_size: float = 1,
    **kwargs: object,
) -> list[tuple[Figure, Axes]]:
    """Draw each connected component of a dual graph on its own fresh figure.

    Args:
        graph (Graph): The dual graph to draw.
        pos (Mapping | None, optional): Explicit node-to-``(x, y)`` positions. When given,
            the coordinate attribute keys are ignored. Defaults to None.
        x (str, optional): Vertex property used as the horizontal (E-W) coordinate. Defaults to
            ``"INTPTLON20"``.
        y (str, optional): Vertex property used as the vertical (N-S) coordinate. Defaults to
            ``"INTPTLAT20"``.
        node_size (float, optional): Vertex size passed to NetworkX. Defaults to 1.
        **kwargs (object): Additional keyword arguments passed to ``networkx.draw``.

    Returns:
        list[tuple[Figure, Axes]]: One ``(Figure, Axes)`` pair per connected component.
    """
    positions = _resolve_positions(graph, pos=pos, x=x, y=y)
    pairs: list[tuple[Figure, Axes]] = []
    for component in nx.connected_components(graph):
        fig = Figure()
        component_ax = fig.subplots()
        # networkx.draw's stub types each forwarded kwarg; this pass-through surface is dynamic.
        nx.draw(
            graph.subgraph(component),
            ax=component_ax,
            pos=positions,
            node_size=node_size,
            **cast("dict[str, Any]", kwargs),
        )
        pairs.append((fig, component_ax))
    return pairs
