
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from gerrychain.graph import Graph
import networkx as nx
from typing import Union, TypedDict, List, Tuple


def drawgraph(
        G: Graph, ax: Axes= None, x: str="INTPTLON20", y: str="INTPTLAT20",
        components: bool=False, node_size: float=1, **kwargs: TypedDict
    ) -> Union[Axes,List[Tuple[Figure, Axes]]]:
    """
    Draws a gerrychain Graph object. Returns a single Axes object (for dual
    graphs drawn whole) and lists of `(Figure, Axes)` pairs for graphs drawn
    component-wise.

    :param G: The dual graph to draw.
    :param ax: Optional; `matplotlib.axes.Axes` object. If not passed, one is
    created.
    :param x: Optional; vertex property used as the horizontal (E-W) coordinate.
    :param y: Optional; vertex property used as the vertical (N-S) coordinate.
    :param components: Optional; if `True`, the graph is assumed to have more
    than one connected component (e.g. Michigan) and is drawn component-wise and
    rather than return a single `Axes` object, return a list of `(Figure, Axes)`
    pairs. If something is passed to `ax`, the same Axes instance is used for
    each new Figure.
    :param node_size: Optional; specifies the default size of a vertex.
    :param kwargs: Optional; arguments to be passed to `nx.draw()`.
    """
    # Create a mapping from identifiers to positions.
    positions = {
        v: (properties[x], properties[y])
        for v, properties in G.nodes(data=True)
    }

    # If `components` is true, plot the graph component-wise. Otherwise plot
    # normally. First, set some properties common to both graphs.
    properties = {"pos": positions, "node_size": node_size }

    # Initialize `pairs` to None.
    pairs = None

    if not components:
        if not ax: axes = plt.axes()
        else: axes = ax
        nx.draw(G, ax=axes, **properties, **kwargs)
    else:
        # Create lists for figures and axes.
        pairs = []

        connected_components = [c for c in nx.connected_components(G)]
        for component in connected_components:
            # Create a new Figure object for each component.
            fig = plt.figure()
            if not ax: ax = plt.axes()

            # Plot the graph.
            subgraph = G.subgraph(component)
            nx.draw(subgraph, ax=ax, **properties, **kwargs)

            # Add them to their respective lists.
            pairs.append((fig, ax))

    return pairs if pairs else axes
