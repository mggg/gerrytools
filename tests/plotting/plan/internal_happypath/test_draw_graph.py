import matplotlib.pyplot as plt
import networkx as nx
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from gerrytools.plotting import draw_graph, draw_graph_components


def _graph() -> nx.Graph:
    graph = nx.Graph()
    graph.add_node(1, x=0, y=0)
    graph.add_node(2, x=1, y=1)
    graph.add_node(3, x=2, y=2)
    graph.add_edge(1, 2)
    return graph


def test_draw_graph_returns_the_axes():
    graph = _graph()
    ax = draw_graph(graph, x="x", y="y")
    assert isinstance(ax, Axes)
    assert isinstance(ax.figure, Figure)  # narrows Figure | SubFigure for plt.close
    plt.close(ax.figure)


def test_draw_graph_components_returns_figure_axes_pairs():
    graph = _graph()
    pairs = draw_graph_components(graph, x="x", y="y")
    assert len(pairs) == 2
    assert pairs[0][1] is not pairs[1][1]
    for fig, _ in pairs:
        plt.close(fig)


def test_draw_graph_accepts_explicit_positions():
    graph = nx.Graph()
    graph.add_edge("a", "b")
    ax = draw_graph(graph, pos={"a": (0.0, 0.0), "b": (1.0, 1.0)})
    assert isinstance(ax, Axes)
    assert isinstance(ax.figure, Figure)  # narrows Figure | SubFigure for plt.close
    plt.close(ax.figure)


def test_draw_graph_components_do_not_register_pyplot_figures():
    plt.close("all")
    graph = nx.Graph()
    graph.add_nodes_from([(0, {"x": 0, "y": 0}), (1, {"x": 1, "y": 1})])

    pairs = draw_graph_components(graph, x="x", y="y")

    assert len(pairs) == 2
    assert plt.get_fignums() == []


def test_draw_graph_coerces_coordinate_attributes_to_float(monkeypatch):
    graph = nx.Graph()
    graph.add_node("a", x="-90.5", y="40.25")
    captured = {}

    def capture_draw(graph, *, ax, pos, node_size, **kwargs):
        captured.update(pos)

    monkeypatch.setattr(nx, "draw", capture_draw)
    ax = draw_graph(graph, x="x", y="y")

    assert captured == {"a": (-90.5, 40.25)}
    assert isinstance(ax.figure, Figure)
    plt.close(ax.figure)


def test_draw_graph_missing_coordinates_raises_clearly():
    graph = nx.Graph()
    graph.add_node("a")
    with pytest.raises(ValueError, match="coordinate attributes"):
        draw_graph(graph)
