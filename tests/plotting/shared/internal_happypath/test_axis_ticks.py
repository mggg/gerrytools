from gerrytools.plotting.data.scatterplot import ScatterPlot


def test_set_xticks_and_set_yticks_apply_locations_and_labels():
    plot = ScatterPlot()
    plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0], name="line")

    plot.set_xticks([0.0, 0.5, 1.0], labels=["0%", "50%", "100%"])
    plot.set_yticks([0.0, 1.0], labels=["low", "high"])

    ax = plot.ax

    assert list(ax.get_xticks()) == [0.0, 0.5, 1.0]
    assert [tick.get_text() for tick in ax.get_xticklabels()] == ["0%", "50%", "100%"]

    assert list(ax.get_yticks()) == [0.0, 1.0]
    assert [tick.get_text() for tick in ax.get_yticklabels()] == ["low", "high"]


def test_set_xticks_and_set_yticks_can_clear_ticks_and_labels():
    plot = ScatterPlot()
    plot.add_series(x=[0.0, 1.0], y=[0.0, 1.0], name="line")

    plot.set_xticks([])
    plot.set_yticks([])

    ax = plot.ax
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
