from pathlib import Path

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from gerrytools.plotting.data.seatsvotes import SeatsVotesPlot
from tests._image_snapshots import assert_image_snapshot
from tests.plotting._snapshot_utils import RNG_SEED, render_plot

SNAPSHOTS_DIR = Path(__file__).with_name("image_snapshots")


def make_seatsvotes_data(rng: np.random.Generator) -> list[float]:
    return sorted(rng.uniform(0.35, 0.65, 10).tolist())


class TestSeatsVotesSnapshots:
    @pytest.mark.snapshot
    def test_seatsvotes_basic_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)
        vote_shares = make_seatsvotes_data(rng)

        plot = SeatsVotesPlot(figure_size=(7, 6), dpi=100, legend=True)
        plot.add_election(
            target_party_vote_shares=vote_shares,
            name="Election A",
            linecolor="denim",
        )
        plot.add_proportionality_line(linecolor="grey", name="Proportional")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="seatsvotes_basic",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )

    @pytest.mark.snapshot
    def test_seatsvotes_multiple_elections_snapshot(self, tmp_path):
        rng = np.random.default_rng(RNG_SEED)

        plot = SeatsVotesPlot(figure_size=(7, 6), dpi=100, legend=True)
        for i in range(3):
            plot.add_election(
                target_party_vote_shares=make_seatsvotes_data(rng),
                name=f"Election {i + 1}",
            )
        plot.add_proportionality_line(linecolor="grey", name="Proportional")
        plot.add_efficiency_gap_line(linecolor="black", name="EG")
        img = render_plot(plot, tmp_path)

        assert_image_snapshot(
            img=img,
            name="seatsvotes_multiple_elections",
            snapshots_dir=SNAPSHOTS_DIR,
            artifacts_dir=tmp_path / "snapshot_artifacts",
        )
