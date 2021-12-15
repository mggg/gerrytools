from evaltools.evaluation import block_level_coi_preservation
from gerrychain.grid import Grid


def test_block_level_coi_preservation_small_cois():
    n = 10
    grid = Grid((n, n))  # default plan is 4 squares

    def coord_to_blocks(x, y):
        return {(2 * n * (2 * x)) + (2 * y), (2 * n * (2 * x)) + (2 * y) + 1,
                (2 * n * (2 * x + 1)) + (2 * y),
                (2 * n * (2 * x + 1)) + (2 * y) + 1}

    # Let the "blocks" be the 20x20 grid.
    unit_blocks = {(x, y): coord_to_blocks(x, y) for x, y in grid.graph.nodes}
    block_pops = {b: 1 / 4 for b in range(20**2)}

    # Let the COIs be squares of size 3 tiling a 9x9 grid contained
    # within the 10x10 grid.
    coi_blocks = {
        idx: set.union(*(coord_to_blocks(x + 3 * (idx // 3), y + 3 * (idx % 3))
                         for x in range(3) for y in range(3)))
        for idx in range(9)
    }

    thresholds = [0.6, 0.75, 1.]
    # dist 1 is [0, 4] x [0, 4] (inclusive)
    # dist 2 is [0, 4] x [5, 9]
    # dist 3 is [5, 9] x [0, 4]
    # dist 4 is [5, 9] x [5, 9]
    # => [60% threshold, 75% threshold, 100% threshold]
    # [1, 1, 1] COI 0 is [0, 2] x [0, 2] -> all pop in dist 1
    # [1, 0, 0] COI 1 is [0, 2] x [3, 5] -> 2/3 pop in dist 1, 1/3 pop in dist 2
    # [1, 1, 1] COI 2 is [0, 2] x [6, 8] -> all pop in dist 2
    # [1, 0, 0] COI 3 is [3, 5] x [0, 2] -> 2/3 pop in dist 1, 1/3 pop in dist 3
    # [0, 0, 0] COI 4 is [3, 5] x [3, 5] ->
    #  [3, 4] x [3, 4] in dist 1 -> 4/9 pop in dist 1
    #  [3, 4] x [5, 5] in dist 2 -> 2/9 pop in dist 2
    #  [5, 5] x [3, 4] in dist 3 -> 2/9 pop in dist 3
    #  [5, 5] x [5, 5] in dist 4 -> 1/9 pop in dist 4
    # [1, 0, 0] COI 5 is [3, 5] x [6, 8] -> 2/3 pop in dist 2, 1/3 pop in dist 4
    # [1, 1, 1] COI 6 is [6, 8] x [0, 2] -> all pop in dist 3
    # [1, 0, 0] COI 7 is [6, 8] x [3, 5] -> 2/3 pop in dist 3, 1/3 pop in dist 4
    # [1, 1, 1] COI 8 is [6, 8] x [6, 8] -> all pop in dist 4
    expected_scores = {0.6: 8, 0.75: 4, 1.0: 4}
    coi_score_fn = block_level_coi_preservation(unit_blocks, coi_blocks,
                                                block_pops, thresholds)
    assert coi_score_fn(grid) == expected_scores


def test_block_level_coi_preservation_large_cois():
    n = 10
    grid = Grid((n, n))  # default plan is 4 squares

    def coord_to_blocks(x, y):
        return {(2 * n * (2 * x)) + (2 * y), (2 * n * (2 * x)) + (2 * y) + 1,
                (2 * n * (2 * x + 1)) + (2 * y),
                (2 * n * (2 * x + 1)) + (2 * y) + 1}

    # Let the "blocks" be the 20x20 grid.
    unit_blocks = {(x, y): coord_to_blocks(x, y) for x, y in grid.graph.nodes}
    block_pops = {b: 1 / 4 for b in range(20**2)}

    # Let the COIs be three 10x3 horizonal strips.
    coi_blocks = {
        idx: set.union(*(coord_to_blocks(x, y + 3 * idx) for x in range(10)
                         for y in range(3)))
        for idx in range(3)
    }

    thresholds = [0.5, 0.6, 0.75, 1]
    # dist 1 is [0, 4] x [0, 4] (inclusive)
    # dist 2 is [0, 4] x [5, 9]
    # dist 3 is [5, 9] x [0, 4]
    # dist 4 is [5, 9] x [5, 9]
    #
    # f_1 (COI containment within district):
    # COI 0 is [0, 9] x [0, 2] -> 1/2 in dist 1, 1/2 in dist 2
    # COI 1 is [0, 9] x [3, 5] ->
    #   [0, 4] x [3, 4] -> 1/3 in dist 1
    #   [5, 9] x [3, 4] -> 1/3 in dist 2
    #   [0, 4] x [5, 5] -> 1/6 in dist 3
    #   [5, 9] x [5, 5] -> 1/6 in dist 4
    # COI 2 is [0, 9] x [6, 8] -> 1/2 in dist 3, 1/2 in dist 4
    # Thus, f_1 contributes at most 2 points with a threshold of
    # ≤50% and 0 points otherwise.

    # f_2 (district containment within COI):
    # COI 0 is [0, 9] x [0, 2] ->
    #   [0, 4] x [0, 2] -> contains 60% of dist 1 pop
    #   [5, 9] x [0, 2] -> contains 60% of dist 2 pop
    # COI 1 is [0, 9] x [3, 5] ->
    #   [0, 4] x [3, 4] -> contains 40% of dist 1 pop
    #   [5, 9] x [3, 4] -> contains 40% of dist 2 pop
    #   [0, 4] x [5, 5] -> contains 20% of dist 3 pop
    #   [5, 9] x [5, 5] -> contains 20% of dist 4 pop
    # COI 2 is [0, 9] x [6, 8] ->
    #   [0, 4] x [6, 8] -> contains 60% of dist 3 pop
    #   [5, 9] x [6, 8] -> contains 60% of dist 4 pop
    # Thus, f_2 contributes at most 2 points with a threshold of
    # ≤60% and 0 points otherwise.
    expected_scores = {0.5: 2, 0.6: 2, 0.75: 0, 1.0: 0}
    score_fn = block_level_coi_preservation(unit_blocks=unit_blocks,
                                            coi_blocks=coi_blocks,
                                            block_pops=block_pops,
                                            thresholds=thresholds,
                                            partial_districts=False)
    assert score_fn(grid) == expected_scores

    # For the alternative version of f_2, we have
    #   2 districts ≤60% contained in COI 0 / 1.2 districts' pop in COI 0
    #     => 2/1.2 for COI 0 when threshold ≤60%, 0 otherwise
    #   ...and similarly for COI 2.
    expected_scores_partial_dists = {
        0.5: 4 / 1.2,
        0.6: 4 / 1.2,
        0.75: 0,
        1.0: 0
    }
    score_fn_partial_dists = block_level_coi_preservation(
        unit_blocks=unit_blocks,
        coi_blocks=coi_blocks,
        block_pops=block_pops,
        thresholds=thresholds,
        partial_districts=True)
    scores_partial_dists = score_fn_partial_dists(grid)
    assert scores_partial_dists.keys() == expected_scores_partial_dists.keys()
    assert all(
        abs(expected - scores_partial_dists[t]) < 1e-10
        for t, expected in expected_scores_partial_dists.items())
