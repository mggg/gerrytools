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

    # As a contrived example, let the COIs be squares of size 3
    # tiling a 9x9 grid contained within the 10x10 grid.
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
