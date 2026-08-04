from collections.abc import Hashable
from typing import cast

import numpy as np
import pytest
from gerrychain.updaters import Election
from gerrychain.updaters.election import ElectionResults

from gerrytools.scoring import formulas


def gerrychain_results(party: np.ndarray, opposition: np.ndarray) -> ElectionResults:
    election = Election("test", {"A": "party", "B": "opposition"})
    districts = range(len(party))
    counts: dict[str, dict[Hashable, int | float]] = {
        "A": {district: float(votes) for district, votes in enumerate(party)},
        "B": {district: float(votes) for district, votes in enumerate(opposition)},
    }
    return ElectionResults(election, counts, districts)


def test_generated_partisan_formulas_match_gerrychain() -> None:
    rng = np.random.default_rng(20_260_721)

    for _ in range(100):
        district_count = int(rng.integers(1, 20))
        party = rng.integers(1, 1_000, district_count).astype(np.float64)
        opposition = rng.integers(1, 1_000, district_count).astype(np.float64)
        expected = gerrychain_results(party, opposition)

        assert formulas.seats(party, opposition) == expected.seats("A")
        assert formulas.overall_vote_share(party, opposition) == pytest.approx(
            expected.percent("A")
        )
        assert formulas.simplified_efficiency_gap(party, opposition) == pytest.approx(
            expected.seats("A") / district_count + 0.5 - 2 * expected.percent("A")
        )
        assert formulas.mean_median(party, opposition) == pytest.approx(expected.mean_median())
        shares = party / (party + opposition)
        reference = np.mean(shares)
        expected_bias = np.sum(shares > reference) + 0.5 * np.sum(shares == reference)
        expected_bias = expected_bias / district_count - 0.5
        assert formulas.partisan_bias(party, opposition) == pytest.approx(expected_bias)


def test_disproportionality_uses_aggregate_vote_share() -> None:
    party = [90, 1]
    opposition = [10, 9]

    assert formulas.disproportionality(party, opposition) == pytest.approx(0.5 - 91 / 110)


def test_tied_districts_are_not_wins_and_waste_all_votes() -> None:
    party = np.array([60.0, 50.0])
    opposition = np.array([40.0, 50.0])

    np.testing.assert_equal(formulas.district_wins(party, opposition), [True, False])
    assert formulas.seats(party, opposition) == 1
    # 60-40: threshold 51, party wastes 9, opposition 40; the 50-50 tie contributes zero.
    assert formulas.efficiency_gap(party, opposition) == pytest.approx((40 - 9) / 200)
    assert formulas.efficiency_gap([50], [50]) == pytest.approx(0.0)
    assert formulas.efficiency_gap([51], [50]) == pytest.approx(50 / 101)


def test_efficiency_gap_matches_hand_computed_wasted_votes() -> None:
    # Thresholds are all 51. District 1: party wastes 60-51=9, opposition 40.
    # District 2: opposition wastes 55-51=4, party 45. District 3 is tied: both waste all 50.
    party = [60.0, 45.0, 50.0]
    opposition = [40.0, 55.0, 50.0]

    expected = ((40 - 9) + (4 - 45) + (50 - 50)) / 300
    assert formulas.efficiency_gap(party, opposition) == pytest.approx(expected)


def test_efficiency_gap_uses_half_turnout_threshold_for_fractional_tallies() -> None:
    # Integer tallies keep the historical integer thresholds (all 51 here).
    party = [60.0, 45.0, 50.0]
    opposition = [40.0, 55.0, 50.0]
    assert formulas.efficiency_gap(party, opposition) == pytest.approx(((40 - 9) + (4 - 45)) / 300)
    # Share-scale tallies reproduce the classic value: each winner wastes 0.1, each loser 0.4.
    assert formulas.efficiency_gap([0.6] * 10, [0.4] * 10) == pytest.approx(0.3)
    # Fractional tallies use threshold total/2 = 4.0 in both districts, so the wasted-vote
    # differentials are (1.5 - 2.5) and (0.5 - 3.5).
    assert formulas.efficiency_gap([6.5, 3.5], [1.5, 4.5]) == pytest.approx(-4.0 / 16.0)


def test_efficiency_gap_threshold_choice_is_per_district() -> None:
    # District 1 keeps the integer threshold 51 (differential 40 - 9); district 2 has fractional
    # tallies and threshold 0.5 (differential 0.4 - 0.1).
    assert formulas.efficiency_gap([60.0, 0.6], [40.0, 0.4]) == pytest.approx((31 + 0.3) / 101)


def test_efficiency_gap_reverses_sign_when_parties_swap() -> None:
    rng = np.random.default_rng(20_260_728)
    party = rng.integers(0, 100, size=(20, 8)).astype(np.float64)
    opposition = rng.integers(0, 100, size=(20, 8)).astype(np.float64)
    opposition[0] = party[0]  # a fully tied plan must map to exactly zero
    opposition[1, :4] = party[1, :4]  # and partial ties must stay antisymmetric

    forward = np.asarray(formulas.efficiency_gap(party, opposition))
    backward = np.asarray(formulas.efficiency_gap(opposition, party))
    np.testing.assert_allclose(forward, -backward)
    assert forward[0] == 0.0
    np.testing.assert_array_less(np.abs(forward), 1.0 + 1e-12)


def test_zero_turnout_is_excluded_from_partisan_bias() -> None:
    party = np.array([50.0, 0.0])
    opposition = np.array([50.0, 0.0])

    np.testing.assert_equal(formulas.district_wins(party, opposition), [False, False])
    np.testing.assert_allclose(formulas.district_vote_shares(party, opposition), [0.5, np.nan])
    assert formulas.seats(party, opposition) == 0
    assert formulas.partisan_bias(party, opposition) == 0
    assert np.isnan(formulas.partisan_bias([0], [0]))
    assert np.isnan(formulas.mean_median(party, opposition))
    assert np.isnan(formulas.partisan_gini(party, opposition))


@pytest.mark.parametrize("district_count", [3, 6, 7])
def test_partisan_bias_treats_uniform_roundoff_as_ties(district_count: int) -> None:
    party = np.full(district_count, 2.0)
    opposition = np.full(district_count, 3.0)

    assert formulas.partisan_bias(party, opposition) == 0.0


@pytest.mark.parametrize(
    ("offsets", "expected"),
    [
        ([0.5e-9, -0.25e-9, -0.25e-9], 0.0),
        ([1.5e-9, -0.75e-9, -0.75e-9], 1 / 6),
    ],
)
def test_partisan_bias_pins_the_tie_tolerance(offsets, expected: float) -> None:
    shares = 0.5 + np.asarray(offsets)

    assert formulas.partisan_bias(shares, 1 - shares) == pytest.approx(expected)


def test_turnout_models_half_seats_and_bounded_partisan_gini() -> None:
    party = np.array([400.0, 45.0, 60.0])
    opposition = np.array([600.0, 55.0, 40.0])

    assert formulas.partisan_bias(party, opposition, turnout_model="equal") == pytest.approx(-1 / 6)
    assert formulas.partisan_bias(party, opposition, turnout_model="observed") == pytest.approx(
        1 / 6
    )

    symmetric = np.array([10.0, 20.0, 40.0, 60.0, 70.0])
    assert formulas.partisan_bias(symmetric, 100 - symmetric) == pytest.approx(0)
    assert formulas.partisan_bias(100 - symmetric, symmetric) == pytest.approx(0)

    extreme = np.array([1.0, 99.0, 99.0])
    assert formulas.partisan_gini(extreme, 100 - extreme) == pytest.approx(1 / 3)

    unequal_symmetric = np.array([400.0, 60.0])
    unequal_opposition = np.array([600.0, 40.0])
    assert formulas.partisan_gini(
        unequal_symmetric, unequal_opposition, turnout_model="equal"
    ) == pytest.approx(0)
    assert formulas.partisan_gini(
        unequal_symmetric, unequal_opposition, turnout_model="observed"
    ) == pytest.approx(9 / 55)


def test_partisan_scores_preserve_party_symmetry() -> None:
    rng = np.random.default_rng(20_260_722)

    for turnout_model in ("equal", "observed"):
        for _ in range(100):
            party = rng.integers(1, 1_000, 20)
            opposition = rng.integers(1, 1_000, 20)
            bias = formulas.partisan_bias(party, opposition, turnout_model=turnout_model)
            reverse_bias = formulas.partisan_bias(opposition, party, turnout_model=turnout_model)
            gini = formulas.partisan_gini(party, opposition, turnout_model=turnout_model)
            reverse_gini = formulas.partisan_gini(opposition, party, turnout_model=turnout_model)

            assert bias == pytest.approx(-reverse_bias)
            assert gini == pytest.approx(reverse_gini)
            assert 0 <= gini <= 1


def test_cross_election_scores_preserve_strict_win_and_margin_rules() -> None:
    party = np.array([[60, 48, 40, 50], [55, 45, 60, 51]], dtype=np.float64)
    opposition = np.array([[40, 52, 60, 50], [45, 55, 40, 49]], dtype=np.float64)

    np.testing.assert_array_equal(formulas.seats(party, opposition), [1, 3])
    np.testing.assert_array_equal(formulas.party_wins_by_district(party, opposition), [2, 0, 1, 1])
    assert formulas.competitive_contests(party, opposition) == 3
    assert formulas.swing_districts(party, opposition) == 2
    assert formulas.party_districts(party, opposition) == 1
    assert formulas.opposition_party_districts(party, opposition) == 1
    assert formulas.aggregate_seats(party, opposition) == 4

    seat_shares = np.array([0.25, 0.75])
    vote_shares = np.sum(party, axis=-1) / np.sum(party + opposition, axis=-1)
    assert formulas.mean_signed_seat_vote_gap(party, opposition) == pytest.approx(
        np.mean(seat_shares - vote_shares)
    )
    assert formulas.mean_absolute_seat_vote_gap(party, opposition) == pytest.approx(
        np.mean(np.abs(seat_shares - vote_shares))
    )


def test_cross_election_mean_gaps_are_nan_when_an_election_has_zero_turnout() -> None:
    party = np.array([[60, 40], [0, 0]], dtype=np.float64)
    opposition = np.array([[40, 60], [0, 0]], dtype=np.float64)

    assert np.isnan(formulas.mean_signed_seat_vote_gap(party, opposition))
    assert np.isnan(formulas.mean_absolute_seat_vote_gap(party, opposition))


def test_tied_districts_are_swing_but_not_stable_for_either_party() -> None:
    party = np.array([[50, 60, 40], [50, 55, 45]], dtype=np.float64)
    opposition = np.array([[50, 40, 60], [50, 45, 55]], dtype=np.float64)

    assert formulas.party_districts(party, opposition) == 1
    assert formulas.opposition_party_districts(party, opposition) == 1
    assert formulas.swing_districts(party, opposition) == 1


def test_competitive_contests_use_an_open_interval() -> None:
    party = np.array([[47, 48, 53]], dtype=np.float64)
    opposition = 100 - party

    assert formulas.competitive_contests(party, opposition, vote_share_margin=0.03) == 1


def test_election_scores_accept_leading_plan_batches() -> None:
    party = np.array(
        [
            [[60, 40], [55, 45]],
            [[40, 60], [45, 55]],
        ],
        dtype=np.float64,
    )
    opposition = 100 - party

    np.testing.assert_array_equal(formulas.seats(party, opposition), [[1, 1], [1, 1]])
    np.testing.assert_array_equal(formulas.aggregate_seats(party, opposition), [2, 2])
    np.testing.assert_array_equal(formulas.party_districts(party, opposition), [1, 1])


def test_eguia_uses_population_weighted_region_winners() -> None:
    score = formulas.eguia(
        party_votes=[60, 40, 55, 45],
        opposition_votes=[40, 60, 45, 55],
        region_party_votes=[100, 40, 60],
        region_opposition_votes=[90, 50, 55],
        region_populations=[100, 200, 300],
    )

    assert score == pytest.approx(0.5 - 400 / 600)


def test_population_demographic_and_compactness_derivations() -> None:
    populations = np.array([90.0, 100.0, 110.0])
    np.testing.assert_allclose(formulas.population_deviations(populations), [-0.1, 0.0, 0.1])
    assert formulas.max_absolute_population_deviation(populations) == 10
    assert formulas.max_absolute_population_deviation(
        populations, relative_to_ideal=True
    ) == pytest.approx(0.1)
    assert formulas.max_population_deviation(populations) == 20
    assert formulas.max_population_deviation(populations, relative_to_ideal=True) == pytest.approx(
        0.2
    )

    shares = formulas.demographic_shares([45, 60, 0], [90, 100, 0])
    np.testing.assert_allclose(shares, [0.5, 0.6, np.nan])
    assert formulas.districts_above_threshold([45, 60, 0], [90, 100, 0]) == 1
    np.testing.assert_allclose(
        formulas.schwartzberg(polsby_popper_scores=[0.25, 1.0]),
        [2.0, 1.0],
    )


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: formulas.seats([], []), "nonempty district axis"),
        (lambda: formulas.seats([1], [1, 2]), "same shape"),
        (lambda: formulas.seats([-1], [1]), "cannot contain negative"),
        (lambda: formulas.seats([np.nan], [1]), "finite values"),
        (
            lambda: formulas.competitive_contests([[1]], [[1]], vote_share_margin=0.6),
            "between zero and one half",
        ),
        (
            lambda: formulas.districts_above_threshold([1], [2], threshold=1.1),
            "between zero and one",
        ),
        (
            lambda: formulas.districts_above_threshold([1], [2], threshold=cast(float, "bad")),
            "real number",
        ),
        (
            lambda: formulas.districts_above_threshold([1], [2], threshold=True),
            "real number",
        ),
        (
            lambda: formulas.demographic_shares([2], [1]),
            "cannot exceed total_populations",
        ),
        (
            lambda: formulas.eguia([1], [0], [1, 0], [0, 1], [1]),
            "same region axis",
        ),
        (
            lambda: formulas.eguia(
                [1],
                [0],
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 1], [1, 1], [1, 1]],
            ),
            "cannot be broadcast",
        ),
        (
            lambda: formulas.eguia(
                [[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]],
                [[1, 1], [1, 1], [1, 1]],
            ),
            "election axes are incompatible",
        ),
        (
            lambda: formulas.population_deviations([0, 0]),
            "positive totals",
        ),
        (
            lambda: formulas.max_population_deviation([0, 0]),
            "positive totals",
        ),
        (lambda: formulas.schwartzberg([0]), "must be positive"),
        (lambda: formulas.schwartzberg([1.1]), "cannot exceed one"),
        (
            lambda: formulas.partisan_bias(
                [1], [1], turnout_model=cast(formulas.TurnoutModel, "invalid")
            ),
            "must be 'equal' or 'observed'",
        ),
    ],
)
def test_derived_scores_reject_invalid_inputs(call, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        call()
