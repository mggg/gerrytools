"""Array formulas for scores derived from district tallies.

The last axis always represents districts. Functions that combine elections use the penultimate
axis for elections and preserve any earlier batch axes. Party and opposition inputs are treated as
the two-party vote, so these functions do not include votes for third parties or independents.

Unless a function says otherwise, a tied district is not a win for either party. The formulas are
reported from the point of view of ``party_votes``. Each score documents its sign convention,
turnout assumptions, and relationship to the cited literature.
"""

from numbers import Real
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gerrytools import _partisan_math
from gerrytools._partisan_math import _divide, _matching, _nonnegative, _paired

TurnoutModel = Literal["equal", "observed"]
_PARTISAN_BIAS_TIE_TOLERANCE = 1e-9


def district_vote_shares(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64]:
    r"""Return the party's two-party vote share in each district.

    For district :math:`i`, the share is
    :math:`v_i = p_i / (p_i + o_i)`, where :math:`p_i` and :math:`o_i` are the party and
    opposition tallies. A district with no two-party votes has no defined vote share and is
    represented by ``NaN``.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64]: An array with the input shape containing district-level shares in
            :math:`[0, 1]`, or ``NaN`` for zero-turnout districts.
    """
    return _partisan_math.district_vote_shares(party_votes, opposition_votes)


def district_wins(party_votes: ArrayLike, opposition_votes: ArrayLike) -> NDArray[np.bool_]:
    r"""Identify districts strictly won by the party.

    A district is a win exactly when its party tally exceeds its opposition tally. Ties and
    zero-turnout districts are not wins. This is the strict two-party seat-outcome convention used
    by DeFord et al.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.bool\_]: A Boolean array with the same shape as the inputs.
    """
    return _partisan_math.district_wins(party_votes, opposition_votes)


def seats(party_votes: ArrayLike, opposition_votes: ArrayLike) -> NDArray[np.intp] | np.intp:
    r"""Count the party's seats under a strict two-party win rule.

    The result is :math:`\#\{i : p_i > o_i\}` along the district axis. Ties and zero-turnout
    districts contribute no seat. Earlier axes, such as election or batch axes, are preserved.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.intp] | np.intp: The seat count for each row preceding the district axis, or a
            scalar for one-dimensional inputs.
    """
    return np.sum(district_wins(party_votes, opposition_votes), axis=-1)


def overall_vote_share(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Return the party's aggregate two-party vote share.

    The result is :math:`\sum_i p_i / \sum_i (p_i + o_i)`. It therefore weights district vote
    shares by their observed two-party turnout. This generally differs from the arithmetic mean of
    district vote shares; the two coincide when turnout is equal across districts. A row with no
    two-party votes returns ``NaN``.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The aggregate share for each row preceding the district
            axis, or a scalar for one-dimensional inputs.
    """
    return _partisan_math.overall_vote_share(party_votes, opposition_votes)


def _partisan_shares_and_reference(
    party_votes: ArrayLike,
    opposition_votes: ArrayLike,
    turnout_model: TurnoutModel,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute district vote shares and the selected statewide reference share.

    ``"equal"`` takes the arithmetic mean of valid district shares, excluding zero-turnout
    districts. ``"observed"`` computes the aggregate two-party share, weighting districts by
    observed turnout. The reference retains a length-one district axis so it broadcasts against
    the district shares.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.
        turnout_model (TurnoutModel): ``"equal"`` for equal district weights or ``"observed"`` for
            observed two-party turnout weights.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: District shares with the input shape and
            reference shares with shape ``(..., 1)``. Zero-turnout district shares and references
            for rows with no votes are ``NaN``.

    Raises:
        ValueError: If the vote arrays are invalid or ``turnout_model`` is unsupported.
    """
    party, opposition = _paired(party_votes, opposition_votes)
    shares = _partisan_math.district_vote_shares(party, opposition)
    if turnout_model == "equal":
        valid = np.isfinite(shares)
        reference = _divide(
            np.sum(shares, axis=-1, where=valid, keepdims=True),
            np.sum(valid, axis=-1, keepdims=True),
        )
    elif turnout_model == "observed":
        reference = _divide(
            np.sum(party, axis=-1, keepdims=True),
            np.sum(party + opposition, axis=-1, keepdims=True),
        )
    else:
        raise ValueError("turnout_model must be 'equal' or 'observed'")
    return shares, reference


def efficiency_gap(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Return the wasted-vote efficiency gap from the party's point of view.

    Losing votes and votes cast for a winner beyond the minimum needed to win are "wasted." If
    :math:`W_p` and :math:`W_o` are the party and opposition totals, this function returns
    :math:`(W_o - W_p) / \sum_i (p_i + o_i)`. Positive values indicate an advantage for the party;
    negative values indicate an opposition advantage.

    The winning threshold is chosen per district. When both tallies are integral (checked exactly
    on the float values), it is the integer threshold :math:`\lfloor (p_i + o_i)/2 \rfloor + 1`;
    otherwise it is :math:`(p_i + o_i)/2`, so a winner wastes only its votes beyond half the
    turnout. Fractional tallies arise from, for example, areal disaggregation. A tied district
    elects nobody, so both sides waste all of their votes and the district contributes zero to the
    differential. A row with no votes returns ``NaN``.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The efficiency gap for each row preceding the district
            axis, or a scalar for one-dimensional inputs.

    References:
        - Bernstein and Duchin, "A Formula Goes to Court: Partisan Gerrymandering and the Efficiency
            Gap," Notices of the AMS 64 (2017), 1020-1024.
            https://arxiv.org/abs/1705.10812
        - Stephanopoulos and McGhee, "Partisan Gerrymandering and the Efficiency Gap," University
            of Chicago Law Review 82 (2015), 831-900.
            https://lawreview.uchicago.edu/online-archive/partisan-gerrymandering-and-efficiency-gap
    """
    party, opposition = _paired(party_votes, opposition_votes)
    total = party + opposition
    # The integer threshold would let a fractional winner waste a negative number of votes, so
    # districts with any fractional tally use the exact half-turnout threshold instead.
    integral = (party == np.floor(party)) & (opposition == np.floor(opposition))
    threshold = np.where(integral, np.floor(total / 2) + 1, total / 2)
    party_waste = np.where(party > opposition, party - threshold, party)
    opposition_waste = np.where(opposition > party, opposition - threshold, opposition)
    return _divide(
        np.sum(opposition_waste - party_waste, axis=-1),
        np.sum(total, axis=-1),
    )


def simplified_efficiency_gap(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Return the seat-vote formula commonly used for the efficiency gap.

    For party seat share :math:`S` and aggregate vote share :math:`V`, the formula is
    :math:`S - 2V + 1/2`. It is the familiar algebraic reduction of the wasted-vote definition
    under equal district turnout and a 50-percent winning threshold. This implementation's exact
    integer-vote threshold can produce a small difference even with equal turnout. Unequal turnout
    can produce a larger difference, so use :func:`efficiency_gap` when the original wasted-vote
    definition is required.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The simplified gap for each row preceding the district
            axis, or a scalar for one-dimensional inputs.

    References:
        - Bernstein and Duchin, "A Formula Goes to Court: Partisan Gerrymandering and the Efficiency
            Gap," Notices of the AMS 64 (2017), 1020-1024.
            https://arxiv.org/abs/1705.10812
        - Stephanopoulos and McGhee, "Partisan Gerrymandering and the Efficiency Gap," University
            of Chicago Law Review 82 (2015), 831-900.
            https://lawreview.uchicago.edu/online-archive/partisan-gerrymandering-and-efficiency-gap
    """
    party, opposition = _paired(party_votes, opposition_votes)
    seat_share = np.mean(party > opposition, axis=-1)
    return seat_share + 0.5 - 2 * overall_vote_share(party, opposition)


def mean_median(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Return the party's mean-median partisan-symmetry score.

    For district vote shares :math:`v_i`, the score is
    :math:`\operatorname{median}(v_i) - \operatorname{mean}(v_i)`. It is vote-denominated and can
    be viewed as a horizontal displacement of the uniform-swing seats-votes curve from
    :math:`(1/2, 1/2)`. The mean weights districts equally, not by turnout. Positive and negative
    values indicate the direction of asymmetry, but DeFord et al. show that the sign need not
    identify the party favored by a map. Any zero-turnout district makes the result ``NaN``.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The mean-median score for each row preceding the district
            axis, or a scalar for one-dimensional inputs.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023). https://doi.org/10.1017/pan.2021.49
        - Grofman, "Measures of Bias and Proportionality in Seats-Votes Relationships," Political
            Methodology 9 (1983), 295-327. https://www.jstor.org/stable/25791195
    """
    shares = district_vote_shares(party_votes, opposition_votes)
    return np.median(shares, axis=-1) - np.mean(shares, axis=-1)


def partisan_bias(
    party_votes: ArrayLike,
    opposition_votes: ArrayLike,
    *,
    turnout_model: TurnoutModel = "equal",
) -> NDArray[np.float64] | np.float64:
    r"""Return the party's seat-denominated partisan bias at 50 percent.

    Partisan bias is :math:`\gamma(1/2) - 1/2`, where :math:`\gamma` is the seats-votes curve
    induced by uniform partisan swing. In this implementation, districts above the reference vote
    share are seats at 50 percent, districts within numerical tolerance of the reference contribute
    half a seat, and zero-turnout districts are excluded. Positive values mean the party would win
    more than half the seats at the modeled 50-percent vote point.

    ``turnout_model="equal"`` uses the arithmetic mean of valid district shares as the reference,
    matching the usual equal-turnout construction. ``turnout_model="observed"`` instead uses the
    aggregate, turnout-weighted two-party vote share. Under both models, every district still has
    equal weight in the seat count.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.
        turnout_model (TurnoutModel, optional): How to locate the observed election on the vote
            axis. Defaults to ``"equal"``.
            ``"equal"`` gives districts equal weight; ``"observed"`` weights their shares by
            two-party turnout.

    Returns:
        NDArray[np.float64] | np.float64: Partisan bias for each row preceding the district axis, or
            a scalar for one-dimensional inputs. A row with no valid district shares returns
            ``NaN``.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023). https://doi.org/10.1017/pan.2021.49
        - Katz, King, and Rosenblatt, "Theoretical Foundations and Empirical Evaluations of Partisan
            Fairness in District-Based Democracies," American Political Science Review 114 (2020),
            164-178. https://doi.org/10.1017/S000305541900056X
    """
    shares, reference = _partisan_shares_and_reference(party_votes, opposition_votes, turnout_model)
    valid = np.isfinite(shares)
    difference = shares - reference
    seats_at_half = np.sum(difference > _PARTISAN_BIAS_TIE_TOLERANCE, axis=-1) + 0.5 * np.sum(
        valid & (np.abs(difference) <= _PARTISAN_BIAS_TIE_TOLERANCE), axis=-1
    )
    return _divide(seats_at_half, np.sum(valid, axis=-1)) - 0.5


def partisan_gini(
    party_votes: ArrayLike,
    opposition_votes: ArrayLike,
    *,
    turnout_model: TurnoutModel = "equal",
) -> NDArray[np.float64] | np.float64:
    r"""Return the unsigned partisan Gini score under uniform partisan swing.

    Partisan Gini is the area between the seats-votes curve and its reflection through
    :math:`(1/2, 1/2)`. It measures deviation from partisan symmetry over the full modeled curve,
    rather than at one vote share. The score is nonnegative and zero exactly when the constructed
    curve equals its reflection. Because it is unsigned, it does not identify which party benefits.

    ``turnout_model="equal"`` anchors uniform swing at the arithmetic mean of valid district vote
    shares, as in DeFord et al. ``turnout_model="observed"`` anchors it at the aggregate,
    turnout-weighted two-party vote share. Districts still have equal weight when constructing the
    seats-votes curve. Swing breakpoints are clipped to the modeled vote-share interval ``[0, 1]``.
    A zero-turnout district makes the result ``NaN`` because the complete curve cannot be
    constructed from an undefined district share.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.
        turnout_model (TurnoutModel, optional): How to locate the observed election on the vote
            axis. Defaults to ``"equal"``.
            ``"equal"`` gives districts equal weight; ``"observed"`` weights their shares by
            two-party turnout.

    Returns:
        NDArray[np.float64] | np.float64: The partisan Gini score for each row preceding the
            district axis, or a scalar for one-dimensional inputs.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023). https://doi.org/10.1017/pan.2021.49
        - Grofman, "Measures of Bias and Proportionality in Seats-Votes Relationships," Political
            Methodology 9 (1983), 295-327. https://www.jstor.org/stable/25791195
        - Katz, King, and Rosenblatt, "Theoretical Foundations and Empirical Evaluations of Partisan
            Fairness in District-Based Democracies," American Political Science Review 114 (2020),
            164-178. https://doi.org/10.1017/S000305541900056X
    """
    shares, reference = _partisan_shares_and_reference(party_votes, opposition_votes, turnout_model)
    descending = np.flip(np.sort(shares, axis=-1), axis=-1)
    seats_votes = _partisan_math.swing_breakpoints(descending, reference)
    reflected = 1 - np.flip(seats_votes, axis=-1)
    return np.mean(np.abs(seats_votes - reflected), axis=-1)


def _election_stack(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Validate matching vote arrays with nonempty election and district axes.

    The final axis represents districts and the penultimate axis represents elections. Earlier
    axes, if present, are batch axes. The helper preserves all axes and performs no reshaping.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies with elections on the penultimate
            axis.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        tuple[NDArray[np.float64], NDArray[np.float64]]: Party and opposition tallies as finite,
            nonnegative ``float64`` arrays.

    Raises:
        ValueError: If the vote arrays are invalid, have different shapes, lack an election axis,
            or have an empty election axis.
    """
    party, opposition = _paired(party_votes, opposition_votes)
    if party.ndim < 2 or party.shape[-2] == 0:
        raise ValueError("cross-election scores require a nonempty election axis")
    return party, opposition


def competitive_contests(
    party_votes: ArrayLike,
    opposition_votes: ArrayLike,
    *,
    vote_share_margin: float = 0.03,
) -> NDArray[np.intp] | np.intp:
    r"""Count election-district contests within an open interval around 50 percent.

    A contest is counted when the party's two-party share :math:`v` satisfies
    :math:`1/2-d < v < 1/2+d`, where :math:`d` is ``vote_share_margin``. The endpoints are
    excluded. The function counts contests, not unique districts, so a district can contribute
    once for each supplied election. Zero-turnout contests are excluded.

    This is an operational competitiveness count rather than a canonical partisan-symmetry score.
    A threshold is a modeling choice and should not be interpreted as a universal definition of a
    competitive district. DeFord et al. separately discuss the jurisdiction-level competitiveness
    caveat for partisan-symmetry analysis.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.
        vote_share_margin (float, optional): Half-width of the open competitive interval. Must lie
            in :math:`[0, 1/2]`. Defaults to 0.03.

    Returns:
        NDArray[np.intp] | np.intp: The number of competitive election-district contests for each
            preceding batch row, or a scalar when no batch axes are present.

    Raises:
        ValueError: If the margin or vote arrays are invalid.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023), Section 1.2.3. https://doi.org/10.1017/pan.2021.49
    """
    if not isinstance(vote_share_margin, Real) or isinstance(vote_share_margin, bool):
        raise ValueError("vote_share_margin must be a real number")
    if not np.isfinite(vote_share_margin) or not 0 <= vote_share_margin <= 0.5:
        raise ValueError("vote_share_margin must be finite and between zero and one half")
    party, opposition = _election_stack(party_votes, opposition_votes)
    shares = district_vote_shares(party, opposition)
    close = (shares > 0.5 - vote_share_margin) & (shares < 0.5 + vote_share_margin)
    return np.sum(close, axis=(-2, -1))


def party_wins_by_district(party_votes: ArrayLike, opposition_votes: ArrayLike) -> NDArray[np.intp]:
    r"""Count the party's strict wins in each district across elections.

    This descriptive tally preserves the district axis and sums over the election axis. Ties and
    zero-turnout results are not wins. It is an operational multi-election summary, not a named
    partisan-fairness metric in the literature.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.intp]: An integer array indexed by any batch axes and then by district.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023), for the strict two-party seat-outcome convention.
            https://doi.org/10.1017/pan.2021.49
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    return np.sum(party > opposition, axis=-2)


def swing_districts(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.intp] | np.intp:
    r"""Count districts that are not strict wins for one side in every election.

    A district is classified as swing unless the party strictly wins every supplied election or
    the opposition strictly wins every supplied election. Consequently, a tied or zero-turnout
    result makes a district swing. This is a reproducible GerryTools classification, not a claim
    that the literature has a universal election set or threshold for defining a swing district.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.intp] | np.intp: The number of swing districts for each preceding batch row, or a
            scalar without batch axes.
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    stable_party = np.all(party > opposition, axis=-2)
    stable_opposition = np.all(opposition > party, axis=-2)
    return np.sum(~stable_party & ~stable_opposition, axis=-1)


def party_districts(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.intp] | np.intp:
    r"""Count districts strictly won by the party in *every* supplied election.

    For example, suppose three elections have district outcomes ``A=(W, W, L, W)``,
    ``B=(W, L, T, W)``, and ``C=(W, W, W, L)``. This returns 1 because only the first district is
    a strict party win in every election.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.intp] | np.intp: The number of stable party districts for each preceding batch
            row, or a scalar without batch axes.
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    return np.sum(np.all(party > opposition, axis=-2), axis=-1)


def opposition_party_districts(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.intp] | np.intp:
    r"""Count districts strictly won by the opposition in every supplied election.

    This is the opposition-side counterpart to :func:`party_districts`. A district is counted
    only when :math:`o_{ei} > p_{ei}` for every supplied election :math:`e`. A tie or zero-turnout
    contest therefore prevents the district from being classified as stable for the opposition.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.intp] | np.intp: The number of stable opposition districts for each preceding
            batch row, or a scalar without batch axes.
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    return np.sum(np.all(opposition > party, axis=-2), axis=-1)


def aggregate_seats(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.intp] | np.intp:
    r"""Count strict party wins across all supplied elections and districts.

    This is the sum of the binary election-district outcomes. It is useful as an aggregate tally,
    but elections with different offices, electorates, or years still receive equal weight. Ties
    and zero-turnout contests contribute zero.

    For example, suppose three elections have district outcomes ``A=(W, W, L, W)``,
    ``B=(W, L, T, W)``, and ``C=(W, W, W, L)``. This returns 8 because the party strictly wins
    8 of the 12 election-district contests.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.intp] | np.intp: The aggregate count for each preceding batch row, or a scalar
            without batch axes.
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    return np.sum(party > opposition, axis=(-2, -1))


def mean_signed_seat_vote_gap(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Average the party's signed seat-share minus vote-share gap over elections.

    For election :math:`e`, the gap is :math:`S_e - V_e`, where :math:`S_e` is the share of
    districts strictly won and :math:`V_e` is the aggregate two-party vote share. The function
    gives each election equal weight. Positive values indicate average overrepresentation relative
    to a proportional seats-votes benchmark; negative values indicate underrepresentation.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The mean signed gap for each preceding batch row, or a
            scalar without batch axes.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023). https://doi.org/10.1017/pan.2021.49
        - Duchin and Schoenbach, "Redistricting for Proportionality," The Forum 20 (2022),
            371-393. https://doi.org/10.1515/for-2022-2064
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
            Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    differences = np.mean(party > opposition, axis=-1) - overall_vote_share(party, opposition)
    return np.mean(differences, axis=-1)


def mean_absolute_seat_vote_gap(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Average the absolute seat-share minus vote-share gap over elections.

    For each election, this computes :math:`|S_e - V_e|` and then gives every election equal
    weight. Zero means exact proportionality in every supplied election; larger values mean more
    seat-vote disproportionality without assigning it a partisan direction.

    Args:
        party_votes (ArrayLike): Nonnegative tallies with elections on the penultimate axis and
            districts last.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The mean absolute gap for each preceding batch row, or a
            scalar without batch axes.

    References:
        - DeFord et al., "Implementing Partisan Symmetry: Problems and Paradoxes," Political
            Analysis 31 (2023). https://doi.org/10.1017/pan.2021.49
        - Duchin and Schoenbach, "Redistricting for Proportionality," The Forum 20 (2022),
            371-393. https://doi.org/10.1515/for-2022-2064
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
            Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """
    party, opposition = _election_stack(party_votes, opposition_votes)
    differences = np.mean(party > opposition, axis=-1) - overall_vote_share(party, opposition)
    return np.mean(np.abs(differences), axis=-1)


def eguia(
    party_votes: ArrayLike,
    opposition_votes: ArrayLike,
    region_party_votes: ArrayLike,
    region_opposition_votes: ArrayLike,
    region_populations: ArrayLike,
) -> NDArray[np.float64] | np.float64:
    r"""Return Eguia's jurisdictional partisan-advantage score for supplied regions.

    The benchmark is the share of the regional population living in regions where the party
    strictly wins the two-party vote. If :math:`S` is the party's district seat share and
    :math:`J` is that population-weighted regional share, the score is :math:`S-J`. Positive values
    indicate that the districting plan awards the party more seats than the regional benchmark;
    negative values indicate fewer.

    Eguia defines the benchmark using political jurisdictions such as counties and towns. This
    function generalizes the computation to caller-supplied regions, so its interpretation depends
    on what those regions represent. It does not infer region populations from vote turnout.
    District and region ties count as wins for neither side.

    Args:
        party_votes (ArrayLike): Nonnegative party tallies with districts on the last axis.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.
        region_party_votes (ArrayLike): Nonnegative party tallies with regions on the last axis.
        region_opposition_votes (ArrayLike): Nonnegative opposition tallies matching
            ``region_party_votes``.
        region_populations (ArrayLike): Nonnegative region populations, broadcastable to the region
            vote shape.

    Returns:
        NDArray[np.float64] | np.float64: The Eguia score over compatible district and region
            election or batch axes. A row whose regions have zero total population returns ``NaN``.

    Raises:
        ValueError: If vote or population arrays are invalid or have incompatible shapes.

    References:
        - Eguia, "A Measure of Partisan Advantage in Redistricting," Election Law Journal 21
            (2022), 84-103. https://doi.org/10.1089/elj.2020.0691
        - Duchin et al., "Locating the Representational Baseline: Republicans in Massachusetts,"
            Election Law Journal 18 (2019), 388-401. https://arxiv.org/abs/1810.09051
    """
    party, opposition = _paired(party_votes, opposition_votes)
    region_party, region_opposition = _matching(
        region_party_votes,
        "region_party_votes",
        region_opposition_votes,
        "region_opposition_votes",
    )
    populations = _nonnegative(region_populations, "region_populations")
    if populations.shape[-1] != region_party.shape[-1]:
        raise ValueError("region_populations must have the same region axis as region votes")
    try:
        populations = np.broadcast_to(populations, region_party.shape)
    except ValueError as error:
        raise ValueError("region_populations cannot be broadcast to the region votes") from error

    seat_share = np.mean(party > opposition, axis=-1)
    ideal = _divide(
        np.sum((region_party > region_opposition) * populations, axis=-1),
        np.sum(populations, axis=-1),
    )
    try:
        seat_share, ideal = np.broadcast_arrays(seat_share, ideal)
    except ValueError as error:
        raise ValueError("district and region election axes are incompatible") from error
    return seat_share - ideal


def _positive_population_totals(populations: ArrayLike) -> NDArray[np.float64]:
    """Validate nonnegative district populations with positive plan totals.

    Each row preceding the district axis is treated as a separate plan or batch row. Individual
    districts may have zero population, but every row must have a strictly positive total.

    Args:
        populations (ArrayLike): Nonnegative populations with districts on the last axis.

    Returns:
        NDArray[np.float64]: Populations as a finite, nonnegative ``float64`` array with the input
            shape.

    Raises:
        ValueError: If the input fails :func:`_nonnegative` validation or any row has zero total
            population.
    """
    values = _nonnegative(populations, "populations")
    if np.any(np.sum(values, axis=-1) == 0):
        raise ValueError("populations must have positive totals")
    return values


def population_deviations(populations: ArrayLike) -> NDArray[np.float64]:
    r"""Return each district's signed proportional deviation from ideal population.

    For district population :math:`P_i` and ideal population :math:`P^* = (\sum_i P_i)/k`, the
    result is :math:`(P_i-P^*)/P^*`. Positive values identify districts above ideal and negative
    values identify districts below ideal. Each plan or batch row must have positive total
    population.

    Args:
        populations (ArrayLike): Nonnegative district populations. The last axis represents
            districts.

    Returns:
        NDArray[np.float64]: An array with the input shape containing signed deviations as
            proportions of ideal.

    References:
        - Evenwel v. Abbott, 578 U.S. 54 (2016), discussing equal population and maximum
            population deviation in state and local legislative districting.
            https://www.govinfo.gov/app/details/USREPORTS-578
        - Duchin and Walch, eds., Political Geometry: Rethinking Redistricting in the US with Math,
            Law, and Everything In Between (2022). https://doi.org/10.1007/978-3-319-69161-9
    """
    values = _positive_population_totals(populations)
    ideal = np.mean(values, axis=-1, keepdims=True)
    return _divide(values - ideal, ideal)


def max_absolute_population_deviation(
    populations: ArrayLike, *, relative_to_ideal: bool = False
) -> NDArray[np.float64] | np.float64:
    r"""Return the largest one-district absolute deviation from ideal population.

    The raw result is :math:`\max_i |P_i-P^*|`. With ``relative_to_ideal=True``, it is divided by
    ideal population :math:`P^*`. This is the worst individual departure from ideal. It is not the
    conventional plan-wide "maximum population deviation," which adds the largest positive and
    negative percentage deviations and is computed by :func:`max_population_deviation`.

    Args:
        populations (ArrayLike): Nonnegative district populations. The last axis represents
            districts.
        relative_to_ideal (bool, optional): Return a proportion of ideal population instead of a
            population count. Defaults to False.

    Returns:
        NDArray[np.float64] | np.float64: The largest absolute deviation for each preceding batch
            row, or a scalar without batch axes.

    References:
        - Evenwel v. Abbott, 578 U.S. 54, 60 n.2 (2016), defining the conventional plan-wide
            maximum population deviation for comparison.
            https://www.govinfo.gov/app/details/USREPORTS-578
        - Duchin and Walch, eds., Political Geometry: Rethinking Redistricting in the US with Math,
            Law, and Everything In Between (2022). https://doi.org/10.1007/978-3-319-69161-9
    """
    values = _positive_population_totals(populations)
    ideal = np.mean(values, axis=-1)
    deviation = np.max(np.abs(values - np.expand_dims(ideal, axis=-1)), axis=-1)
    return _divide(deviation, ideal) if relative_to_ideal else deviation


def max_population_deviation(
    populations: ArrayLike, *, relative_to_ideal: bool = False
) -> NDArray[np.float64] | np.float64:
    r"""Return the conventional maximum population deviation of a plan.

    The raw result is :math:`\max_i P_i-\min_i P_i`. With ``relative_to_ideal=True``, it is divided
    by the ideal population. The relative form equals the largest positive percentage deviation
    plus the magnitude of the largest negative percentage deviation, which is the convention
    described in Evenwel v. Abbott. This plan-wide range differs from the worst single-district
    absolute deviation returned by :func:`max_absolute_population_deviation`.

    Args:
        populations (ArrayLike): Nonnegative district populations. The last axis represents
            districts.
        relative_to_ideal (bool, optional): Return a proportion of ideal population instead of a
            population count. Defaults to False.

    Returns:
        NDArray[np.float64] | np.float64: Maximum population deviation for each preceding batch row,
            or a scalar without batch axes.

    References:
        - Evenwel v. Abbott, 578 U.S. 54, 60 n.2 (2016).
            https://www.govinfo.gov/app/details/USREPORTS-578
        - Karcher v. Daggett, 462 U.S. 725 (1983), discussing population equality for
            congressional districts. https://www.law.cornell.edu/supremecourt/text/462/725
    """
    values = _positive_population_totals(populations)
    deviation = np.max(values, axis=-1) - np.min(values, axis=-1)
    if not relative_to_ideal:
        return deviation
    return _divide(deviation, np.mean(values, axis=-1))


def demographic_shares(
    subgroup_populations: ArrayLike, total_populations: ArrayLike
) -> NDArray[np.float64]:
    r"""Return a subgroup's share of the specified total population in each district.

    The result is :math:`G_i/P_i`. A district with zero total population returns ``NaN``.

    Args:
        subgroup_populations (ArrayLike): Nonnegative subgroup populations by district.
        total_populations (ArrayLike): Matching nonnegative totals that contain the subgroup
            populations.

    Returns:
        NDArray[np.float64]: An array of shares in :math:`[0, 1]`, with ``NaN`` where the
            denominator is zero.

    Raises:
        ValueError: If the population arrays are invalid or a subgroup exceeds its total.
    """
    subgroup, total = _matching(
        subgroup_populations,
        "subgroup_populations",
        total_populations,
        "total_populations",
    )
    if np.any(subgroup > total):
        raise ValueError("subgroup_populations cannot exceed total_populations")
    return _divide(subgroup, total)


def districts_above_threshold(
    subgroup_populations: ArrayLike,
    total_populations: ArrayLike,
    *,
    threshold: float = 0.5,
) -> NDArray[np.intp] | np.intp:
    r"""Count districts whose subgroup share is strictly above a threshold.

    Args:
        subgroup_populations (ArrayLike): Nonnegative subgroup populations by district.
        total_populations (ArrayLike): Matching nonnegative totals that contain the subgroup
            populations.
        threshold (float, optional): Share threshold in :math:`[0, 1]`; the comparison is strictly
            greater than. Defaults to 0.5.

    Returns:
        NDArray[np.intp] | np.intp: The number of districts above the threshold for each preceding
            batch row, or a scalar without batch axes.

    Raises:
        ValueError: If the threshold or population arrays are invalid.
    """
    if not isinstance(threshold, Real) or isinstance(threshold, bool):
        raise ValueError("threshold must be a real number")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be finite and between zero and one")
    shares = demographic_shares(subgroup_populations, total_populations)
    return np.sum(shares > threshold, axis=-1)


def schwartzberg(polsby_popper_scores: ArrayLike) -> NDArray[np.float64]:
    r"""Derive Schwartzberg compactness from Polsby-Popper compactness.

    For a positive Polsby-Popper score :math:`PP`, this function returns
    :math:`1/\sqrt{PP}`. A circle has value 1, and larger values indicate a longer perimeter
    relative to area and therefore less compactness. This direction is the inverse of
    Polsby-Popper, for which 1 is best and values approach 0 as contour complexity increases.

    The formula assumes that the Polsby-Popper values were computed from compatible area and
    perimeter measurements. Contour-based scores are sensitive to projection, boundary detail,
    water, and geographic resolution; they should not be treated as standalone measures of plan
    fairness.

    Args:
        polsby_popper_scores (ArrayLike): Positive Polsby-Popper scores no greater than 1, within
            numerical tolerance.

    Returns:
        NDArray[np.float64]: Schwartzberg scores with the same shape as the input.

    Raises:
        ValueError: If a Polsby-Popper score is nonpositive, greater than one, or nonfinite.

    References:
        - Schwartzberg, "Reapportionment, Gerrymanders, and the Notion of Compactness," Minnesota
            Law Review 50 (1966), 443. https://doi.org/10.24926/265535.2601
        - Polsby and Popper, "The Third Criterion: Compactness as a Procedural Safeguard Against
            Partisan Gerrymandering," Yale Law & Policy Review 9 (1991), 301-353.
            `Article record <https://openyls.law.yale.edu/handle/20.500.13051/17448>`_.
        - Duchin and Tenner, "Discrete Geometry for Electoral Geography," Political Geography 109
            (2024), 103040. https://doi.org/10.1016/j.polgeo.2023.103040
    """
    scores = _nonnegative(polsby_popper_scores, "polsby_popper_scores")
    if np.any(scores == 0):
        raise ValueError("polsby_popper_scores must be positive")
    if np.any(scores > 1 + 1e-12):
        raise ValueError("polsby_popper_scores cannot exceed one")
    return 1 / np.sqrt(np.minimum(scores, 1))


def disproportionality(
    party_votes: ArrayLike, opposition_votes: ArrayLike
) -> NDArray[np.float64] | np.float64:
    r"""Return the party's signed seat-share minus aggregate vote-share gap.

    Positive values indicate that the party wins a larger share of seats than votes; negative
    values indicate that it wins a smaller share. District ties are not wins.

    Args:
        party_votes (ArrayLike): Nonnegative party vote tallies. The last axis represents districts.
        opposition_votes (ArrayLike): Nonnegative opposition tallies with the same shape as
            ``party_votes``.

    Returns:
        NDArray[np.float64] | np.float64: The disproportionality score for each row preceding the
            district axis, or a scalar for one-dimensional inputs.
    """
    party_votes, opposition_votes = _paired(party_votes, opposition_votes)
    seat_share = np.mean(party_votes > opposition_votes, axis=-1)
    vote_share = overall_vote_share(party_votes, opposition_votes)
    return seat_share - vote_share
