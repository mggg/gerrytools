use super::*;

/// Build a district tally with ids `0..n` from column slices in column order.
fn tally(columns: &[&[f64]]) -> DistrictTable {
    let district_count = columns[0].len();
    let districts = (0..district_count as u16).collect();
    let values = columns
        .iter()
        .flat_map(|column| column.iter().copied())
        .collect();
    DistrictTable::new(districts, values, columns.len())
}

fn plan_value(metric: &SharedTallyMetric, tally: &DistrictTable) -> f64 {
    match metric.score(tally).unwrap() {
        MetricScore::Plan(table) => table.values()[0],
        MetricScore::District(_) => panic!("expected a plan-shaped result"),
    }
}

fn district_values(metric: &SharedTallyMetric, tally: &DistrictTable) -> Vec<f64> {
    match metric.score(tally).unwrap() {
        MetricScore::District(table) => table.column(0).unwrap().to_vec(),
        MetricScore::Plan(_) => panic!("expected a district-shaped result"),
    }
}

fn paired(kind: &str, turnout_model: &str, party: &[f64], opposition: &[f64]) -> f64 {
    let metric = SharedTallyMetric::paired(kind, 0, 1, turnout_model).unwrap();
    plan_value(&metric, &tally(&[party, opposition]))
}

fn efficiency_gap(party: &[f64], opposition: &[f64]) -> f64 {
    paired("efficiency_gap", "equal", party, opposition)
}

#[test]
fn efficiency_gap_uses_the_exact_half_threshold_for_fractional_tallies() {
    // Vote shares: each district wastes 0.1 for the winner and 0.4 for the loser, giving the
    // classic (0.4 - 0.1) / 1.0 = 0.3 even though 0.6 + 0.4 == 1.0 exactly in f64.
    let gap = efficiency_gap(&[0.6; 10], &[0.4; 10]);
    assert!((gap - 0.3).abs() < 1e-12, "got {gap}");
}

#[test]
fn efficiency_gap_keeps_the_whole_vote_threshold() {
    // 100 whole votes need 51 to win: party wastes 60 - 51 = 9, opposition wastes 40.
    let gap = efficiency_gap(&[60.0], &[40.0]);
    assert!((gap - 0.31).abs() < 1e-12, "got {gap}");
}

#[test]
fn efficiency_gap_ties_waste_everything_on_both_sides() {
    let gap = efficiency_gap(&[50.0, 60.0], &[50.0, 40.0]);
    // The tied district contributes zero; the decided district contributes 40 - 9 = 31.
    assert!((gap - 31.0 / 200.0).abs() < 1e-12, "got {gap}");
}

#[test]
fn mean_median_uses_the_middle_share_for_odd_district_counts() {
    // Shares [0.6, 0.45, 0.5]: median 0.5, mean 1.55 / 3.
    let value = paired(
        "mean_median",
        "equal",
        &[60.0, 45.0, 50.0],
        &[40.0, 55.0, 50.0],
    );
    assert!((value - (0.5 - 1.55 / 3.0)).abs() < 1e-12, "got {value}");
}

#[test]
fn mean_median_averages_the_middle_shares_for_even_district_counts() {
    // Shares [0.6, 0.4, 0.4, 0.4]: median (0.4 + 0.4) / 2 = 0.4, mean 0.45.
    let value = paired(
        "mean_median",
        "equal",
        &[60.0, 40.0, 40.0, 40.0],
        &[40.0, 60.0, 60.0, 60.0],
    );
    assert!((value + 0.05).abs() < 1e-12, "got {value}");
}

#[test]
fn disproportionality_uses_aggregate_vote_share() {
    let value = paired("disproportionality", "equal", &[90.0, 1.0], &[10.0, 9.0]);
    let expected = 0.5 - 91.0 / 110.0;
    assert!((value - expected).abs() < 1e-12, "got {value}");
}

#[test]
fn partisan_bias_awards_half_seats_within_the_tie_tolerance() {
    // Mirrors the Python-pinned tolerance cases: offsets sum to zero, so the equal-turnout
    // reference share is exactly 0.5 and each offset is compared against the 1e-9 tolerance.
    let cases = [
        (vec![0.5e-9, -0.25e-9, -0.25e-9], 0.0),
        (vec![1.5e-9, -0.75e-9, -0.75e-9], 1.0 / 6.0),
    ];
    for (offsets, expected) in cases {
        let party: Vec<f64> = offsets.iter().map(|offset| 0.5 + offset).collect();
        let opposition: Vec<f64> = party.iter().map(|share| 1.0 - share).collect();
        let value = paired("partisan_bias", "equal", &party, &opposition);
        assert!((value - expected).abs() < 1e-12, "got {value}");
    }
}

#[test]
fn partisan_bias_supports_both_turnout_models() {
    // Python-pinned: the lopsided first district drags the observed reference share below
    // the equal-turnout mean, flipping the sign of the bias.
    let party = [400.0, 45.0, 60.0];
    let opposition = [600.0, 55.0, 40.0];
    let equal = paired("partisan_bias", "equal", &party, &opposition);
    let observed = paired("partisan_bias", "observed", &party, &opposition);
    assert!((equal + 1.0 / 6.0).abs() < 1e-12, "got {equal}");
    assert!((observed - 1.0 / 6.0).abs() < 1e-12, "got {observed}");
}

#[test]
fn partisan_gini_clips_the_curve_and_reflects_it() {
    // Shares [0.01, 0.99, 0.99], equal-turnout reference 1.99 / 3. Curve values
    // clamp(reference - share + 0.5) over descending shares are [c, c, 1.0] with
    // c = 1.99 / 3 - 0.49 (the last entry clips from ~1.153). Reflection differences are
    // |c - 0|, |c - (1 - c)|, |1 - (1 - c)|, and their mean is exactly 1 / 3.
    let value = paired(
        "partisan_gini",
        "equal",
        &[1.0, 99.0, 99.0],
        &[99.0, 1.0, 1.0],
    );
    assert!((value - 1.0 / 3.0).abs() < 1e-12, "got {value}");
}

#[test]
fn partisan_gini_supports_both_turnout_models() {
    // Python-pinned: symmetric under equal turnout, 9 / 55 under observed turnout.
    let party = [400.0, 60.0];
    let opposition = [600.0, 40.0];
    let equal = paired("partisan_gini", "equal", &party, &opposition);
    let observed = paired("partisan_gini", "observed", &party, &opposition);
    assert!(equal.abs() < 1e-12, "got {equal}");
    assert!((observed - 9.0 / 55.0).abs() < 1e-12, "got {observed}");
}

#[test]
fn population_deviations_and_maxima_match_pinned_values() {
    let populations: &[f64] = &[90.0, 100.0, 110.0];
    let deviations = SharedTallyMetric::population("population_deviations", 0, false).unwrap();
    let values = district_values(&deviations, &tally(&[populations]));
    for (value, expected) in values.iter().zip([-0.1, 0.0, 0.1]) {
        assert!((value - expected).abs() < 1e-12, "got {values:?}");
    }

    let cases = [
        ("max_absolute_population_deviation", false, 10.0),
        ("max_absolute_population_deviation", true, 0.1),
        ("max_population_deviation", false, 20.0),
        ("max_population_deviation", true, 0.2),
    ];
    for (kind, relative, expected) in cases {
        let metric = SharedTallyMetric::population(kind, 0, relative).unwrap();
        let value = plan_value(&metric, &tally(&[populations]));
        assert!((value - expected).abs() < 1e-12, "{kind}: got {value}");
    }
}

#[test]
fn demographic_shares_and_threshold_counts_match_pinned_values() {
    let subgroup: &[f64] = &[45.0, 60.0, 0.0];
    let total: &[f64] = &[90.0, 100.0, 0.0];
    let shares = SharedTallyMetric::demographic("demographic_shares", 0, 1, 0.0).unwrap();
    let values = district_values(&shares, &tally(&[subgroup, total]));
    assert_eq!(&values[..2], &[0.5, 0.6]);
    assert!(values[2].is_nan());

    let above = SharedTallyMetric::demographic("districts_above_threshold", 0, 1, 0.5).unwrap();
    assert_eq!(plan_value(&above, &tally(&[subgroup, total])), 1.0);
}

#[test]
fn cross_election_kinds_match_pinned_values() {
    // Two elections over four districts (Python-pinned): district 0 is stably party, district 1
    // stably opposition, district 2 swings, and district 3 ties the first election.
    let party_one: &[f64] = &[60.0, 48.0, 40.0, 50.0];
    let party_two: &[f64] = &[55.0, 45.0, 60.0, 51.0];
    let opposition_one: &[f64] = &[40.0, 52.0, 60.0, 50.0];
    let opposition_two: &[f64] = &[45.0, 55.0, 40.0, 49.0];
    let tally = tally(&[party_one, party_two, opposition_one, opposition_two]);
    let metric = |kind: &str, points_within: f64| {
        SharedTallyMetric::cross_election(kind, vec![0, 1], vec![2, 3], points_within).unwrap()
    };

    assert_eq!(
        district_values(&metric("party_wins_by_district", 0.0), &tally),
        vec![2.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(
        plan_value(&metric("competitive_contests", 0.03), &tally),
        3.0
    );
    assert_eq!(plan_value(&metric("swing_districts", 0.0), &tally), 2.0);
    assert_eq!(plan_value(&metric("party_districts", 0.0), &tally), 1.0);
    assert_eq!(
        plan_value(&metric("opposition_party_districts", 0.0), &tally),
        1.0
    );
    assert_eq!(plan_value(&metric("aggregate_seats", 0.0), &tally), 4.0);

    // Seat shares are [0.25, 0.75]; vote shares are 198/400 and 211/400.
    let signed = plan_value(&metric("mean_signed_seat_vote_gap", 0.0), &tally);
    let absolute = plan_value(&metric("mean_absolute_seat_vote_gap", 0.0), &tally);
    assert!((signed + 0.01125).abs() < 1e-12, "got {signed}");
    assert!((absolute - 0.23375).abs() < 1e-12, "got {absolute}");
}

#[test]
fn eguia_constructor_validates_the_benchmark() {
    for benchmark in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(matches!(
            SharedTallyMetric::eguia(0, 1, benchmark),
            Err(Error::InvalidInput(message))
                if message == "Eguia benchmark must be finite and between zero and one"
        ));
    }
    assert!(SharedTallyMetric::eguia(0, 1, 0.0).is_ok());
    assert!(SharedTallyMetric::eguia(0, 1, 1.0).is_ok());
}
