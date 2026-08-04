// Parity with gerrytools/scoring/formulas.py is contractual: the Python mirror is
// differential-tested against these implementations, so convention changes must land in both.
use crate::{DistrictTable, Error, MetricScore, PlanTable, Result, TableShape};

type ElectionColumns<'a> = (Vec<&'a [f64]>, Vec<&'a [f64]>);
const PARTISAN_BIAS_TIE_TOLERANCE: f64 = 1e-9;

#[derive(Clone, Copy)]
pub(crate) enum TurnoutModel {
    Equal,
    Observed,
}

impl TurnoutModel {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "equal" => Ok(Self::Equal),
            "observed" => Ok(Self::Observed),
            _ => Err(Error::InvalidInput(
                "turnout model must be 'equal' or 'observed'".into(),
            )),
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) enum PairedKind {
    DistrictVoteShares,
    DistrictWins,
    Seats,
    OverallVoteShare,
    EfficiencyGap,
    SimplifiedEfficiencyGap,
    MeanMedian,
    Disproportionality,
    PartisanBias(TurnoutModel),
    PartisanGini(TurnoutModel),
}

#[derive(Clone, Copy)]
pub(crate) enum PopulationKind {
    Deviations,
    MaxAbsolute { relative: bool },
    Max { relative: bool },
}

#[derive(Clone, Copy)]
pub(crate) enum DemographicKind {
    Shares,
    DistrictsAboveThreshold { threshold: f64 },
}

#[derive(Clone, Copy)]
pub(crate) enum CrossElectionKind {
    CompetitiveContests { points_within: f64 },
    PartyWinsByDistrict,
    SwingDistricts,
    PartyDistricts,
    OppositionPartyDistricts,
    AggregateSeats,
    MeanSignedSeatVoteGap,
    MeanAbsoluteSeatVoteGap,
}

#[derive(Clone)]
pub(crate) enum SharedTallyMetric {
    Projection(Vec<usize>),
    Eguia {
        party: usize,
        opposition: usize,
        benchmark: f64,
    },
    Paired {
        kind: PairedKind,
        party: usize,
        opposition: usize,
    },
    Population {
        kind: PopulationKind,
        population: usize,
    },
    Demographic {
        kind: DemographicKind,
        subgroup: usize,
        total: usize,
    },
    CrossElection {
        kind: CrossElectionKind,
        party: Vec<usize>,
        opposition: Vec<usize>,
    },
}

impl SharedTallyMetric {
    /// Validated Eguia constructor, shared by the engine and Python registration paths.
    pub(crate) fn eguia(party: usize, opposition: usize, benchmark: f64) -> Result<Self> {
        if !benchmark.is_finite() || !(0.0..=1.0).contains(&benchmark) {
            return Err(Error::InvalidInput(
                "Eguia benchmark must be finite and between zero and one".into(),
            ));
        }
        Ok(Self::Eguia {
            party,
            opposition,
            benchmark,
        })
    }

    pub(crate) fn paired(
        kind: &str,
        party: usize,
        opposition: usize,
        turnout_model: &str,
    ) -> Result<Self> {
        let kind = match kind {
            "district_vote_shares" => PairedKind::DistrictVoteShares,
            "district_wins" => PairedKind::DistrictWins,
            "seats" => PairedKind::Seats,
            "overall_vote_share" => PairedKind::OverallVoteShare,
            "efficiency_gap" => PairedKind::EfficiencyGap,
            "simplified_efficiency_gap" => PairedKind::SimplifiedEfficiencyGap,
            "mean_median" => PairedKind::MeanMedian,
            "disproportionality" => PairedKind::Disproportionality,
            "partisan_bias" => PairedKind::PartisanBias(TurnoutModel::parse(turnout_model)?),
            "partisan_gini" => PairedKind::PartisanGini(TurnoutModel::parse(turnout_model)?),
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unknown paired formula metric {kind:?}"
                )));
            }
        };
        Ok(Self::Paired {
            kind,
            party,
            opposition,
        })
    }

    pub(crate) fn population(kind: &str, population: usize, relative: bool) -> Result<Self> {
        let kind = match kind {
            "population_deviations" => PopulationKind::Deviations,
            "max_absolute_population_deviation" => PopulationKind::MaxAbsolute { relative },
            "max_population_deviation" => PopulationKind::Max { relative },
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unknown population metric {kind:?}"
                )));
            }
        };
        Ok(Self::Population { kind, population })
    }

    pub(crate) fn demographic(
        kind: &str,
        subgroup: usize,
        total: usize,
        threshold: f64,
    ) -> Result<Self> {
        let kind = match kind {
            "demographic_shares" => DemographicKind::Shares,
            "districts_above_threshold" => {
                if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                    return Err(Error::InvalidInput(
                        "demographic threshold must be finite and between zero and one".into(),
                    ));
                }
                DemographicKind::DistrictsAboveThreshold { threshold }
            }
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unknown demographic metric {kind:?}"
                )));
            }
        };
        Ok(Self::Demographic {
            kind,
            subgroup,
            total,
        })
    }

    pub(crate) fn cross_election(
        kind: &str,
        party: Vec<usize>,
        opposition: Vec<usize>,
        points_within: f64,
    ) -> Result<Self> {
        if party.is_empty() || party.len() != opposition.len() {
            return Err(Error::InvalidInput(
                "cross-election metrics require equal nonempty party and opposition columns".into(),
            ));
        }
        let kind = match kind {
            "competitive_contests" => {
                if !points_within.is_finite() || !(0.0..=0.5).contains(&points_within) {
                    return Err(Error::InvalidInput(
                        "points_within must be finite and between zero and one half".into(),
                    ));
                }
                CrossElectionKind::CompetitiveContests { points_within }
            }
            "party_wins_by_district" => CrossElectionKind::PartyWinsByDistrict,
            "swing_districts" => CrossElectionKind::SwingDistricts,
            "party_districts" => CrossElectionKind::PartyDistricts,
            "opposition_party_districts" => CrossElectionKind::OppositionPartyDistricts,
            "aggregate_seats" => CrossElectionKind::AggregateSeats,
            "mean_signed_seat_vote_gap" => CrossElectionKind::MeanSignedSeatVoteGap,
            "mean_absolute_seat_vote_gap" => CrossElectionKind::MeanAbsoluteSeatVoteGap,
            _ => {
                return Err(Error::InvalidInput(format!(
                    "unknown cross-election metric {kind:?}"
                )));
            }
        };
        Ok(Self::CrossElection {
            kind,
            party,
            opposition,
        })
    }

    pub(crate) fn kind(&self) -> &'static str {
        match self {
            Self::Projection(_) => "tally",
            Self::Eguia { .. } => "eguia",
            Self::Paired { kind, .. } => match kind {
                PairedKind::DistrictVoteShares => "district_vote_shares",
                PairedKind::DistrictWins => "district_wins",
                PairedKind::Seats => "seats",
                PairedKind::OverallVoteShare => "overall_vote_share",
                PairedKind::EfficiencyGap => "efficiency_gap",
                PairedKind::SimplifiedEfficiencyGap => "simplified_efficiency_gap",
                PairedKind::MeanMedian => "mean_median",
                PairedKind::Disproportionality => "disproportionality",
                PairedKind::PartisanBias(_) => "partisan_bias",
                PairedKind::PartisanGini(_) => "partisan_gini",
            },
            Self::Population { kind, .. } => match kind {
                PopulationKind::Deviations => "population_deviations",
                PopulationKind::MaxAbsolute { .. } => "max_absolute_population_deviation",
                PopulationKind::Max { .. } => "max_population_deviation",
            },
            Self::Demographic { kind, .. } => match kind {
                DemographicKind::Shares => "demographic_shares",
                DemographicKind::DistrictsAboveThreshold { .. } => "districts_above_threshold",
            },
            Self::CrossElection { kind, .. } => match kind {
                CrossElectionKind::CompetitiveContests { .. } => "competitive_contests",
                CrossElectionKind::PartyWinsByDistrict => "party_wins_by_district",
                CrossElectionKind::SwingDistricts => "swing_districts",
                CrossElectionKind::PartyDistricts => "party_districts",
                CrossElectionKind::OppositionPartyDistricts => "opposition_party_districts",
                CrossElectionKind::AggregateSeats => "aggregate_seats",
                CrossElectionKind::MeanSignedSeatVoteGap => "mean_signed_seat_vote_gap",
                CrossElectionKind::MeanAbsoluteSeatVoteGap => "mean_absolute_seat_vote_gap",
            },
        }
    }

    pub(crate) fn column_count(&self) -> usize {
        match self {
            Self::Projection(columns) => columns.len(),
            _ => 1,
        }
    }

    pub(crate) fn shape(&self) -> TableShape {
        match self {
            Self::Projection(_) => TableShape::District,
            Self::Paired {
                kind: PairedKind::DistrictVoteShares | PairedKind::DistrictWins,
                ..
            }
            | Self::Population {
                kind: PopulationKind::Deviations,
                ..
            }
            | Self::Demographic {
                kind: DemographicKind::Shares,
                ..
            }
            | Self::CrossElection {
                kind: CrossElectionKind::PartyWinsByDistrict,
                ..
            } => TableShape::District,
            _ => TableShape::Plan,
        }
    }

    pub(crate) fn required_columns(&self) -> Vec<usize> {
        match self {
            Self::Projection(columns) => columns.clone(),
            Self::Eguia {
                party, opposition, ..
            }
            | Self::Paired {
                party, opposition, ..
            } => vec![*party, *opposition],
            Self::Population { population, .. } => vec![*population],
            Self::Demographic {
                subgroup, total, ..
            } => vec![*subgroup, *total],
            Self::CrossElection {
                party, opposition, ..
            } => party.iter().chain(opposition).copied().collect(),
        }
    }

    pub(crate) fn score(&self, tally: &DistrictTable) -> Result<MetricScore> {
        match self {
            Self::Projection(columns) => score_projection(tally, columns),
            Self::Eguia {
                party,
                opposition,
                benchmark,
            } => {
                let (party, opposition) = paired_columns(tally, *party, *opposition)?;
                let wins = party
                    .iter()
                    .zip(opposition)
                    .filter(|(party, opposition)| party > opposition)
                    .count();
                plan_score(tally, wins as f64 / party.len() as f64 - benchmark)
            }
            Self::Paired {
                kind,
                party,
                opposition,
            } => score_paired(tally, *kind, *party, *opposition),
            Self::Population { kind, population } => score_population(tally, *kind, *population),
            Self::Demographic {
                kind,
                subgroup,
                total,
            } => score_demographic(tally, *kind, *subgroup, *total),
            Self::CrossElection {
                kind,
                party,
                opposition,
            } => score_cross_election(tally, *kind, party, opposition),
        }
    }
}

fn score_projection(tally: &DistrictTable, columns: &[usize]) -> Result<MetricScore> {
    let mut values = Vec::with_capacity(columns.len() * tally.district_ids().len());
    for &column in columns {
        values.extend_from_slice(tally.column(column).ok_or_else(|| {
            Error::InvalidInput("tally projection references an out-of-range column".into())
        })?);
    }
    Ok(MetricScore::District(DistrictTable::new(
        tally.district_ids().to_vec(),
        values,
        columns.len(),
    )))
}

fn paired_columns(
    tally: &DistrictTable,
    party: usize,
    opposition: usize,
) -> Result<(&[f64], &[f64])> {
    let party = tally
        .column(party)
        .ok_or_else(|| Error::InvalidInput("party tally column is out of range".into()))?;
    let opposition = tally
        .column(opposition)
        .ok_or_else(|| Error::InvalidInput("opposition tally column is out of range".into()))?;
    if party.is_empty() {
        return Err(Error::InvalidInput(
            "formula metrics require at least one observed district".into(),
        ));
    }
    Ok((party, opposition))
}

fn district_score(tally: &DistrictTable, values: Vec<f64>) -> MetricScore {
    MetricScore::District(DistrictTable::new(tally.district_ids().to_vec(), values, 1))
}

fn plan_score(tally: &DistrictTable, value: f64) -> Result<MetricScore> {
    PlanTable::new(tally.district_ids().to_vec(), vec![value]).map(MetricScore::Plan)
}

fn divide(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        f64::NAN
    }
}

fn vote_shares(party: &[f64], opposition: &[f64]) -> Vec<f64> {
    party
        .iter()
        .zip(opposition)
        .map(|(&party, &opposition)| divide(party, party + opposition))
        .collect()
}

fn overall_vote_share(party: &[f64], opposition: &[f64]) -> f64 {
    let party_total = party.iter().sum();
    let total = party
        .iter()
        .zip(opposition)
        .map(|(party, opposition)| party + opposition)
        .sum();
    divide(party_total, total)
}

fn reference_share(
    shares: &[f64],
    party: &[f64],
    opposition: &[f64],
    turnout_model: TurnoutModel,
) -> f64 {
    match turnout_model {
        TurnoutModel::Equal => {
            let valid: Vec<_> = shares
                .iter()
                .copied()
                .filter(|share| share.is_finite())
                .collect();
            divide(valid.iter().sum(), valid.len() as f64)
        }
        TurnoutModel::Observed => overall_vote_share(party, opposition),
    }
}

fn score_paired(
    tally: &DistrictTable,
    kind: PairedKind,
    party_column: usize,
    opposition_column: usize,
) -> Result<MetricScore> {
    let (party, opposition) = paired_columns(tally, party_column, opposition_column)?;
    match kind {
        PairedKind::DistrictVoteShares => Ok(district_score(tally, vote_shares(party, opposition))),
        PairedKind::DistrictWins => Ok(district_score(
            tally,
            party
                .iter()
                .zip(opposition)
                .map(|(party, opposition)| f64::from(party > opposition))
                .collect(),
        )),
        PairedKind::Seats => plan_score(
            tally,
            party
                .iter()
                .zip(opposition)
                .filter(|(party, opposition)| party > opposition)
                .count() as f64,
        ),
        PairedKind::OverallVoteShare => plan_score(tally, overall_vote_share(party, opposition)),
        PairedKind::EfficiencyGap => {
            let (gap, total) = party.iter().zip(opposition).fold(
                (0.0, 0.0),
                |(gap, total_sum), (&party, &opposition)| {
                    let turnout = party + opposition;
                    // Whole-vote districts need floor(total/2)+1 votes to win; fractional tallies
                    // (vote shares, weighted data) use exactly half. Checked per tally because a
                    // share pair can sum to an exact integer (0.6 + 0.4 == 1.0 in f64).
                    let whole_votes = party == party.floor() && opposition == opposition.floor();
                    let threshold = if whole_votes {
                        (turnout / 2.0).floor() + 1.0
                    } else {
                        turnout / 2.0
                    };
                    // A tied district elects nobody: both sides waste every vote.
                    let (party_waste, opposition_waste) = if party > opposition {
                        (party - threshold, opposition)
                    } else if opposition > party {
                        (party, opposition - threshold)
                    } else {
                        (party, opposition)
                    };
                    (gap + opposition_waste - party_waste, total_sum + turnout)
                },
            );
            plan_score(tally, divide(gap, total))
        }
        PairedKind::SimplifiedEfficiencyGap => {
            let wins = party
                .iter()
                .zip(opposition)
                .filter(|(party, opposition)| party > opposition)
                .count();
            let seat_share = wins as f64 / party.len() as f64;
            plan_score(
                tally,
                seat_share + 0.5 - 2.0 * overall_vote_share(party, opposition),
            )
        }
        PairedKind::MeanMedian => {
            let mut shares = vote_shares(party, opposition);
            if shares.iter().any(|share| share.is_nan()) {
                return plan_score(tally, f64::NAN);
            }
            let mean = shares.iter().sum::<f64>() / shares.len() as f64;
            shares.sort_by(f64::total_cmp);
            let middle = shares.len() / 2;
            let median = if shares.len().is_multiple_of(2) {
                (shares[middle - 1] + shares[middle]) / 2.0
            } else {
                shares[middle]
            };
            plan_score(tally, median - mean)
        }
        PairedKind::Disproportionality => {
            let seat_share = party
                .iter()
                .zip(opposition)
                .filter(|(party, opposition)| party > opposition)
                .count() as f64
                / party.len() as f64;
            let vote_share = overall_vote_share(party, opposition);
            plan_score(tally, seat_share - vote_share)
        }
        PairedKind::PartisanBias(turnout_model) => {
            let shares = vote_shares(party, opposition);
            let reference = reference_share(&shares, party, opposition, turnout_model);
            let mut seats_at_half = 0.0;
            let mut valid = 0;
            for share in shares {
                if !share.is_finite() {
                    continue;
                }
                valid += 1;
                let difference = share - reference;
                if difference > PARTISAN_BIAS_TIE_TOLERANCE {
                    seats_at_half += 1.0;
                } else if difference.abs() <= PARTISAN_BIAS_TIE_TOLERANCE {
                    seats_at_half += 0.5;
                }
            }
            plan_score(tally, divide(seats_at_half, valid as f64) - 0.5)
        }
        PairedKind::PartisanGini(turnout_model) => {
            let mut shares = vote_shares(party, opposition);
            if shares.iter().any(|share| share.is_nan()) {
                return plan_score(tally, f64::NAN);
            }
            let reference = reference_share(&shares, party, opposition, turnout_model);
            shares.sort_by(|left, right| right.total_cmp(left));
            let curve: Vec<_> = shares
                .iter()
                .map(|share| (reference - share + 0.5).clamp(0.0, 1.0))
                .collect();
            let difference = curve
                .iter()
                .zip(curve.iter().rev())
                .map(|(value, reflected)| (value - (1.0 - reflected)).abs())
                .sum::<f64>()
                / curve.len() as f64;
            plan_score(tally, difference)
        }
    }
}

fn score_population(
    tally: &DistrictTable,
    kind: PopulationKind,
    population: usize,
) -> Result<MetricScore> {
    let populations = tally
        .column(population)
        .ok_or_else(|| Error::InvalidInput("population tally column is out of range".into()))?;
    let ideal = divide(populations.iter().sum(), populations.len() as f64);
    match kind {
        PopulationKind::Deviations => Ok(district_score(
            tally,
            populations
                .iter()
                .map(|population| divide(population - ideal, ideal))
                .collect(),
        )),
        PopulationKind::MaxAbsolute { relative } => {
            let deviation = populations
                .iter()
                .map(|population| (population - ideal).abs())
                .fold(0.0, f64::max);
            plan_score(
                tally,
                if relative {
                    divide(deviation, ideal)
                } else {
                    deviation
                },
            )
        }
        PopulationKind::Max { relative } => {
            let minimum = populations.iter().copied().fold(f64::INFINITY, f64::min);
            let maximum = populations
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let deviation = maximum - minimum;
            plan_score(
                tally,
                if relative {
                    divide(deviation, ideal)
                } else {
                    deviation
                },
            )
        }
    }
}

fn score_demographic(
    tally: &DistrictTable,
    kind: DemographicKind,
    subgroup: usize,
    total: usize,
) -> Result<MetricScore> {
    let subgroup = tally
        .column(subgroup)
        .ok_or_else(|| Error::InvalidInput("subgroup tally column is out of range".into()))?;
    let total = tally
        .column(total)
        .ok_or_else(|| Error::InvalidInput("total tally column is out of range".into()))?;
    let shares: Vec<_> = subgroup
        .iter()
        .zip(total)
        .map(|(&subgroup, &total)| divide(subgroup, total))
        .collect();
    match kind {
        DemographicKind::Shares => Ok(district_score(tally, shares)),
        DemographicKind::DistrictsAboveThreshold { threshold } => plan_score(
            tally,
            shares.iter().filter(|share| **share > threshold).count() as f64,
        ),
    }
}

fn cross_columns<'a>(
    tally: &'a DistrictTable,
    party: &[usize],
    opposition: &[usize],
) -> Result<ElectionColumns<'a>> {
    let party = party
        .iter()
        .map(|&column| {
            tally
                .column(column)
                .ok_or_else(|| Error::InvalidInput("party tally column is out of range".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    let opposition = opposition
        .iter()
        .map(|&column| {
            tally.column(column).ok_or_else(|| {
                Error::InvalidInput("opposition tally column is out of range".into())
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((party, opposition))
}

fn score_cross_election(
    tally: &DistrictTable,
    kind: CrossElectionKind,
    party_columns: &[usize],
    opposition_columns: &[usize],
) -> Result<MetricScore> {
    let (party, opposition) = cross_columns(tally, party_columns, opposition_columns)?;
    let district_count = tally.district_ids().len();
    match kind {
        CrossElectionKind::CompetitiveContests { points_within } => {
            let lower = 0.5 - points_within;
            let upper = 0.5 + points_within;
            let count = party
                .iter()
                .zip(&opposition)
                .flat_map(|(party, opposition)| party.iter().zip(*opposition))
                .filter(|(&party, &opposition)| {
                    let share = divide(party, party + opposition);
                    share > lower && share < upper
                })
                .count();
            plan_score(tally, count as f64)
        }
        CrossElectionKind::PartyWinsByDistrict => {
            let values = (0..district_count)
                .map(|district| {
                    party
                        .iter()
                        .zip(&opposition)
                        .filter(|(party, opposition)| party[district] > opposition[district])
                        .count() as f64
                })
                .collect();
            Ok(district_score(tally, values))
        }
        CrossElectionKind::SwingDistricts
        | CrossElectionKind::PartyDistricts
        | CrossElectionKind::OppositionPartyDistricts => {
            let mut swing = 0;
            let mut stable_party = 0;
            let mut stable_opposition = 0;
            for district in 0..district_count {
                let party_wins = party
                    .iter()
                    .zip(&opposition)
                    .all(|(party, opposition)| party[district] > opposition[district]);
                let opposition_wins = party
                    .iter()
                    .zip(&opposition)
                    .all(|(party, opposition)| opposition[district] > party[district]);
                stable_party += usize::from(party_wins);
                stable_opposition += usize::from(opposition_wins);
                swing += usize::from(!party_wins && !opposition_wins);
            }
            let value = match kind {
                CrossElectionKind::SwingDistricts => swing,
                CrossElectionKind::PartyDistricts => stable_party,
                CrossElectionKind::OppositionPartyDistricts => stable_opposition,
                _ => unreachable!(),
            };
            plan_score(tally, value as f64)
        }
        CrossElectionKind::AggregateSeats => {
            let wins = party
                .iter()
                .zip(&opposition)
                .map(|(party, opposition)| {
                    party
                        .iter()
                        .zip(*opposition)
                        .filter(|(party, opposition)| party > opposition)
                        .count()
                })
                .sum::<usize>();
            plan_score(tally, wins as f64)
        }
        CrossElectionKind::MeanSignedSeatVoteGap | CrossElectionKind::MeanAbsoluteSeatVoteGap => {
            let mut total = 0.0;
            for (party, opposition) in party.iter().zip(&opposition) {
                let seats = party
                    .iter()
                    .zip(*opposition)
                    .filter(|(party, opposition)| party > opposition)
                    .count() as f64
                    / district_count as f64;
                let gap = seats - overall_vote_share(party, opposition);
                total += match kind {
                    CrossElectionKind::MeanSignedSeatVoteGap => gap,
                    CrossElectionKind::MeanAbsoluteSeatVoteGap => gap.abs(),
                    _ => unreachable!(),
                };
            }
            plan_score(tally, total / party.len() as f64)
        }
    }
}

#[cfg(test)]
#[path = "../tests/formulas.rs"]
mod tests;
