use crate::{Error, Result};

/// One metric's district-valued columns for a single plan.
///
/// Values are column-major: each contiguous `district_ids.len()` values belongs to one column.
#[derive(Clone, Debug, PartialEq)]
pub struct DistrictTable {
    district_ids: Vec<u16>,
    values: Vec<f64>,
    column_count: usize,
}

impl DistrictTable {
    pub(crate) fn new(district_ids: Vec<u16>, values: Vec<f64>, column_count: usize) -> Self {
        debug_assert_eq!(values.len(), district_ids.len() * column_count);
        Self {
            district_ids,
            values,
            column_count,
        }
    }

    /// Return the strictly increasing district IDs shared by every column.
    pub fn district_ids(&self) -> &[u16] {
        &self.district_ids
    }

    /// Return the number of district-valued columns.
    pub fn column_count(&self) -> usize {
        self.column_count
    }

    /// Return one column in district-ID order, or `None` when `index` is out of range.
    pub fn column(&self, index: usize) -> Option<&[f64]> {
        if index >= self.column_count {
            return None;
        }
        let district_count = self.district_ids.len();
        let start = index.checked_mul(district_count)?;
        self.values.get(start..start + district_count)
    }
}

/// One metric's plan-valued columns for a single plan.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanTable {
    district_ids: Vec<u16>,
    values: Vec<f64>,
}

impl PlanTable {
    /// Create a plan-valued table with strictly increasing district IDs.
    pub fn new(district_ids: Vec<u16>, values: Vec<f64>) -> Result<Self> {
        if district_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::InvalidInput(
                "district ids must be unique and strictly increasing".into(),
            ));
        }
        if values.is_empty() {
            return Err(Error::InvalidInput(
                "a plan table requires at least one value".into(),
            ));
        }
        Ok(Self {
            district_ids,
            values,
        })
    }

    /// Return the strictly increasing district IDs present in the plan.
    pub fn district_ids(&self) -> &[u16] {
        &self.district_ids
    }

    /// Return the plan-valued metric columns.
    pub fn values(&self) -> &[f64] {
        &self.values
    }
}

/// One registered metric's values for a single plan.
#[derive(Clone, Debug, PartialEq)]
pub enum MetricScore {
    /// Values with one row per district.
    District(DistrictTable),
    /// Values with one row per plan.
    Plan(PlanTable),
}

impl MetricScore {
    /// Return the district IDs represented by this score.
    pub fn district_ids(&self) -> &[u16] {
        match self {
            Self::District(table) => table.district_ids(),
            Self::Plan(table) => table.district_ids(),
        }
    }
}

/// True when every score's district set equals `expected`.
pub(crate) fn all_district_ids_match<'a>(
    expected: &[u16],
    scores: impl IntoIterator<Item = &'a MetricScore>,
) -> bool {
    scores
        .into_iter()
        .all(|score| score.district_ids() == expected)
}
