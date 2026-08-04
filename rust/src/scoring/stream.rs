//! Assignment-stream traversal, batching, and emission.

use super::uniqueness::{assignment_hashes, UniquenessCounts};
use super::*;

/// Reject a frame that would expand to zero samples, before it is scored or emitted.
///
/// The ben readers already reject zero-count frames at decode time; this re-check keeps the
/// failure at the offending frame instead of surfacing as a poisoned [`crate::RunWriter`] push.
pub(super) fn ensure_positive_repetitions(repetitions: u16, frame: u64) -> Result<()> {
    if repetitions == 0 {
        return Err(Error::ZeroRepetitionFrame { frame });
    }
    Ok(())
}

impl<'a> Scorer<'a> {
    /// Score a source directly into one atomically published version-1 run directory.
    pub fn score_run(
        &self,
        source: &AssignmentSource,
        options: StreamOptions,
        output_path: impl AsRef<Path>,
        metadata: crate::RunMetadata,
    ) -> Result<StreamSummary> {
        self.validate_run_metadata(&metadata)?;
        let mut writer = crate::RunWriter::new(output_path, metadata)?;
        let summary = self.score_stream(source, options, |score| writer.push(&score))?;
        writer.finish(summary)?;
        Ok(summary)
    }

    /// Score a source in file order and emit each accepted frame without retaining prior rows.
    pub fn score_stream<F>(
        &self,
        source: &AssignmentSource,
        options: StreamOptions,
        emit: F,
    ) -> Result<StreamSummary>
    where
        F: FnMut(PlanScore) -> Result<()>,
    {
        if self.metrics.is_empty() {
            return Err(Error::EmptyScorer);
        }
        let node_count = self.node_count.ok_or(Error::EmptyScorer)?;
        if options.max_samples == Some(0) {
            let mut summary = StreamSummary::default();
            if options.track_uniqueness {
                UniquenessCounts::default().apply_to(&mut summary)?;
            }
            return Ok(summary);
        }

        match source.variant()? {
            BenVariant::TwoDelta => self.score_twodelta(
                source,
                node_count,
                options.max_samples,
                options.track_uniqueness,
                emit,
            ),
            // MkvChain frames are full assignments, but adjacent frames are updated incrementally.
            BenVariant::MkvChain => self.score_mkvchain(source, options, emit),
            // NOTE: Standard plans may be unrelated. I tried delta derivation with a retained
            // 10k-plan, but that made the cut-only scorer 2.37x slower, so every assignment is
            // still scored independently.
            BenVariant::Standard => self.score_standard(source, options, emit),
        }
    }

    /// Score many independent assignments in parallel, requiring one shared district set.
    ///
    /// Returns the shared district ids and, per assignment, one [`MetricScore`] per registered
    /// metric in registration order.
    pub fn score_batch(
        &self,
        assignments: &[Vec<u16>],
    ) -> Result<(Vec<u16>, Vec<Vec<MetricScore>>)> {
        if self.metrics.is_empty() {
            return Err(Error::EmptyScorer);
        }
        let rows = assignments
            .par_iter()
            .map(|assignment| self.score_assignment(assignment))
            .collect::<Result<Vec<_>>>()?;

        let district_ids = rows
            .first()
            .and_then(|row| row.first())
            .map_or_else(Vec::new, |score| score.district_ids().to_vec());
        if !all_district_ids_match(&district_ids, rows.iter().flatten()) {
            return Err(Error::InvalidInput(
                "district labels must be the same for every metric and assignment".into(),
            ));
        }
        Ok((district_ids, rows))
    }

    /// Validate run metadata against the registered metrics.
    ///
    /// Registration-time counterpart of the Python projection validator (`validate_projections`
    /// in python.rs); only the shape check is shared, via [`super::output::check_declared_shape`].
    fn validate_run_metadata(&self, metadata: &crate::RunMetadata) -> Result<()> {
        if self.metrics.is_empty() {
            return Err(Error::EmptyScorer);
        }
        if metadata.metrics.len() != self.metrics.len() {
            return Err(Error::InvalidInput(format!(
                "run metadata has {} metrics; scorer has {}",
                metadata.metrics.len(),
                self.metrics.len()
            )));
        }
        for (expected, actual) in self.metrics.iter().zip(&metadata.metrics) {
            if actual.instance != expected.name {
                return Err(Error::InvalidInput(format!(
                    "run metric instance is {:?}; expected {:?}",
                    actual.instance, expected.name
                )));
            }
            if actual.kind != expected.source.kind() {
                return Err(Error::InvalidInput(format!(
                    "run metric {:?} has kind {:?}; expected {:?}",
                    actual.instance,
                    actual.kind,
                    expected.source.kind()
                )));
            }
            super::output::check_declared_shape(actual, expected.source.shape())?;
            if let Some(subkeys) = &expected.subkeys {
                if &actual.subkeys != subkeys {
                    return Err(Error::InvalidInput(format!(
                        "run metric {:?} declares subkeys {:?}; the scorer registered {:?}",
                        actual.instance, actual.subkeys, subkeys
                    )));
                }
            } else if actual.subkeys.len() != expected.source.column_count() {
                return Err(Error::InvalidInput(format!(
                    "run metric {:?} declares {} subkeys; expected {}",
                    actual.instance,
                    actual.subkeys.len(),
                    expected.source.column_count()
                )));
            }
        }
        Ok(())
    }

    pub(super) fn score_assignment(&self, assignment: &[u16]) -> Result<Vec<MetricScore>> {
        let tally = self
            .tally_bank
            .map(|metric| metric.score(assignment))
            .transpose()?;
        self.metrics
            .iter()
            .map(|entry| match &entry.source {
                MetricSource::Independent(metric) => metric.score(assignment),
                MetricSource::SharedTally(metric) => metric.score(
                    tally
                        .as_ref()
                        .expect("shared tally metric requires shared tally bank"),
                ),
            })
            .collect()
    }

    pub(super) fn incremental_state(&self, assignment: &[u16]) -> Result<ScorerState<'a>> {
        let tally = self
            .tally_bank
            .map(|metric| metric.incremental(assignment))
            .transpose()?;
        let metrics = self
            .metrics
            .iter()
            .map(|entry| match &entry.source {
                MetricSource::Independent(metric) => metric
                    .incremental(assignment)
                    .map(Box::new)
                    .map(LogicalState::Independent),
                MetricSource::SharedTally(metric) => Ok(LogicalState::SharedTally(metric.clone())),
            })
            .collect::<Result<_>>()?;
        Ok(ScorerState {
            assignment: assignment.to_vec(),
            tally,
            metrics,
        })
    }

    fn score_standard<F>(
        &self,
        source: &AssignmentSource,
        options: StreamOptions,
        mut emit: F,
    ) -> Result<StreamSummary>
    where
        F: FnMut(PlanScore) -> Result<()>,
    {
        let mut frames = source.open_frames()?;
        let batch_size = options.batch_size.max(1);
        let mut batch = Vec::with_capacity(batch_size.min(DEFAULT_BATCH_SIZE));
        let mut remaining = options.max_samples;
        let mut frame_index = 0_u64;
        let mut summary = StreamSummary::default();
        let mut uniqueness = options.track_uniqueness.then(UniquenessCounts::default);

        while remaining != Some(0) {
            let Some(frame) = frames.next() else {
                break;
            };
            let (frame, repetitions) = frame?;
            ensure_positive_repetitions(repetitions, frame_index)?;
            frame_index += 1;
            batch.push((frame, cap_repetitions(&mut remaining, repetitions)));
            if batch.len() == batch_size {
                self.emit_full_batch(&batch, &mut summary, &mut uniqueness, &mut emit)?;
                batch.clear();
            }
        }
        self.emit_full_batch(&batch, &mut summary, &mut uniqueness, &mut emit)?;
        if let Some(uniqueness) = uniqueness {
            uniqueness.apply_to(&mut summary)?;
        }
        Ok(summary)
    }

    fn emit_full_batch<F>(
        &self,
        batch: &[(DecodeFrame, u16)],
        summary: &mut StreamSummary,
        uniqueness: &mut Option<UniquenessCounts>,
        emit: &mut F,
    ) -> Result<()>
    where
        F: FnMut(PlanScore) -> Result<()>,
    {
        let results = batch
            .par_iter()
            .map(|(frame, repetitions)| {
                let assignment = frame
                    .expand_self_contained()
                    .map_err(|error| Error::InvalidInput(error.to_string()))?;
                let metrics = self.score_assignment(&assignment)?;
                let hashes = uniqueness
                    .is_some()
                    .then(|| assignment_hashes(&assignment))
                    .transpose()?;
                Ok((*repetitions, metrics, hashes))
            })
            .collect::<Result<Vec<_>>>()?;

        for (repetitions, metrics, hashes) in results {
            emit(PlanScore {
                sample_offset: summary.samples,
                repetitions,
                accepted_index: summary.accepted,
                metrics,
            })?;
            if let (Some(uniqueness), Some(hashes)) = (uniqueness.as_mut(), hashes) {
                uniqueness.observe_hashes(hashes);
            }
            summary.samples = summary
                .samples
                .checked_add(repetitions as u64)
                .ok_or_else(|| Error::InvalidInput("sample count overflowed u64".into()))?;
            summary.accepted += 1;
        }
        Ok(())
    }

    fn score_mkvchain<F>(
        &self,
        source: &AssignmentSource,
        options: StreamOptions,
        mut emit: F,
    ) -> Result<StreamSummary>
    where
        F: FnMut(PlanScore) -> Result<()>,
    {
        let expected = self.node_count.expect("stream scoring requires metrics");
        let mut frames = source.open_frames()?;
        let mut remaining = options.max_samples;
        let mut frame_index = 0_u64;
        let mut summary = StreamSummary::default();
        let mut uniqueness = options.track_uniqueness.then(UniquenessCounts::default);
        let mut current: Option<ScorerState<'a>> = None;

        while remaining != Some(0) {
            let Some(frame) = frames.next() else {
                break;
            };
            let (frame, repetitions) = frame?;
            ensure_positive_repetitions(repetitions, frame_index)?;
            frame_index += 1;
            let repetitions = cap_repetitions(&mut remaining, repetitions);
            let assignment = frame
                .expand_self_contained()
                .map_err(|error| Error::InvalidInput(error.to_string()))?;
            if assignment.len() != expected {
                return Err(Error::AssignmentLength {
                    actual: assignment.len(),
                    expected,
                });
            }
            let changes = current
                .as_ref()
                .map(|state| {
                    state
                        .assignment
                        .iter()
                        .zip(&assignment)
                        .enumerate()
                        .filter_map(|(node, (&old, &new))| {
                            (old != new).then_some(DeltaChange { node, old, new })
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default();
            if summary.accepted.is_multiple_of(MKVCHAIN_RESYNC_INTERVAL) {
                current = Some(self.incremental_state(&assignment)?);
            } else {
                let state = current
                    .as_mut()
                    .expect("a non-resync frame follows an initialized frame");
                state.update(&changes)?;
            }
            let metrics = current
                .as_ref()
                .expect("every frame initializes or updates incremental state")
                .result()?;
            emit(PlanScore {
                sample_offset: summary.samples,
                repetitions,
                accepted_index: summary.accepted,
                metrics,
            })?;
            if let Some(uniqueness) = &mut uniqueness {
                if uniqueness.has_current() {
                    uniqueness.observe_changes(&changes)?;
                } else {
                    uniqueness.reset(&assignment);
                }
            }
            summary.samples = summary
                .samples
                .checked_add(repetitions as u64)
                .ok_or_else(|| Error::InvalidInput("sample count overflowed u64".into()))?;
            summary.accepted += 1;
        }
        if let Some(uniqueness) = uniqueness {
            uniqueness.apply_to(&mut summary)?;
        }
        Ok(summary)
    }

    fn score_twodelta<F>(
        &self,
        source: &AssignmentSource,
        node_count: usize,
        max_samples: Option<u64>,
        track_uniqueness: bool,
        mut emit: F,
    ) -> Result<StreamSummary>
    where
        F: FnMut(PlanScore) -> Result<()>,
    {
        // Incremental states, absent until the first snapshot arrives. Each state retains its own
        // current assignment, so deltas are validated without threading a canonical plan here.
        let mut current: Option<ScorerState<'_>> = None;
        let mut remaining = max_samples;
        let mut summary = StreamSummary::default();
        let mut uniqueness = track_uniqueness.then(UniquenessCounts::default);

        for event in source.open_reader()?.into_twodelta_events() {
            if remaining == Some(0) {
                break;
            }
            let repetitions = match event? {
                TwoDeltaFrameEvent::Snapshot {
                    assignment: snapshot,
                    changes,
                    count,
                } => {
                    ensure_positive_repetitions(count, summary.accepted)?;
                    if snapshot.len() != node_count {
                        return Err(Error::AssignmentLength {
                            actual: snapshot.len(),
                            expected: node_count,
                        });
                    }
                    let changes = changes.map(delta_changes);
                    match (&mut current, changes.as_deref()) {
                        (Some(states), Some(changes)) => {
                            states.apply_snapshot_delta(&snapshot, changes)?;
                        }
                        (Some(states), None) => states.reset(&snapshot)?,
                        (current @ None, _) => {
                            *current = Some(self.incremental_state(&snapshot)?);
                        }
                    }
                    let repetitions = cap_repetitions(&mut remaining, count);
                    if let Some(uniqueness) = &mut uniqueness {
                        if uniqueness.has_current() && changes.is_some() {
                            uniqueness
                                .observe_changes(changes.as_deref().expect("checked as present"))?;
                        } else {
                            uniqueness.reset(&snapshot);
                        }
                    }
                    repetitions
                }
                TwoDeltaFrameEvent::Delta { changes, count } => {
                    ensure_positive_repetitions(count, summary.accepted)?;
                    let states = current.as_mut().ok_or(Error::DeltaBeforeSnapshot)?;
                    let changes = delta_changes(changes);
                    states.update(&changes)?;
                    let repetitions = cap_repetitions(&mut remaining, count);
                    if let Some(uniqueness) = &mut uniqueness {
                        uniqueness.observe_changes(&changes)?;
                    }
                    repetitions
                }
            };

            let states = current
                .as_ref()
                .expect("both event arms leave a current snapshot in place");
            let metrics = states.result()?;
            emit(PlanScore {
                sample_offset: summary.samples,
                repetitions,
                accepted_index: summary.accepted,
                metrics,
            })?;
            summary.samples = summary
                .samples
                .checked_add(repetitions as u64)
                .ok_or_else(|| Error::InvalidInput("sample count overflowed u64".into()))?;
            summary.accepted += 1;
        }
        if let Some(uniqueness) = uniqueness {
            uniqueness.apply_to(&mut summary)?;
        }
        Ok(summary)
    }
}
