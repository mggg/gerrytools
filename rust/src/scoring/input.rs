use crate::{Error, Result};
use ben::format::banners::has_known_banner_prefix;
use ben::io::bundle::format::{ASSET_TYPE_GRAPH, BENDL_MAGIC};
use ben::io::bundle::{BendlReader, ExactLen};
use ben::io::reader::{BenStreamFrameReader, BenStreamReader, BenWireFormat};
use ben::BenVariant;
use serde::Deserialize;
use serde_json::Value;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

const XZ_MAGIC: [u8; 6] = [0xFD, 0x37, 0x7A, 0x58, 0x5A, 0x00];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputKind {
    Bundle,
    Ben,
    Xben,
}

#[derive(Debug)]
/// A validated, repeatably readable BEN, XBEN, or finalized BENDL assignment source.
///
/// The source retains the validated file descriptor and permits only one active read pass.
pub struct AssignmentSource {
    /// The file handle validation ran against. Every later pass clones this descriptor instead of
    /// reopening the path, so a swapped-out file cannot bypass the checks performed at open time.
    file: File,
    kind: SourceKind,
    active_pass: Arc<AtomicBool>,
}

#[derive(Debug)]
enum SourceKind {
    File {
        wire: BenWireFormat,
    },
    Bundle {
        offset: u64,
        len: u64,
        wire: BenWireFormat,
        sample_count: usize,
    },
}

#[derive(Deserialize)]
struct EmbeddedGraph {
    #[serde(default)]
    nodes: Vec<EmbeddedNode>,
}

#[derive(Deserialize)]
struct EmbeddedNode {
    id: Value,
}

struct PassGuard {
    active: Arc<AtomicBool>,
}

impl PassGuard {
    fn acquire(active: Arc<AtomicBool>) -> Result<Self> {
        active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| {
                Error::InvalidInput("assignment source already has an active pass".into())
            })?;
        Ok(Self { active })
    }
}

impl Drop for PassGuard {
    fn drop(&mut self) {
        self.active.store(false, Ordering::Release);
    }
}

struct PassReader {
    inner: Box<dyn Read + Send>,
    _guard: PassGuard,
}

impl Read for PassReader {
    fn read(&mut self, buffer: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buffer)
    }
}

impl AssignmentSource {
    /// Open a BEN, XBEN, or finalized BENDL assignment source based on its leading bytes.
    ///
    /// BENDL streams are checksum-verified and checked against their declared sample count before
    /// this returns.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let mut file = File::open(path)?;
        let kind = match sniff(path, &mut file)? {
            InputKind::Ben => SourceKind::File {
                wire: BenWireFormat::Ben,
            },
            InputKind::Xben => SourceKind::File {
                wire: BenWireFormat::XBen,
            },
            InputKind::Bundle => Self::open_bundle(&file)?,
        };
        Ok(Self {
            file,
            kind,
            active_pass: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Return the assignment encoding variant declared by the source.
    pub fn variant(&self) -> Result<BenVariant> {
        Ok(self.open_reader()?.variant())
    }

    /// Verify every BENDL asset and compare its embedded graph against evaluator node order.
    ///
    /// This reads through clones of the descriptor retained by `AssignmentSource`, so validation
    /// and scoring cannot be separated by a path replacement.
    pub fn validate_bundle_graph_node_order(&self, expected_json: Option<&str>) -> Result<()> {
        let SourceKind::Bundle { .. } = &self.kind else {
            return Ok(());
        };
        let mut reader = BendlReader::open(clone_at(&self.file, 0)?)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        reader
            .verify_all_asset_checksums()
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        let Some(entry) = reader.find_asset_by_type(ASSET_TYPE_GRAPH).cloned() else {
            return Ok(());
        };
        let expected_json = expected_json.ok_or_else(|| {
            Error::InvalidInput(
                "BENDL graph node labels must be JSON-compatible for evaluator alignment".into(),
            )
        })?;
        let expected: Vec<Value> = serde_json::from_str(expected_json).map_err(|error| {
            Error::InvalidInput(format!("invalid evaluator node order: {error}"))
        })?;
        let graph: EmbeddedGraph = serde_json::from_slice(
            &reader
                .asset_bytes(&entry)
                .map_err(|error| Error::InvalidInput(error.to_string()))?,
        )
        .map_err(|error| Error::InvalidInput(format!("invalid embedded BENDL graph: {error}")))?;
        if graph.nodes.into_iter().map(|node| node.id).ne(expected) {
            return Err(Error::InvalidInput(
                "BENDL graph node order must exactly match evaluator node order".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn declared_sample_count(&self) -> Option<usize> {
        match &self.kind {
            SourceKind::File { .. } => None,
            SourceKind::Bundle { sample_count, .. } => Some(*sample_count),
        }
    }

    /// Start a fresh pass over the validated file, rejecting overlapping passes.
    pub(crate) fn open_reader(&self) -> Result<BenStreamReader<Box<dyn Read + Send + 'static>>> {
        let guard = PassGuard::acquire(Arc::clone(&self.active_pass))?;
        let (reader, wire): (Box<dyn Read + Send>, BenWireFormat) = match &self.kind {
            SourceKind::File { wire } => (Box::new(clone_at(&self.file, 0)?), *wire),
            SourceKind::Bundle {
                offset, len, wire, ..
            } => (
                Box::new(ExactLen::bounded(clone_at(&self.file, *offset)?, *len)),
                *wire,
            ),
        };
        let reader: Box<dyn Read + Send> = Box::new(PassReader {
            inner: reader,
            _guard: guard,
        });
        let reader = match wire {
            BenWireFormat::Ben => BenStreamReader::from_ben(reader),
            BenWireFormat::XBen => BenStreamReader::from_xben(reader),
        }
        .map_err(|error| Error::InvalidInput(error.to_string()))?;
        Ok(reader.silent(true))
    }

    pub(crate) fn open_frames(
        &self,
    ) -> Result<BenStreamFrameReader<Box<dyn Read + Send + 'static>>> {
        Ok(self.open_reader()?.into_frames())
    }

    fn open_bundle(file: &File) -> Result<SourceKind> {
        let file_len = file.metadata()?.len();
        let mut reader = BendlReader::open(clone_at(file, 0)?)
            .map_err(|error| Error::InvalidInput(error.to_string()))?;
        let (offset, len) = reader.assignment_stream_range()?;
        let end = offset.checked_add(len).ok_or_else(|| {
            Error::InvalidInput(format!("bundle stream range [{offset}, +{len}) overflows"))
        })?;
        if end > file_len {
            return Err(Error::InvalidInput(format!(
                "bundle assignment stream range [{offset}, {end}) exceeds file length {file_len}"
            )));
        }
        let wire: BenWireFormat = reader
            .assignment_format()
            .ok_or_else(|| Error::InvalidInput("bundle has no assignment format".into()))?
            .into();
        let sample_count: usize = reader
            .sample_count()
            .ok_or_else(|| Error::InvalidInput("bundle is not finalized".into()))?
            .try_into()
            .map_err(|_| {
                Error::InvalidInput("bundle sample count is negative or too large".into())
            })?;

        let actual_samples = reader
            .open_assignment_reader()
            .map_err(|error| Error::InvalidInput(error.to_string()))?
            .silent(true);
        // This verified decode checks CRC32C at EOF and is intentionally the only checksum pass.
        // Will move scoring onto verified frame/twodelta readers if binary-ensemble exposes them.
        let actual_samples = actual_samples.count_samples()?;
        if actual_samples != sample_count {
            return Err(Error::InvalidInput(format!(
                "bundle declares {sample_count} samples but its stream contains {actual_samples}"
            )));
        }

        Ok(SourceKind::Bundle {
            offset,
            len,
            wire,
            sample_count,
        })
    }
}

/// Duplicate the held descriptor and position it for a fresh pass.
fn clone_at(file: &File, offset: u64) -> Result<File> {
    let mut clone = file.try_clone()?;
    clone.seek(SeekFrom::Start(offset))?;
    Ok(clone)
}

fn sniff(path: &Path, file: &mut File) -> Result<InputKind> {
    let mut bytes = [0_u8; 17];
    let mut filled = 0;
    while filled < bytes.len() {
        match file.read(&mut bytes[filled..]) {
            Ok(0) => break,
            Ok(count) => filled += count,
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    let bytes = &bytes[..filled];

    if bytes.starts_with(&BENDL_MAGIC) {
        Ok(InputKind::Bundle)
    } else if bytes.starts_with(b"BENDL") {
        Err(Error::InvalidInput(format!(
            "unrecognized BENDL version magic {:?}",
            &bytes[..bytes.len().min(BENDL_MAGIC.len())]
        )))
    } else if has_known_banner_prefix(bytes) {
        Ok(InputKind::Ben)
    } else if bytes.starts_with(&XZ_MAGIC) {
        Ok(InputKind::Xben)
    } else {
        Err(Error::InvalidInput(format!(
            "{} is not a BEN, XBEN, or BENDL input",
            path.display()
        )))
    }
}

#[cfg(test)]
#[path = "../tests/input.rs"]
mod tests;
