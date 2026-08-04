use super::*;
use crate::test_support::TempPath;
use ben::io::bundle::format::AssignmentFormat;
use ben::io::bundle::{BendlReader, BendlWriter};
use ben::io::writer::{BenStreamWriter, XzEncodeOptions};
use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};

fn write_ben(path: &Path, xben: bool) {
    let file = File::create(path).unwrap();
    if xben {
        let mut writer =
            BenStreamWriter::for_xben(file, BenVariant::Standard, XzEncodeOptions::new()).unwrap();
        writer.write_assignment(vec![1, 1, 2, 2]).unwrap();
        writer.finish().unwrap();
    } else {
        let mut writer = BenStreamWriter::for_ben(file, BenVariant::Standard).unwrap();
        writer.write_assignment(vec![1, 1, 2, 2]).unwrap();
        writer.finish().unwrap();
    }
}

fn write_bundle(path: &Path, sample_count: i64) {
    let writer = BendlWriter::new(File::create(path).unwrap(), AssignmentFormat::Ben).unwrap();
    let mut session = writer.into_stream_session().unwrap();
    {
        let mut stream = BenStreamWriter::for_ben(&mut session, BenVariant::Standard).unwrap();
        stream.write_assignment(vec![1, 1, 2, 2]).unwrap();
        stream.finish().unwrap();
    }
    session.finish_into_writer(sample_count).finish().unwrap();
}

#[test]
fn detects_ben_xben_and_verified_bendl_sources() {
    let ben = TempPath::new("ben");
    write_ben(ben.path(), false);
    let source = AssignmentSource::open(ben.path()).unwrap();
    assert_eq!(source.variant().unwrap(), BenVariant::Standard);
    assert_eq!(source.declared_sample_count(), None);

    let xben = TempPath::new("xben");
    write_ben(xben.path(), true);
    assert_eq!(
        AssignmentSource::open(xben.path())
            .unwrap()
            .variant()
            .unwrap(),
        BenVariant::Standard
    );

    let bendl = TempPath::new("bendl");
    write_bundle(bendl.path(), 1);
    let source = AssignmentSource::open(bendl.path()).unwrap();
    assert_eq!(source.variant().unwrap(), BenVariant::Standard);
    assert_eq!(source.declared_sample_count(), Some(1));
    source.validate_bundle_graph_node_order(None).unwrap();
}

#[test]
fn later_passes_reread_the_file_validated_at_open_time() {
    let ben = TempPath::new("ben");
    write_ben(ben.path(), false);
    let source = AssignmentSource::open(ben.path()).unwrap();

    // Passes clone the held descriptor, so repeated reads see the same frames.
    for _ in 0..2 {
        assert_eq!(source.variant().unwrap(), BenVariant::Standard);
        let mut frames = source.open_frames().unwrap();
        let (frame, repetitions) = frames.next().unwrap().unwrap();
        assert_eq!(frame.expand_self_contained().unwrap(), vec![1, 1, 2, 2]);
        assert_eq!(repetitions, 1);
        assert!(frames.next().is_none());
    }

    // Replacing the path no longer swaps the scored bytes out from under the validated source.
    #[cfg(unix)]
    {
        std::fs::remove_file(ben.path()).unwrap();
        write_bundle(ben.path(), 1);
        assert_eq!(source.variant().unwrap(), BenVariant::Standard);
        let (frame, _) = source.open_frames().unwrap().next().unwrap().unwrap();
        assert_eq!(frame.expand_self_contained().unwrap(), vec![1, 1, 2, 2]);
    }
}

#[test]
fn rejects_unknown_input_and_corrupt_bundle_stream() {
    let unknown = TempPath::new("data");
    File::create(unknown.path())
        .unwrap()
        .write_all(b"not an ensemble")
        .unwrap();
    assert!(matches!(
        AssignmentSource::open(unknown.path()),
        Err(Error::InvalidInput(_))
    ));

    let bendl = TempPath::new("bendl");
    write_bundle(bendl.path(), 1);
    let mut reader = BendlReader::open(File::open(bendl.path()).unwrap()).unwrap();
    let (offset, _) = reader.assignment_stream_range().unwrap();
    let mut file = OpenOptions::new().write(true).open(bendl.path()).unwrap();
    file.seek(SeekFrom::Start(offset + 17)).unwrap();
    file.write_all(&[0xFF]).unwrap();

    let error = AssignmentSource::open(bendl.path()).unwrap_err();
    assert!(error.to_string().to_lowercase().contains("checksum"));

    let wrong_count = TempPath::new("bendl");
    write_bundle(wrong_count.path(), 2);
    let error = AssignmentSource::open(wrong_count.path()).unwrap_err();
    assert!(error.to_string().contains("declares 2 samples"));
}

#[test]
fn rejects_overlapping_passes_that_would_share_a_file_cursor() {
    let ben = TempPath::new("ben");
    write_ben(ben.path(), false);
    let source = AssignmentSource::open(ben.path()).unwrap();
    let first = source.open_frames().unwrap();

    let error = match source.open_frames() {
        Ok(_) => panic!("overlapping pass unexpectedly opened"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("already has an active pass"));

    drop(first);
    assert!(source.open_frames().is_ok());
}
