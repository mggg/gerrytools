
import us
from evaltools.processing import (
    submissions, tabularized, AssignmentCompressor, remap
)
import jsonlines
from pathlib import Path
import os
import pandas as pd
import cProfile
import gzip
import json

root = Path(os.getcwd()) / Path("tests/test-resources/")

def create_small_sample(size=13):
    with gzip.open(root/"test-assignments/assignments.jsonl.gz") as g:
        with jsonlines.Reader(g) as r:
            counter = 0
            assignments = []

            for assignment in r:
                if counter < size:
                    assignments.append(assignment)
                    counter += 1
                else:
                    break
        with jsonlines.open(root/"test-assignments/test-multiple-assignments.jsonl", "w") as w:
            for assignment in assignments:
                w.write(assignment)

def test_assignmentcompressor_compress():
    # Get the GEOIDs from the blocks.
    blocks = pd.read_csv(root / "test-assignments/test-block-ids.csv")
    geoids = set(blocks["GEOID20"].astype(str))

    # Delete the existing file.
    location = root/"test-assignments/compressed.ac"
    if location.exists(): os.remove(location)

    # Create an AssignmentCompressor.
    ac = AssignmentCompressor(geoids, window=10, location=root/"test-assignments/compressed.ac")

    with ac as compressor:
        with jsonlines.open(root / "test-assignments/test-multiple-assignments.jsonl", mode="r") as reader:
            for submission in reader:
                assignment = { str(k): str(v) for k, v in submission["assignment"].items() }
                compressor.compress(assignment)


def test_assignmentcompressor_decompress():
    # Get the GEOIDs from the blocks.
    blocks = pd.read_csv(root / "test-assignments/test-block-ids.csv")
    geoids = set(blocks["GEOID20"].astype(str))

    # Create an AssignmentCompressor and decompress all the assignments.
    ac = AssignmentCompressor(geoids, location=root / "test-assignments/compressed.ac")
    decompresseds = [assignment for assignment in ac.decompress()]

    # Load all the original assignments into memory.
    originals = []
    with jsonlines.open(root / "test-assignments/test-multiple-assignments.jsonl", mode="r") as reader:
        for submission in reader:
            assignment = { str(k): str(v) for k, v in submission["assignment"].items() }
            originals.append(assignment)

    # First, assert that we have the same number of assignments.
    assert len(originals) == len(decompresseds)

    # Now, we compare the two!
    for decompressed, original in zip(decompresseds, originals):
        assert decompressed == ac.match(original)


def test_match():
    # Create a fake assignment (with improperly ordered keys) and set a desired
    # one.
    fake_identifiers = ["001", "011", "012", "002"]
    fake_assignment = { "002": "12.0", "012": "A", "001": "3" }
    desired_assignment = { "001": "3", "002": "12.0", "011": "-1", "012": "A"}

    # Create an AssignmentCompressor object and test whether our fake and desired
    # assignments match!
    ac = AssignmentCompressor(fake_identifiers)
    indexed = ac.match(fake_assignment)

    # Assert stuff about keys and values!
    for f, d in zip(indexed.items(), desired_assignment.items()):
        fkey, fval = f
        dkey, dval = d
        assert fkey == dkey
        assert type(fkey) == type(dkey)
        assert fval == dval
        assert type(fval) == type(dval)

def test_assignmentcompressor_compress_all():
    # Get the GEOIDs from the blocks.
    blocks = pd.read_csv(root / "test-assignments/test-block-ids.csv")
    geoids = set(blocks["GEOID20"].astype(str))

    # Delete the existing file.
    location = root/"test-assignments/compressed.ac"
    if location.exists(): os.remove(location)

    # Create an AssignmentCompressor.
    ac = AssignmentCompressor(geoids, location=root/"test-assignments/compressed.ac")

    # Get the assignments in one place.
    assignments = []
    with jsonlines.open(root / "test-assignments/test-multiple-assignments.jsonl", mode="r") as reader:
        for submission in reader:
            assignment = { str(k): str(v) for k, v in submission["assignment"].items() }
            assignments.append(assignment)

    # Compress.
    ac.compress_all(assignments)

def profile_assignmentcompressor_compress():
    # Get the GEOIDs from the blocks.
    blocks = pd.read_csv(root / "test-assignments/test-block-ids.csv")
    geoids = set(blocks["GEOID20"].astype(str))

    # Delete the existing file.
    location = root/"test-assignments/compressed.ac"
    if location.exists(): os.remove(location)

    # Create an AssignmentCompressor.
    ac = AssignmentCompressor(geoids, location=root/"test-assignments/compressed.ac")

    # Get the assignments in one place.
    assignments = []
    with jsonlines.open(root / "test-assignments/test-multiple-assignments.jsonl", mode="r") as reader:
        for submission in reader:
            assignment = { str(k): str(v) for k, v in submission["assignment"].items() }
            assignments.append(assignment)

    # Profile compression.
    with cProfile.Profile() as profiler:
        with ac as compressor:
            for assignment in assignments: compressor.compress(assignment)

    profiler.dump_stats(root/"test-assignments/compress.pstats")


def profile_assignmentcompressor_decompress():
    # Get the GEOIDs from the blocks.
    blocks = pd.read_csv(root / "test-assignments/test-block-ids.csv")
    geoids = set(blocks["GEOID20"].astype(str))
    ac = AssignmentCompressor(geoids, location=root/"test-assignments/compressed.ac")

    # Profile decompression.
    with cProfile.Profile() as profiler:
        for _ in ac.decompress(): pass

    profiler.dump_stats(root/"test-assignments/decompress.pstats")


def test_submissions():
    # Select a state; we'll use Wisconsin. Set a sample size.
    state = us.states.IN
    sample = 20

    # Try to get stuff.
    subs = submissions(state, sample=sample)
    plans, cois, written = tabularized(state, subs)

    # Write these to file.
    plans.to_csv(root / "test-districtr.csv", index=False)

    # Assert that we have the right number of plans; this should match the sample
    # size we set earlier.
    assert len(subs) == sample

    # Assert that we have the right columns in the dataframes. With a sample size
    # of 20, neither of these should fail.
    assert not plans["plan"].isna().all()
    assert not cois["plan"].isna().all()


def test_remap():
    # Read from file.
    plans = pd.read_csv(root / "test-districtr.csv")

    # Get the population mapping.

    # Read the unit mappings in.
    with open(root / "test-vtds-to-blocks.json") as f: vtds_to_blocks = json.load(f)
    with open(root / "test-blocks-to-vtds.json") as f: blocks_to_vtds = json.load(f)
    with open(root / "test-precincts-to-blocks.json") as f: precincts_to_blocks = json.load(f)
    with open(root / "test-blocks-to-vtds.json") as f: blocks_to_vtds = json.load(f)

    # Create the mapping of mappings!
    unitmaps = {
        "2020 VTDs": vtds_to_blocks,
        # "Precincts": precincts_to_blocks
    }

    # Remap things. Appears to work fine for now!
    plans = remap(plans, unitmaps)

if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
    # create_small_sample()
    # profile_assignmentcompressor_compress()
    # profile_assignmentcompressor_decompress()

    # Create images.
    # os.system(f"sh {root/'test-assignments/profiling.sh'}")

    # Test that we're getting the right outputs.
    # test_submissions()
    test_remap()
