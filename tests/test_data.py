
import us
import jsonlines
from pathlib import Path
import os
import pandas as pd
import cProfile
import gzip
import json
from evaltools.data import (
    cvap, acs5, census, variables, submissions, tabularized, AssignmentCompressor,
    remap
)
import us

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_cvap_tracts():
    al = us.states.AL
    data = cvap(al, geometry="tract")

    # Set some testing variables.
    columns = {
        "TRACT10", "CVAP19", "NHCVAP19", "NHAICVAP19", "NHACVAP19", "NHBCVAP19",
        "NHNHPICVAP19", "NHWCVAP19", "NHAIWCVAP19", "NHAWCVAP19", "NHBWCVAP19",
        "NHAIBCVAP19", "NHOTHCVAP19", "HCVAP19", "POCCVAP19"
    }
    tracts = 1181

    # Do some assert-ing.
    assert set(list(data)) == columns
    assert len(data) == tracts

def test_cvap_bgs():
    al = us.states.AL
    data = cvap(al, geometry="block group")
    
    # Set some testing variables.
    columns = {
        "BLOCKGROUP10", "CVAP19", "NHCVAP19", "NHAICVAP19", "NHACVAP19", "NHBCVAP19",
        "NHNHPICVAP19", "NHWCVAP19", "NHAIWCVAP19", "NHAWCVAP19", "NHBWCVAP19",
        "NHAIBCVAP19", "NHOTHCVAP19", "HCVAP19", "POCCVAP19"
    }
    bgs = 3438

    # Do some assert-ing.
    assert set(list(data)) == columns
    assert len(data) == bgs

def test_acs5_tracts():
    state = us.states.AL
    data = acs5(state, geometry="tract")

    tracts = 1181
    columns = {
        "TOTPOP19", "WHITE19", "BLACK19", "AMIN19", "ASIAN19", "NHPI19", "OTH19",
        "2MORE19", "NHISP19", "WVAP19", "BVAP19", "AMINVAP19", "ASIANVAP19",
        "NHPIVAP19", "OTHVAP19", "2MOREVAP19", "HVAP19", "TRACT10", "VAP19",
        "WCVAP19", "BCVAP19", "AMINCVAP19", "ASIANCVAP19", "NHPICVAP19", "OTHCVAP19",
        "2MORECVAP19", "NHWCVAP19", "HCVAP19", "CVAP19", "POCVAP19", "NHWVAP19"
    }

    # Assert some stuff.
    assert len(data) == tracts
    assert set(list(data)) == columns

def test_acs5_bgs():
    AL = us.states.AL
    data = acs5(AL, geometry="block group")
    bgs = 3438
    columns = {
        "TOTPOP19", "WHITE19", "BLACK19", "AMIN19", "ASIAN19", "NHPI19", "OTH19",
        "2MORE19", "NHISP19", "WVAP19", "BVAP19", "AMINVAP19", "ASIANVAP19",
        "NHPIVAP19", "OTHVAP19", "2MOREVAP19", "HVAP19", "BLOCKGROUP10", "VAP19",
        "WCVAP19", "BCVAP19", "AMINCVAP19", "ASIANCVAP19", "NHPICVAP19", "OTHCVAP19",
        "2MORECVAP19", "NHWCVAP19", "HCVAP19", "CVAP19", "POCVAP19", "NHWVAP19"
    }

    # Assert some stuff.
    assert len(data) == bgs
    assert set(list(data)) == columns

def test_census_tracts():
    AL = us.states.AL
    data = census(AL, geometry="tract", table="P4")
    columns = {"GEOID20"} | set(variables("P4").values())
    tracts = 1437

    assert len(data) == tracts
    assert set(list(data)) == columns
    
    # Download additional variables and verify whether they match appropriately.
    data = census(AL, table="P2", columns={"P2_003N": "NHISP"}, geometry="tract")
    columns = {"GEOID20", "NHISP"}

    assert len(data) == tracts
    assert set(list(data)) == columns

    # Now download *more* additional data and verify whether they match
    # appropriately.
    vars = variables("P3")
    varnames = {
        var: name
        for var, name in vars.items() if "WHITE" in name or "BLACK" in name
    }
    columns = {"GEOID20"} | set(varnames.values())

    # Get the data.
    data = census(AL, table="P3", columns=varnames, geometry="tract")

    assert len(data) == tracts
    assert set(list(data)) == columns

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
    state = us.states.WI
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
    test_cvap_bgs()
 