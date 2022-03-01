
import us
import jsonlines
from pathlib import Path
import os
import pandas as pd
import json
from evaltools.data import (
    cvap, acs5, census, variables, submissions, tabularized, AssignmentCompressor,
    remap, estimatecvap
)
import us
import geopandas as gpd
import math
from gerrychain import Graph, Partition
import random

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_estimate_cvap():
    al = us.states.AL
    base = gpd.read_file(root/"AL-bgs")

    triplets = [
        ("NHBCVAP19", "BVAP19", "APBVAP20"),
        ("NHACVAP19", "ASIANVAP19", "ASIANVAP20")
    ]

    # Try estimating some CVAP! More rigorous tests should be added later.
    estimated = estimatecvap(base, al, triplets, 1, 1/10)

    # Set the test cases for two random block groups. These were hand-calculated
    # from the set of Alabama test data retrieved from data.mggg.org.
    tests = {
        "010950307012": 19.6857142,
        "010770113001": 2
    }

    # Check two random block groups to check whether they're correct; we ask
    # whether the test value and the reported value are within one millionth of
    # each other.
    for bgid, ground in tests.items():
        row = estimated.loc[estimated["GEOID20"] == bgid]
        assert math.isclose(list(row["NHBCVAP20_EST"])[0], ground, abs_tol=1e-6)

    tests = {
        "010919733003": 0.5993019060318456*0
    }

    # Check that there are no NaNs in the columns. This is important to check because
    # Alabama has a county where there are 0 Asian CVAP19 and 0 Asian VAP19 people.
    assert not estimated["NHACVAP20_EST"].isnull().any()

    for bgid, ground in tests.items():
        assert math.isclose(list(row["NHACVAP20_EST"])[0], ground, abs_tol=1e-6)


def test_cvap_tracts():
    al = us.states.AL
    special = cvap(al, geometry="tract")
    acs = acs5(al, geometry="tract")
    
    # Set some testing variables.
    columns = {
        "TRACT10", "CVAP19", "NHCVAP19", "NHAICVAP19", "NHACVAP19", "NHBCVAP19",
        "NHNHPICVAP19", "NHWCVAP19", "NHAIWCVAP19", "NHAWCVAP19", "NHBWCVAP19",
        "NHAIBCVAP19", "NHOTHCVAP19", "HCVAP19", "POCCVAP19", "CVAP19e", "NHCVAP19e",
        "NHAICVAP19e", "NHACVAP19e", "NHBCVAP19e", "NHNHPICVAP19e", "NHWCVAP19e",
        "NHAIWCVAP19e", "NHAWCVAP19e", "NHBWCVAP19e", "NHAIBCVAP19e", "NHOTHCVAP19e", 
        "HCVAP19e",
    }
    tracts = 1181

    # Do some assert-ing.
    assert set(list(special)) == columns
    assert len(special) == tracts

def test_cvap_bgs():
    al = us.states.AL
    data = cvap(al, geometry="block group")
    
    # Set some testing variables.
    columns = {
        "BLOCKGROUP10", "CVAP19", "NHCVAP19", "NHAICVAP19", "NHACVAP19", "NHBCVAP19",
        "NHNHPICVAP19", "NHWCVAP19", "NHAIWCVAP19", "NHAWCVAP19", "NHBWCVAP19",
        "NHAIBCVAP19", "NHOTHCVAP19", "HCVAP19", "POCCVAP19", "CVAP19e", "NHCVAP19e",
        "NHAICVAP19e", "NHACVAP19e", "NHBCVAP19e", "NHNHPICVAP19e", "NHWCVAP19e",
        "NHAIWCVAP19e", "NHAWCVAP19e", "NHBWCVAP19e", "NHAIBCVAP19e", "NHOTHCVAP19e", 
        "HCVAP19e",
    }
    bgs = 3438

    # Do some assert-ing.
    assert set(list(data)) == columns
    assert len(data) == bgs

def test_acs5_tracts():
    AL = us.states.AL
    data = acs5(AL, geometry="tract")

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

    # Also verify that the CVAP data reported from the ACS are within the margin
    # of error reported by the ACS Special Tabulation.
    special = cvap(AL)
    cvapcols = [c for c in list(data) if "CVAP" in c]
    data = data.rename({c: c + "ACS" for c in cvapcols}, axis=1)
    joined = special.merge(data, on="TRACT10")

    # Create a column for the absolute difference between the ACS-reported and
    # Special-Tab reported CVAP numbers, and assert whether the difference falls
    # within the margin of error.
    joined["DIFF"] = (joined["CVAP19"] - joined["CVAP19ACS"]).abs()
    joined["WITHINMOE"] = joined["DIFF"] <= joined["CVAP19e"]

    assert joined["WITHINMOE"].all()

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


# These test statistics are taken from the Alabama 2020 PL94-171 P1 through P4
# tables from the Census Data Explorer (https://data.census.gov/cedsci/). We then
# use the `census()` method to retrieve Alabama 2020 PL94-171 at all three levels
# of geography, checking that the columns on the data and the columns below sum
# to the same values.
CENSUSTESTDATA = {
    "P1": [
        ("TOTPOP20", 5024279),
        ("WHITEASIANPOP20", 18510)
    ],
    "P2": [
        ("NHWHITEAMINASIANPOP20", 821),
        ("HPOP20", 264047)
    ],
    "P3": [
        ("WHITEASIANOTHVAP20", 201),
        ("VAP20", 3917166)
    ],
    "P4": [
        ("NHBLACKASIANOTHVAP20", 31),
        ("HVAP20", 166856)
    ]
}


def test_census_tracts():
    # Get a test set of data on Alabama.
    AL = us.states.AL
    tracts = 1437

    # Get the data for each table and verify that the values are correct.
    for table, cases in CENSUSTESTDATA.items():
        data = census(AL, table=table, geometry="tract")
        columns = set(variables(table).values()) | {"GEOID20"}
        
        # Assert we have the correct number of values and the correct columns.
        assert len(data) == tracts
        assert set(list(data)) == columns

        # For each test case, confirm that we have the correct sum.
        for column, correct in cases: assert data[column].sum() == correct


def test_census_bgs():
    # Get a test set of data on Alabama.
    AL = us.states.AL
    bgs = 3925

    # Get the data for each table and verify that the values are correct.
    for table, cases in CENSUSTESTDATA.items():
        data = census(AL, table=table, geometry="block group")
        columns = set(variables(table).values()) | {"GEOID20"}
        
        # Assert we have the correct number of values and the correct columns.
        assert len(data) == bgs
        assert set(list(data)) == columns

        # For each test case, confirm that we have the correct sum.
        for column, correct in cases: assert data[column].sum() == correct


def test_census_blocks():
    AL = us.states.AL
    blocks = 185976

    # Get the data for each table and verify that the values are correct.
    for table, cases in CENSUSTESTDATA.items():
        data = census(AL, table=table, geometry="block")
        columns = set(variables(table).values()) | {"GEOID20"}
        
        # Assert we have the correct number of values and the correct columns.
        assert len(data) == blocks
        assert set(list(data)) == columns

        # For each test case, confirm that we have the correct sum.
        for column, correct in cases: assert data[column].sum() == correct

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
    root = Path(os.getcwd()) / Path("test-resources/")
