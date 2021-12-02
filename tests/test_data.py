
from evaltools.data import cvap, acs5, census
import us

def test_cvap_tracts():
    al = us.states.AL
    data = cvap(al, geometry="tract")

    # Set some testing variables.
    columns = {
        "TRACT10", "CVAP19", "NHCVAP19", "AICVAP19", "ACVAP19", "BCVAP19", "NHPICVAP19",
        "WCVAP19", "AIWCVAP19", "AWCVAP19", "BWCVAP19", "AIBCVAP19", "OCVAP19", "HCVAP19"
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
        "BLOCKGROUP10", "CVAP19", "NHCVAP19", "AICVAP19", "ACVAP19", "BCVAP19", "NHPICVAP19",
        "WCVAP19", "AIWCVAP19", "AWCVAP19", "BWCVAP19", "AIBCVAP19", "OCVAP19", "HCVAP19"
    }
    tracts = 3438

    # Do some assert-ing.
    assert set(list(data)) == columns
    assert len(data) == tracts

def test_acs5_tracts():
    state = us.states.AL
    data = acs5(state, geometry="tract")

    tracts = 1181
    columns = {
        "TOTPOP19", "WHITE19", "BLACK19", "AMIN19", "ASIAN19", "NHPI19", "OTH19",
        "2MORE19", "NHISP19", "WVAP19", "BVAP19", "AMINVAP19", "ASIANVAP19",
        "NHPIVAP19", "OTHVAP19", "2MOREVAP19", "HVAP19", "TRACT10", "VAP19"
    }

    # Assert some stuff.
    assert len(data) == tracts
    assert set(list(data)) == columns

def test_acs5_bgs():
    state = us.states.AL
    data = acs5(state, geometry="block group")
    bgs = 3438
    columns = {
        "TOTPOP19", "WHITE19", "BLACK19", "AMIN19", "ASIAN19", "NHPI19", "OTH19",
        "2MORE19", "NHISP19", "WVAP19", "BVAP19", "AMINVAP19", "ASIANVAP19",
        "NHPIVAP19", "OTHVAP19", "2MOREVAP19", "HVAP19", "BLOCKGROUP10", "VAP19"
    }

    # Assert some stuff.
    assert len(data) == bgs
    assert set(list(data)) == columns

def test_census():
    state = us.states.AL
    census(state, geometry="tract")
    

if __name__ == "__main__":
    test_census()

