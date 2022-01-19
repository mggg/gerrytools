
from evaltools.data import cvap, acs5, census, variables
import us

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

if __name__ == "__main__":
    test_cvap_bgs()
 