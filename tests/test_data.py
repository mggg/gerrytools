
from evaltools.data import cvap, acs5, census
import us

def test_cvap_tracts():
    al = us.states.AL
    data = cvap(al, geometry="tract")

    # Set some testing variables.
    columns = {
        "TRACT10", "CVAP19", "NHCVAP19", "NHAICVAP19", "NHACVAP19", "NHBCVAP19",
        "NHNHPICVAP19", "NHWCVAP19", "NHAIWCVAP19", "NHAWCVAP19", "NHBWCVAP19",
        "NHAIBCVAP19", "NHOCVAP19", "HCVAP19", "POCCVAP19"
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
        "NHAIBCVAP19", "NHOCVAP19", "HCVAP19", "POCCVAP19"
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

def test_census_tracts():
    state = us.states.AL
    data = census(state, geometry="tract", table="P4")
    columns = {
        "GEOID20", 'VAP20', 'HVAP20', 'NHWHITEVAP20', 'NHBLACKVAP20', 'NHAMINVAP20',
        'NHASIANVAP20', 'NHNHPIVAP20', 'NHOTHVAP20', 'NHWHITEBLACKVAP20',
        'NHWHITEAMINVAP20', 'NHWHITEASIANVAP20', 'NHWHITENHPIVAP20',
        'NHWHITEOTHVAP20', 'NHBLACKAMINVAP20', 'NHBLACKASIANVAP20',
        'NHBLACKNHPIVAP20', 'NHBLACKOTHVAP20', 'NHAMINASIANVAP20',
        'NHAMINNHPIVAP20', 'NHAMINOTHVAP20', 'NHASIANNHPIVAP20',
        'NHASIANOTHVAP20', 'NHNHPIOTHVAP20', 'NHWHITEBLACKAMINVAP20',
        'NHWHITEBLACKASIANVAP20', 'NHWHITEBLACKNHPIVAP20', 'NHWHITEBLACKOTHVAP20',
        'NHWHITEAMINASIANVAP20', 'NHWHITEAMINNHPIVAP20', 'NHWHITEAMINOTHVAP20',
        'NHWHITEASIANNHPIVAP20', 'NHWHITEASIANOTHVAP20', 'NHWHITENHPIOTHVAP20',
        'NHBLACKAMINASIANVAP20', 'NHBLACKAMINNHPIVAP20', 'NHBLACKAMINOTHVAP20',
        'NHBLACKASIANNHPIVAP20', 'NHBLACKASIANOTHVAP20', 'NHBLACKNHPIOTHVAP20',
        'NHAMINASIANNHPIVAP20', 'NHAMINASIANOTHVAP20', 'NHAMINNHPIOTHVAP20',
        'NHASIANNHPIOTHVAP20', 'NHWHITEBLACKAMINASIANVAP20',
        'NHWHITEBLACKAMINNHPIVAP20', 'NHWHITEBLACKAMINOTHVAP20',
        'NHWHITEBLACKASIANNHPIVAP20', 'NHWHITEBLACKASIANOTHVAP20',
        'NHWHITEBLACKNHPIOTHVAP20', 'NHWHITEAMINASIANNHPIVAP20',
        'NHWHITEAMINASIANOTHVAP20', 'NHWHITEAMINNHPIOTHVAP20',
        'NHWHITEASIANNHPIOTHVAP20', 'NHBLACKAMINASIANNHPIVAP20',
        'NHBLACKAMINASIANOTHVAP20', 'NHBLACKAMINNHPIOTHVAP20',
        'NHBLACKASIANNHPIOTHVAP20', 'NHAMINASIANNHPIOTHVAP20',
        'NHWHITEBLACKAMINASIANNHPIVAP20', 'NHWHITEBLACKAMINASIANOTHVAP20',
        'NHWHITEBLACKAMINNHPIOTHVAP20', 'NHWHITEBLACKASIANNHPIOTHVAP20',
        'NHWHITEAMINASIANNHPIOTHVAP20', 'NHBLACKAMINASIANNHPIOTHVAP20',
        'NHWHITEBLACKAMINASIANNHPIOTHVAP20'
    }
    tracts = 1437

    assert len(data) == tracts
    assert set(list(data)) == columns

if __name__ == "__main__":
    test_cvap_bgs()

 