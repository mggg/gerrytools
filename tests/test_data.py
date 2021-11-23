
from evaltools.data import raw, cvap
import us

def test_raw():
    al = us.states.AL
    data = cvap(al, geometry="tract10")

    # Set some testing variables.
    columns = {
        "GEOID", "CVAP", "NHCVAP", "AICVAP", "ACVAP", "BCVAP", "NHPICVAP",
        "WCVAP", "AIWCVAP", "AWCVAP", "BWCVAP", "AIBCVAP", "OCVAP", "HCVAP"
    }
    tracts = 1181

    # Do some assert-ing.
    assert set(list(data)) == columns
    assert len(data) == tracts
    

if __name__ == "__main__":
    test_raw()

