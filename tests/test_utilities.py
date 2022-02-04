
from evaltools.utilities import rename, JSONtoObject, jsonify
from pydantic import ValidationError
from pathlib import Path
import os
import shutil

root = Path(os.getcwd()) / Path("tests/test-resources/")

def test_rename():
    # Create a new directory with a single, differently-named file in it.
    testdir = root / "to-be-renamed"
    testfile = testdir / "needs-renaming.ext"

    # Specify new directories.
    newdir = root / "renamed"
    newfile = root / "renamed/renamed.ext"

    # Clear things up first.
    if testdir.exists(): shutil.rmtree(testdir)
    if newdir.exists(): shutil.rmtree(newdir)
    
    testdir.mkdir()
    testfile.touch()

    # Rename!
    rename(str(testdir), "renamed")

    # Check if the renamed directory and file exist, and the old ones don't.
    assert not testdir.exists()
    assert not testfile.exists()
    assert (root/"renamed").exists()
    assert (root/"renamed/renamed.ext").exists()

def test_json():
    # Read in plans.
    goods = jsonify(root / "test-plans.json")

    # Verify that we have a good list of plans.
    assert type(goods) is list
    assert all(type(good) is JSON for good in goods)
    
    # Import some bad ones, but make sure we catch the validation error.
    try: bads = jsonify(root / "test-plans-bad.json")
    except Exception as e:
        assert type(e) is ValidationError

if __name__ == "__main__":
    root = Path(os.getcwd()) / Path("test-resources/")
    test_json()
