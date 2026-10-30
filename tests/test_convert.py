import json

import numpy as np
import pytest

from sureenough.paths import RAW

SUB01 = RAW / "osf-anp5v" / "sub-01.zip"


@pytest.mark.skipif(not SUB01.exists(), reason="raw THINGS-EEG2 data not present")
def test_convert_subject_shapes(tmp_path):
    from sureenough.data.convert import convert_subject

    shapes = convert_subject(SUB01, tmp_path)
    assert shapes == {"train": (16540, 4, 17, 100), "test": (200, 80, 17, 100)}
    test = np.load(tmp_path / "test.npy", mmap_mode="r")
    assert test.dtype == np.float32 and test.shape == (200, 80, 17, 100)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert set(meta["ch_names"]) == {
        "O1", "Oz", "O2", "PO7", "PO3", "POz", "PO4", "PO8",
        "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    }
    assert len(meta["times"]) == 100 and abs(meta["times"][0] + 0.2) < 1e-6
