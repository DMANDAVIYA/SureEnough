import io
import os
import pickle

import numpy as np
import pytest

from sureenough.data.safe_npy import load_pickled_npy


def _npy_with_payload(payload: bytes) -> io.BytesIO:
    buf = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buf, {"descr": "|O", "fortran_order": False, "shape": ()}
    )
    buf.write(payload)
    buf.seek(0)
    return buf


def test_roundtrip_pickled_dict():
    d = {
        "preprocessed_eeg_data": np.arange(6, dtype=np.float64).reshape(2, 3),
        "ch_names": ["O1", "Oz"],
        "times": np.arange(3.0),
    }
    buf = io.BytesIO()
    np.save(buf, d, allow_pickle=True)
    buf.seek(0)
    out = load_pickled_npy(buf)
    assert set(out) == set(d)
    np.testing.assert_array_equal(out["preprocessed_eeg_data"], d["preprocessed_eeg_data"])
    assert out["ch_names"] == ["O1", "Oz"]
    np.testing.assert_array_equal(out["times"], d["times"])


class _Evil:
    def __init__(self, path):
        self.path = path

    def __reduce__(self):
        return (os.system, (f"echo pwned > {self.path}",))


def test_rejects_malicious_pickle(tmp_path):
    marker = tmp_path / "pwned.txt"
    buf = _npy_with_payload(pickle.dumps(_Evil(marker), protocol=2))
    with pytest.raises(pickle.UnpicklingError):
        load_pickled_npy(buf)
    assert not marker.exists()


def test_rejects_non_object_npy():
    buf = io.BytesIO()
    np.save(buf, np.zeros(3))
    buf.seek(0)
    with pytest.raises(ValueError):
        load_pickled_npy(buf)
