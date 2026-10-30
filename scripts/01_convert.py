import time

from sureenough.data.convert import convert_image_metadata, convert_subject
from sureenough.paths import DERIVED, RAW, subject_ids


def main() -> None:
    for s in subject_ids():
        out = DERIVED / "eeg" / f"sub-{s:02d}"
        if (out / "train.npy").exists() and (out / "test.npy").exists() and (out / "meta.json").exists():
            print(f"sub-{s:02d}: already converted")
            continue
        t0 = time.time()
        shapes = convert_subject(RAW / "osf-anp5v" / f"sub-{s:02d}.zip", out)
        print(f"sub-{s:02d}: {shapes} in {time.time() - t0:.0f}s", flush=True)
    convert_image_metadata(RAW / "osf-y63gw" / "image_metadata.npy", DERIVED / "image_meta.json")
    print("image_meta.json written")


if __name__ == "__main__":
    main()
