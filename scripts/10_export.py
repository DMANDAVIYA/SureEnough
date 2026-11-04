import hashlib
import json
import shutil
import zipfile

import numpy as np
from PIL import Image
from sklearn.linear_model import Ridge

from sureenough.analysis.splits import split
from sureenough.calibrate.ltt import calibrate
from sureenough.calibrate.sets import LAM_GRID
from sureenough.config import load_frozen
from sureenough.data.concepts import load_concepts
from sureenough.demo import demo_subjects, load_streams
from sureenough.export.web_data import check_bundle, example_payload, naive_eval_risk_u16, table_payload
from sureenough.features.targets import FEATURES, load_targets
from sureenough.meta_api import output_text
from sureenough.models.store import load_decoder
from sureenough.narrate.display import display_claim, shared_category
from sureenough.narrate.naming import pe_guess
from sureenough.paths import DERIVED, OUTPUTS, RAW, REPO, subject_ids
from sureenough.streams.replay import stream_orders
from sureenough.visuals.cc0 import cc0_image
from sureenough.visuals.conditions import stopped_state

OUT = REPO / "web" / "public" / "data"
OCCIPITAL = ("O1", "Oz", "O2")
TRIM = 400


def write(path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))


def save_webp(img: Image.Image, path, side: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.copy()
    img.thumbnail((side, side))
    img.save(path, quality=82)


def export_images(concepts) -> list[dict]:
    entries = []
    for i, (tid, name) in enumerate(zip(concepts.test_things_ids, concepts.test_names)):
        save_webp(cc0_image(tid, 512), OUT / "img" / "tiles" / f"{tid}.webp", 256)
        save_webp(Image.open(OUTPUTS / "visuals" / "cutouts" / f"{tid}.webp"), OUT / "img" / "cutouts" / f"{tid}.webp", 384)
        entries.append({"i": i, "name": name, "things_id": tid, "tile": f"img/tiles/{tid}.webp",
                        "cutout": f"img/cutouts/{tid}.webp", "category": shared_category({tid}, concepts)})
    return entries


def naming_map(target: str):
    if target == "final":
        return lambda v: v
    source = load_targets(target)[0]
    final = np.load(FEATURES / "pe_train_final.npy").astype(np.float32)
    ridge = Ridge(alpha=1.0).fit(source, final)
    return lambda v: ridge.predict(v[None])[0]


def spark_provenance(narration: dict) -> dict:
    body = narration["response"]
    return {"request": narration["request"],
            "response": {"model": body.get("model"), "output_text": output_text(body), "usage": body.get("usage")}}


def sam_provenance(concepts, j: int) -> dict:
    raw = OUTPUTS / "visuals" / "sam_raw"
    tid = concepts.test_things_ids[j]
    path = raw / f"{tid}_described.json" if (raw / f"{tid}_described.json").exists() else raw / f"{tid}.json"
    record = json.loads(path.read_text())
    request = {"model": "sam-3.1", "input": [record["prompt"], "<public-domain CC0 photo>"]}
    response = {"model": record["response"].get("model"), "output_text": output_text(record["response"])[:TRIM] + " …"}
    return {"request": request, "response": response}


def export_subject(s, cfg, results, concepts, text_feats, to_final, demo, muse) -> dict:
    write(OUT / "tables" / f"sub-{s:02d}.json", table_payload(dict(np.load(OUTPUTS / "tables" / f"sub-{s:02d}.npz"))))
    probs = load_streams(s)
    eeg = np.load(DERIVED / "eeg" / f"sub-{s:02d}" / "test.npy", mmap_mode="r")
    ch_names = json.loads((DERIVED / "eeg" / f"sub-{s:02d}" / "meta.json").read_text())["ch_names"]
    channels = [ch_names.index(c) for c in OCCIPITAL]
    decoder = load_decoder(OUTPUTS / "models" / f"sub-{s:02d}.pt")
    lam_idx = results["per_subject"][str(s)]["pinned_lam_idx"]
    lam = float(LAM_GRID[lam_idx])
    cal, ev = split(s, 0, len(probs))
    outcomes = {}
    for i in map(int, ev):
        order = stream_orders(s, i)[0]
        ex = example_payload(probs[i, 0], np.asarray(eeg[i]), order, lam, cfg["K"], cfg["m"], channels)
        members, _ = stopped_state(probs[i, 0], lam, cfg["K"], cfg["m"])
        claim = display_claim(members.tolist(), concepts)
        embedding = decoder.embed(np.asarray(eeg[i])[order[:ex["stopped_at"]]].mean(axis=0, keepdims=True))[0]
        narration_path = OUTPUTS / "narration" / f"sub-{s:02d}" / f"{i:03d}.json"
        narration = json.loads(narration_path.read_text()) if narration_path.exists() else None
        impression_src = OUTPUTS / "visuals" / "impressions" / f"sub-{s:02d}" / f"{i:03d}_hedged_0.webp"
        impression = None
        if muse and s in demo and impression_src.exists():
            impression = f"img/impressions/sub-{s:02d}/{i:03d}.webp"
            save_webp(Image.open(impression_src), OUT / impression, 512)
        ex |= {
            "claim": {k: claim[k] for k in ("kind", "text", "category")},
            "narration": narration["text"] if narration and narration["faithful"] else None,
            "narration_rejected": bool(narration) and not narration["faithful"],
            "provenance": {
                "spark": spark_provenance(narration) if narration else None,
                "sam": sam_provenance(concepts, int(members[0])),
            },
            "pe_guess": pe_guess(to_final(embedding), text_feats, concepts.all_names),
            "impression": impression,
            "truth_in_set": bool(i in set(members.tolist())),
        }
        write(OUT / "examples" / f"sub-{s:02d}" / f"{i:03d}.json", ex)
        outcomes[i] = ex
    return {"cal": cal.tolist(), "eval": ev.tolist(), "lam_idx": lam_idx, "outcomes": outcomes}


def curated_label_holds(label: str, i: int, ex: dict, K: int, m: int) -> bool:
    final = ex["set_size"][ex["stopped_at"] - 1]
    top_pick = ex["top"][ex["stopped_at"] - 1][0][0]
    return {
        "clean stop": ex["claim"]["kind"] == "single" and ex["truth_in_set"],
        "unsure": ex["stopped_at"] == K and final > m,
        "hedged, top pick wrong": ex["truth_in_set"] and final > 1 and top_pick != i,
        "missed": not ex["truth_in_set"],
    }[label]


def curate(subject: int, outcomes: dict, K: int, m: int) -> list[dict]:
    picks = []
    for label in ("clean stop", "unsure", "hedged, top pick wrong", "missed"):
        hit = next((i for i, ex in sorted(outcomes.items()) if curated_label_holds(label, i, ex, K, m)), None)
        if hit is not None:
            picks.append({"subject": subject, "i": hit, "label": label})
    return picks


def ltt_fixture(cfg) -> dict:
    s = subject_ids()[0]
    table = table_payload(dict(np.load(OUTPUTS / "tables" / f"sub-{s:02d}.npz")))
    losses = np.asarray(table["loss_count"]) / cfg["R"]
    splits = []
    for k in range(5):
        cal, ev = split(s, k, len(losses))
        lam_idx = calibrate(losses[cal], cal, cfg["alpha"], cfg["delta"])
        splits.append({"cal": cal.tolist(), "eval": ev.tolist(), "lam_idx": lam_idx,
                       "eval_risk": float(losses[ev, lam_idx].mean()),
                       "naive_eval_risk": naive_eval_risk_u16(table["naive"]["a_true_u16"], table["naive"]["top3_u16"],
                                                              cal, ev, cfg["alpha"], cfg["K"])})
    return {"alpha": cfg["alpha"], "delta": cfg["delta"], "table": table, "splits": splits}


def stimulus_hashes() -> set[str]:
    hashes = set()
    for name in ("test_images.zip", "training_images.zip"):
        with zipfile.ZipFile(RAW / "osf-y63gw" / name) as zf:
            hashes |= {hashlib.sha256(zf.read(n)).hexdigest() for n in zf.namelist() if not n.endswith("/")}
    return hashes


def main() -> None:
    cfg = load_frozen()
    concepts = load_concepts()
    results = json.loads((OUTPUTS / "results.json").read_text())
    if (OUTPUTS / "h5a.json").exists():
        results["h5a"] = json.loads((OUTPUTS / "h5a.json").read_text())
    muse = json.loads((OUTPUTS / "flags.json").read_text())["muse_image"]
    demo = demo_subjects()
    for sub in ("tables", "examples", "img", "fixtures"):
        shutil.rmtree(OUT / sub, ignore_errors=True)
    concept_entries = export_images(concepts)
    text_feats = np.load(FEATURES / "pe_text_final.npy").astype(np.float32)
    to_final = naming_map(cfg["target"])
    pinned, curated = {}, []
    for s in subject_ids():
        exported = export_subject(s, cfg, results, concepts, text_feats, to_final, demo, muse)
        pinned[str(s)] = {k: exported[k] for k in ("cal", "eval", "lam_idx")}
        if s in demo:
            curated += curate(s, exported["outcomes"], cfg["K"], cfg["m"])
        print(f"sub-{s:02d}: exported", flush=True)
    sweep = json.loads((OUTPUTS / "sweep.json").read_text())["ridge"]
    results["pe_vs_clip"] = {"pe": float(np.mean(list(sweep[cfg["target"]].values()))),
                             "clip": float(np.mean(list(sweep["clip_h"].values())))}
    public = {k: v for k, v in results.items() if k != "config"}
    for summary in public["per_subject"].values():
        summary.pop("per_image", None)
    write(OUT / "manifest.json", {
        "alpha": cfg["alpha"], "delta": cfg["delta"], "K": cfg["K"], "m": cfg["m"], "R": cfg["R"],
        "lam": LAM_GRID.tolist(), "subjects": subject_ids(), "demo_subjects": demo, "muse_image": muse,
        "pinned": pinned, "concepts": concept_entries, "results": public, "curated": curated,
    })
    write(OUT / "fixtures" / "ltt_fixture.json", ltt_fixture(cfg))
    print(f"bundle ok: {check_bundle(OUT, stimulus_hashes()):.1f} MB")


if __name__ == "__main__":
    main()
