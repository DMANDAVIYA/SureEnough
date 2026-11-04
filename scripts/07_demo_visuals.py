import base64
import json
from concurrent.futures import ThreadPoolExecutor

from sureenough.analysis.splits import split
from sureenough.config import load_frozen
from sureenough.data.concepts import load_concepts
from sureenough.demo import demo_subjects, load_streams, pinned_lambda
from sureenough.meta_api import COST_IMAGE, MetaAPIError, MetaClient, image_edit_payload, output_text
from sureenough.narrate.display import display_claim
from sureenough.narrate.spark import narrate, settle
from sureenough.paths import OUTPUTS
from sureenough.visuals.cc0 import cc0_image, data_url
from sureenough.visuals.conditions import condition_members
from sureenough.visuals.impression import impression_prompt

IMPRESSIONS = OUTPUTS / "visuals" / "impressions"
NARRATION = OUTPUTS / "narration"
SAMPLES = 3
MAX_REFERENCES = 4
MAX_NAMES = 12
WORKERS = 16


class DemoVisuals:
    def __init__(self, client: MetaClient, concepts, cfg: dict):
        self.client, self.concepts, self.cfg = client, concepts, cfg
        self.vocab = {n.lower() for n in concepts.all_names} | set(concepts.categories53)

    def names(self, members: list[int]) -> list[str]:
        return [self.concepts.test_names[j].lower() for j in members[:MAX_NAMES]]

    def impress(self, stem, members: list[int]) -> None:
        refused = stem.with_name(f"{stem.name}.refused")
        if stem.with_name(f"{stem.name}_{SAMPLES - 1}.webp").exists() or refused.exists():
            return
        claim = display_claim(members, self.concepts)
        refs = [data_url(cc0_image(self.concepts.test_things_ids[j], 512)) for j in members[:MAX_REFERENCES]]
        payload = image_edit_payload(impression_prompt(claim, self.names(members)), refs, n=SAMPLES)
        try:
            body = self.client.image_edit(payload, est_cost=COST_IMAGE * SAMPLES)
        except MetaAPIError as error:
            if not error.policy_refusal:
                raise
            refused.touch()
            return
        for k, item in enumerate(body["data"]):
            stem.with_name(f"{stem.name}_{k}.webp").write_bytes(base64.b64decode(item["b64_json"]))

    def process(self, probs, subject: int, image: int, lam: float) -> None:
        conds = condition_members(probs, image, lam, self.cfg["K"], self.cfg["m"])
        members = conds["hedged"][0]
        narration_path = NARRATION / f"sub-{subject:02d}" / f"{image:03d}.json"
        claim = display_claim(members, self.concepts)
        if narration_path.exists():
            stored = json.loads(narration_path.read_text())
            text, ok = settle(output_text(stored["response"]).strip(), claim, self.names(members), self.vocab)
            narration_path.write_text(json.dumps(stored | {"claim": claim, "text": text, "faithful": ok}))
        else:
            narration_path.write_text(json.dumps({"claim": claim, **narrate(self.client, claim, self.names(members), self.vocab)}))
        base = IMPRESSIONS / f"sub-{subject:02d}"
        self.impress(base / f"{image:03d}_hedged", members)
        if conds["top1"][0][0] != image:
            for cond in ("top1", "answer_key", "blind_guess"):
                self.impress(base / f"{image:03d}_{cond}", conds[cond][0])


def main() -> None:
    cfg = load_frozen()
    visuals = DemoVisuals(MetaClient(), load_concepts(), cfg)
    jobs = []
    for subject in demo_subjects():
        probs = load_streams(subject)
        lam = pinned_lambda(subject)
        for folder in (NARRATION, IMPRESSIONS):
            (folder / f"sub-{subject:02d}").mkdir(parents=True, exist_ok=True)
        _, eval_idx = split(subject, 0, len(probs))
        jobs += [(probs, subject, int(i), lam) for i in eval_idx]
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        list(pool.map(lambda job: visuals.process(*job), jobs))
    print(f"done, spent ${visuals.client.spent():.2f}", flush=True)


if __name__ == "__main__":
    main()
