import numpy as np

from sureenough.meta_api import output_text
from sureenough.narrate.naming import pe_guess
from sureenough.narrate.spark import narrate
from sureenough.visuals.impression import impression_prompt

VOCAB = {"wolf", "husky", "coyote", "dog", "animal", "sofa"}


class FakeClient:
    def __init__(self, reply: str):
        self.reply = reply
        self.payloads = []

    def responses(self, payload, est_cost):
        self.payloads.append(payload)
        return {"output": [{"type": "message", "content": [{"type": "output_text", "text": self.reply}]}]}


def _claim(kind, text, category=None):
    return {"kind": kind, "text": text, "category": category, "members": [0, 1, 2]}


def test_output_text_joins_message_parts():
    body = {"output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "a"}, {"type": "output_text", "text": "b"}]}]}
    assert output_text(body) == "ab"


def test_faithful_narration_is_kept():
    client = FakeClient("It is a wolf, a husky or a coyote.")
    out = narrate(client, _claim("list", "It's one of these 3."), ["wolf", "husky", "coyote"], VOCAB)
    assert out["faithful"] and out["text"] == "It is a wolf, a husky or a coyote."
    assert client.payloads[0]["store"] is False


def test_unfaithful_narration_falls_back_to_claim_text():
    client = FakeClient("It is probably a dog.")
    out = narrate(client, _claim("list", "It's one of these 3."), ["wolf", "husky", "coyote"], VOCAB)
    assert not out["faithful"] and out["text"] == "It's one of these 3."


def test_category_claim_allows_its_category():
    client = FakeClient("It's an animal.")
    out = narrate(client, _claim("category", "It's an animal.", "animal"), ["wolf", "husky", "coyote"], VOCAB)
    assert out["faithful"]


def test_impression_prompt_names_only_the_claim():
    prompt = impression_prompt(_claim("list", "It's one of these 3."), ["wolf", "husky", "coyote"])
    assert all(n in prompt for n in ("wolf", "husky", "coyote"))
    assert "probably" not in prompt.lower()
    category = impression_prompt(_claim("category", "It's an animal.", "animal"), ["wolf", "husky", "coyote"])
    assert "animal" in category and "wolf" not in category
    single = impression_prompt({"kind": "single", "text": "It's a sofa.", "category": None, "members": [3]}, ["sofa"])
    assert "a sofa" in single


def test_pe_guess_picks_closest_name():
    text = np.eye(3, dtype=np.float32)
    assert pe_guess(np.array([0.1, 0.9, 0.2]), text, ["wolf", "sofa", "dog"]) == "sofa"
