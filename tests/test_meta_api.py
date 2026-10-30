import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from sureenough import meta_api as m

KEY = "LLM_test_secret_do_not_log_123"
PNG = "data:image/png;base64,iVBORw0KGgo="
FIXTURES = Path(__file__).parent / "fixtures" / "api"


def _resp(status=200, body=None, headers=None):
    r = requests.Response()
    r.status_code = status
    r._content = json.dumps(body if body is not None else {}).encode()
    r.headers.update(headers or {})
    return r


@pytest.fixture
def fake(monkeypatch):
    f = SimpleNamespace(queue=[], calls=[], sleeps=[])

    def post(session, url, **kw):
        f.calls.append({"url": url, "headers": dict(session.headers), **kw})
        return f.queue.pop(0) if f.queue else _resp()

    monkeypatch.setattr(requests.Session, "post", post)
    monkeypatch.setattr(m.time, "sleep", f.sleeps.append)
    return f


@pytest.fixture
def client(tmp_path):
    return m.MetaClient(key=KEY, ledger_path=tmp_path / "ledger.jsonl")


def _ledger(c):
    return [json.loads(l) for l in Path(c.ledger_path).read_text().splitlines()]


def test_budget_exceeded_raises_before_any_network(client, fake):
    client.responses(m.sam_payload("dog", PNG), est_cost=59.995)
    assert client.spent() == pytest.approx(59.995)
    fake.calls.clear()
    n_lines = len(_ledger(client))
    with pytest.raises(m.BudgetExceeded):
        client.image_edit(m.image_edit_payload("a dog", [PNG]), est_cost=0.03)
    assert fake.calls == []
    assert len(_ledger(client)) == n_lines


def test_spend_exactly_at_cap_is_allowed(client, fake):
    client.responses(m.sam_payload("dog", PNG), est_cost=59.97)
    fake.queue.append(_resp(body={"data": [{"b64_json": "x"}] * 3}))
    client.image_edit(m.image_edit_payload("a dog", [PNG]), est_cost=0.03)
    assert client.spent() == pytest.approx(60.0)


def test_retry_429_429_200_spends_once(client, fake):
    fake.queue += [_resp(429, headers={"Retry-After": "2"}), _resp(429), _resp(200, {"output": []})]
    before = client.spent()
    client.responses(m.sam_payload("dog", PNG), est_cost=m.COST_SAM)
    assert len(fake.calls) == 3
    statuses = [l["status"] for l in _ledger(client)]
    assert statuses.count("pending") == 1 and statuses.count("done") == 1 and len(statuses) == 2
    assert client.spent() - before == pytest.approx(m.COST_SAM)
    assert fake.sleeps[0] == 2 and len(fake.sleeps) == 2


def test_retries_exhausted_after_five_tries(client, fake):
    fake.queue += [_resp(503)] * 6
    with pytest.raises(m.MetaAPIError):
        client.responses(m.sam_payload("dog", PNG), est_cost=m.COST_SAM)
    assert len(fake.calls) == 5
    assert _ledger(client)[-1]["status"] == "failed"
    assert client.spent() == pytest.approx(m.COST_SAM)


def test_client_error_not_retried_and_not_counted(client, fake):
    fake.queue.append(_resp(400, {"error": {"message": "bad"}}))
    with pytest.raises(m.MetaAPIError):
        client.responses(m.sam_payload("dog", PNG), est_cost=m.COST_SAM)
    assert len(fake.calls) == 1
    assert client.spent() == 0.0


def test_crash_mid_call_counts_pending_spend(client, monkeypatch):
    def boom(self, url, **kw):
        raise requests.ConnectionError("network died")
    monkeypatch.setattr(requests.Session, "post", boom)
    with pytest.raises(requests.ConnectionError):
        client.responses(m.sam_payload("dog", PNG), est_cost=m.COST_SAM)
    assert [l["status"] for l in _ledger(client)] == ["pending"]
    assert client.spent() == pytest.approx(m.COST_SAM)


def test_spark_cost_comes_from_usage(client, fake):
    fake.queue.append(_resp(body={"usage": {"input_tokens": 1000, "output_tokens": 1000}}))
    client.responses(m.spark_payload("Be brief.", "Say OK"), est_cost=0.01)
    assert client.spent() == pytest.approx(m.cost_spark(1000, 1000)) == pytest.approx(0.0055)


def test_responses_forces_store_and_stream_off(client, fake):
    client.responses({**m.spark_payload("s", "u"), "store": True, "stream": True}, est_cost=0.001)
    call = fake.calls[0]
    assert call["url"] == "https://api.meta.ai/v1/responses"
    assert call["json"]["store"] is False and call["json"]["stream"] is False
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"


def test_image_edit_posts_raw_json(client, fake):
    client.image_edit(m.image_edit_payload("a dog", [PNG], n=1), est_cost=0.01)
    call = fake.calls[0]
    assert call["url"] == "https://api.meta.ai/v1/images/edits"
    assert call["json"]["images"] == [{"image_url": PNG}] and "files" not in call


def test_disallowed_model_rejected_before_network(client, fake):
    with pytest.raises(ValueError):
        client.responses({**m.spark_payload("s", "u"), "model": "muse-spark-1.3-contributor"}, 0.001)
    assert fake.calls == []


def test_key_never_in_repr_or_ledger(client, fake):
    assert KEY not in repr(client) and KEY not in str(client)
    client.responses(m.spark_payload("s", "u"), est_cost=0.001)
    assert KEY not in Path(client.ledger_path).read_text()


@pytest.mark.parametrize("payload", [
    m.sam_payload("dog", PNG),
    m.spark_payload("system", "user"),
    m.image_edit_payload("a dog", [PNG]),
], ids=["sam", "spark", "image_edit"])
def test_payload_models_are_standard(payload):
    assert payload["model"] in m.ALLOWED_MODELS
    assert not payload["model"].endswith("-contributor")
    if payload["model"] != m.IMAGE:
        assert payload["store"] is False


def test_image_edit_payload_switches():
    p = m.image_edit_payload("a dog", [PNG, PNG])
    assert p["tool_enablement"] == {
        "enable_image_search": False, "enable_web_search": False, "enable_shell": False}
    assert (p["reasoning_strength"], p["output_format"], p["size"], p["n"]) == ("low", "webp", "1024x1024", 3)
    assert p["images"] == [{"image_url": PNG}] * 2


def test_sam_payload_shape():
    p = m.sam_payload("dog", PNG)
    content = p["input"][0]["content"]
    assert content == [{"type": "input_text", "text": "dog"}, {"type": "input_image", "image_url": PNG}]
    assert p["metadata"]["include_confidence"] == "true"


def test_recorded_fixtures_are_key_free():
    files = sorted(FIXTURES.glob("*.json")) if FIXTURES.is_dir() else []
    if not files:
        pytest.skip("no recorded API responses yet (tests/fixtures/api/ is empty)")
    for f in files:
        text = f.read_text(encoding="utf-8")
        json.loads(text)
        assert "LLM_" not in text and "Bearer" not in text, f.name
