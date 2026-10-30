import json
import os
import random
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import dotenv_values

from sureenough.paths import OUTPUTS, REPO

BASE = "https://api.meta.ai/v1"
SPARK, SAM, IMAGE = "muse-spark-1.3", "sam-3.1", "muse-image-1.0"
ALLOWED_MODELS = frozenset({SPARK, SAM, IMAGE})
COST_SAM = 0.0025
COST_IMAGE = 0.01
RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
TRIES = 5


def cost_spark(in_tok: int, out_tok: int) -> float:
    return in_tok * 1.25e-6 + out_tok * 4.25e-6


class BudgetExceeded(RuntimeError):
    pass


class MetaAPIError(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"HTTP {status}: {text[:500]}")
        self.status = status


class MetaClient:
    def __init__(self, key: str | None = None, ledger_path=OUTPUTS / "api_ledger.jsonl", cap_usd: float = 60.0):
        key = key or dotenv_values(REPO / ".env").get("META_API_KEY")
        if not key:
            raise RuntimeError("META_API_KEY missing from .env")
        self.ledger_path, self.cap_usd = Path(ledger_path), cap_usd
        self._session = requests.Session()
        self._session.headers["Authorization"] = f"Bearer {key}"
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"MetaClient(key=<redacted>, ledger_path={str(self.ledger_path)!r}, cap_usd={self.cap_usd})"

    def responses(self, payload: dict, est_cost: float) -> dict:
        return self._call("/responses", {**payload, "stream": False, "store": False}, est_cost)

    def image_edit(self, payload: dict, est_cost: float) -> dict:
        return self._call("/images/edits", payload, est_cost)

    def spent(self) -> float:
        cost: dict[str, float] = {}
        if self.ledger_path.exists():
            for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cost[rec["id"]] = rec["est_cost"] if rec["status"] == "pending" else rec["cost"]
        return sum(cost.values())

    def _log(self, rec: dict) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **rec}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _call(self, path: str, payload: dict, est_cost: float) -> dict:
        model = payload.get("model")
        if model not in ALLOWED_MODELS:
            raise ValueError(f"model {model!r} not allowed")
        if est_cost < 0:
            raise ValueError("est_cost must be >= 0")
        cid = uuid.uuid4().hex
        with self._lock:
            if round(self.spent() + est_cost, 9) > self.cap_usd:
                raise BudgetExceeded(f"spent ${self.spent():.4f} + ${est_cost:.4f} > cap ${self.cap_usd:.2f}")
            self._log({"id": cid, "status": "pending", "endpoint": path, "model": model, "est_cost": est_cost})
        for attempt in range(TRIES):
            r = self._session.post(BASE + path, json=payload, timeout=(10, 300))
            if r.status_code not in RETRY_STATUS or attempt == TRIES - 1:
                break
            time.sleep(_delay(r, attempt))
        if r.status_code != 200:
            self._log({"id": cid, "status": "failed", "http": r.status_code,
                       "cost": est_cost if r.status_code >= 500 else 0.0})
            raise MetaAPIError(r.status_code, r.text)
        body = r.json()
        self._log({"id": cid, "status": "done", "cost": _actual_cost(model, body, est_cost),
                   "usage": body.get("usage"), "attempts": attempt + 1})
        return body


def _delay(r: requests.Response, attempt: int) -> float:
    try:
        return min(float(r.headers["Retry-After"]), 60.0)
    except (KeyError, ValueError):
        return min(2.0**attempt + random.random(), 30.0)


def _actual_cost(model: str, body: dict, est_cost: float) -> float:
    usage = body.get("usage") or {}
    if model == SPARK and "input_tokens" in usage:
        return cost_spark(usage["input_tokens"], usage.get("output_tokens", 0))
    if model == IMAGE and isinstance(body.get("data"), list):
        return COST_IMAGE * len(body["data"])
    return est_cost


def sam_payload(noun: str, image_data_url: str) -> dict:
    content = [{"type": "input_text", "text": noun}, {"type": "input_image", "image_url": image_data_url}]
    return {"model": SAM, "input": [{"type": "message", "role": "user", "content": content}],
            "metadata": {"include_confidence": "true"}, "stream": False, "store": False}


def spark_payload(system: str, user: str) -> dict:
    return {"model": SPARK, "instructions": system, "input": user, "stream": False, "store": False}


def image_edit_payload(prompt: str, image_data_urls: list[str], n: int = 3) -> dict:
    if not 1 <= len(image_data_urls) <= 10:
        raise ValueError("Muse Image takes 1-10 reference images")
    return {"model": IMAGE, "prompt": prompt, "images": [{"image_url": u} for u in image_data_urls],
            "n": n, "size": "1024x1024", "output_format": "webp", "reasoning_strength": "low",
            "tool_enablement": {"enable_image_search": False, "enable_web_search": False, "enable_shell": False}}


def output_text(body: dict) -> str:
    return "".join(
        part.get("text", "")
        for item in body.get("output", [])
        for part in item.get("content") or []
        if part.get("type") == "output_text"
    )
