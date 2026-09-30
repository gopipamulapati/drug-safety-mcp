"""Offline mode: an httpx transport that answers openFDA-shaped requests from bundled sample data.

Used by the test-suite and by ``DRUG_SAFETY_OFFLINE=1`` for demos without network access.
The bundled data is illustrative (paraphrased, abbreviated, made-up counts) - never cite it.
"""

from __future__ import annotations

import json
import re
from importlib import resources

import httpx

_NAME_RE = re.compile(r'generic_name:"([^"]+)"')


def load_sample() -> dict:
    return json.loads(resources.files(__package__).joinpath("sample_data/offline_sample.json").read_text())


def _resolve(name: str, labels: dict) -> str | None:
    for key, lab in labels.items():
        brands = {b.lower() for b in lab["openfda"].get("brand_name", [])}
        if name == key or name in brands:
            return key
    return None


def offline_transport(data: dict | None = None) -> httpx.MockTransport:
    data = data or load_sample()

    def handler(request: httpx.Request) -> httpx.Response:
        search = request.url.params.get("search", "")
        m = _NAME_RE.search(search)
        key = _resolve(m.group(1), data["labels"]) if m else None
        not_found = httpx.Response(404, json={"error": {"code": "NOT_FOUND", "message": "No matches found!"}})
        if key is None:
            return not_found
        path = request.url.path
        if path.endswith("/drug/label.json"):
            return httpx.Response(200, json={"results": [data["labels"][key]]})
        if path.endswith("/drug/event.json"):
            ev = data["events"][key]
            if request.url.params.get("count"):
                limit = int(request.url.params.get("limit", 10))
                counts = ev["counts"]
                if "serious:1" in search:
                    counts = [{"term": c["term"], "count": c["count"] // 2} for c in counts]
                return httpx.Response(200, json={"results": counts[:limit]})
            return httpx.Response(200, json={"meta": {"results": {"total": ev["total"]}}, "results": [{}]})
        if path.endswith("/drug/enforcement.json"):
            rows = data["recalls"].get(key) or []
            return httpx.Response(200, json={"results": rows}) if rows else not_found
        return httpx.Response(400, json={"error": {"code": "BAD_REQUEST"}})

    return httpx.MockTransport(handler)
