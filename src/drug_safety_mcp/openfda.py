"""Async openFDA client with input sanitising, retries, rate-limit handling and a TTL cache.

Docs: https://open.fda.gov/apis/  (an API key is optional; it raises the daily quota)
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

import httpx

BASE_URL = "https://api.fda.gov"
_SAFE = re.compile(r"[^a-z0-9 \-]")


class OpenFDAError(RuntimeError):
    pass


class NotFound(OpenFDAError):
    pass


def clean_name(name: str) -> str:
    """Lower-case and strip anything that could alter the Lucene query (quotes, colons, operators)."""
    cleaned = _SAFE.sub("", name.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned or len(cleaned) > 80:
        raise ValueError(f"invalid drug name: {name!r}")
    return cleaned


def drug_query(field_prefix: str, name: str) -> str:
    """Match generic OR brand name. Spaces are sent URL-encoded as '+', which openFDA reads as OR.

    e.g. (openfda.generic_name:"metformin" openfda.brand_name:"metformin")
    """
    n = clean_name(name)
    return f'({field_prefix}.generic_name:"{n}" {field_prefix}.brand_name:"{n}")'


class TTLCache:
    def __init__(self, ttl: float = 3600, max_items: int = 512) -> None:
        self.ttl, self.max_items = ttl, max_items
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        hit = self._data.get(key)
        if hit and time.monotonic() - hit[0] < self.ttl:
            return hit[1]
        self._data.pop(key, None)
        return None

    def set(self, key: str, value: Any) -> None:
        if len(self._data) >= self.max_items:
            self._data.pop(next(iter(self._data)))
        self._data[key] = (time.monotonic(), value)


class OpenFDAClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = BASE_URL,
        timeout: float = 20.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.getenv("OPENFDA_API_KEY")
        self.max_retries = max_retries
        self.cache = cache or TTLCache()
        self._http = httpx.AsyncClient(base_url=base_url, timeout=timeout, transport=transport)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get(self, endpoint: str, **params: Any) -> dict:
        if self.api_key:
            params["api_key"] = self.api_key
        key = endpoint + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()) if k != "api_key")
        if (cached := self.cache.get(key)) is not None:
            return cached

        for attempt in range(self.max_retries):
            resp = await self._http.get(endpoint, params=params)
            if resp.status_code == 404:
                # openFDA returns 404 with {"error": {"code": "NOT_FOUND"}} for "no matches".
                raise NotFound(f"no openFDA results for {params.get('search', '')}")
            if resp.status_code in (429, 500, 502, 503) and attempt < self.max_retries - 1:
                await asyncio.sleep(0.5 * 2**attempt)
                continue
            if resp.status_code >= 400:
                raise OpenFDAError(f"openFDA {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            self.cache.set(key, data)
            return data
        raise OpenFDAError("openFDA retries exhausted")

    # ------------------------------------------------------------------ endpoints

    async def label(self, drug: str) -> dict:
        data = await self.get("/drug/label.json", search=drug_query("openfda", drug), limit=1)
        return data["results"][0]

    async def adverse_event_counts(self, drug: str, limit: int = 10, serious_only: bool = False) -> list[dict]:
        search = drug_query("patient.drug.openfda", drug)
        if serious_only:
            search += " AND serious:1"
        data = await self.get(
            "/drug/event.json", search=search, count="patient.reaction.reactionmeddrapt.exact", limit=limit
        )
        return data["results"]

    async def adverse_event_total(self, drug: str) -> int:
        data = await self.get("/drug/event.json", search=drug_query("patient.drug.openfda", drug), limit=1)
        return int(data.get("meta", {}).get("results", {}).get("total", 0))

    async def recalls(self, drug: str, limit: int = 5) -> list[dict]:
        data = await self.get(
            "/drug/enforcement.json", search=drug_query("openfda", drug), limit=limit, sort="report_date:desc"
        )
        return data["results"]
