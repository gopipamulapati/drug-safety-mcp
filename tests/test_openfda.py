import httpx
import pytest

from drug_safety_mcp.openfda import NotFound, OpenFDAClient, OpenFDAError, TTLCache, clean_name, drug_query


def test_clean_name_strips_query_injection():
    assert clean_name('  Metformin" OR openfda.x:"*') == "metformin or openfdax"
    with pytest.raises(ValueError):
        clean_name('""::')


def test_drug_query_matches_generic_or_brand():
    q = drug_query("openfda", "Advil")
    assert q == '(openfda.generic_name:"advil" openfda.brand_name:"advil")'


def test_ttl_cache_expiry():
    c = TTLCache(ttl=0)
    c.set("k", 1)
    assert c.get("k") is None
    c2 = TTLCache(ttl=60, max_items=1)
    c2.set("a", 1)
    c2.set("b", 2)
    assert c2.get("a") is None and c2.get("b") == 2


async def test_retries_on_429_then_caches():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={"results": [{"ok": True}]})

    client = OpenFDAClient(api_key="", transport=httpx.MockTransport(handler))
    assert (await client.get("/drug/label.json", search="x"))["results"][0]["ok"]
    await client.get("/drug/label.json", search="x")  # served from cache
    assert calls["n"] == 2


async def test_404_is_not_found_and_500_is_error():
    nf = OpenFDAClient(api_key="", transport=httpx.MockTransport(lambda r: httpx.Response(404)))
    with pytest.raises(NotFound):
        await nf.label("nothing")
    err = OpenFDAClient(api_key="", max_retries=1, transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    with pytest.raises(OpenFDAError):
        await err.label("x")


async def test_api_key_sent_but_not_part_of_cache_key():
    seen = []

    def handler(req):
        seen.append(req.url.params.get("api_key"))
        return httpx.Response(200, json={"results": [{}]})

    client = OpenFDAClient(api_key="secret", transport=httpx.MockTransport(handler))
    await client.get("/drug/label.json", search="x")
    assert seen == ["secret"]
    assert all("secret" not in k for k in client.cache._data)
