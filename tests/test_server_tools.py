"""Exercise the tools through a real MCP client session (in-memory transport)."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from drug_safety_mcp.server import mcp


def payload(result):
    assert not result.isError, result.content
    return result.structuredContent or json.loads(result.content[0].text)


async def test_lists_tools_resources_prompts(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        tools = {t.name: t for t in (await session.list_tools()).tools}
        assert set(tools) == {"get_drug_label", "top_adverse_events", "recent_recalls", "check_interaction"}
        assert tools["get_drug_label"].outputSchema is not None  # structured output
        assert "drug-safety://about" in [str(r.uri) for r in (await session.list_resources()).resources]
        prompt = await session.get_prompt("safety_brief", {"drug": "warfarin"})
        assert "warfarin" in prompt.messages[0].content.text


async def test_get_drug_label_by_brand_name(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        out = payload(await session.call_tool("get_drug_label", {"drug": "Advil", "sections": ["boxed_warning"]}))
    assert out["found"] and out["generic_name"] == "ibuprofen" and out["has_boxed_warning"]
    assert list(out["sections"]) == ["boxed_warning"]


async def test_unknown_drug_is_not_an_error(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        out = payload(await session.call_tool("get_drug_label", {"drug": "notarealdrug"}))
    assert out == {**out, "found": False}


async def test_bad_section_is_a_tool_error(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        res = await session.call_tool("get_drug_label", {"drug": "warfarin", "sections": ["recipes"]})
    assert res.isError


async def test_top_adverse_events(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        out = payload(await session.call_tool("top_adverse_events", {"drug": "metformin", "limit": 3}))
    assert out["total_reports"] > 0 and len(out["top_reactions"]) == 3
    assert out["top_reactions"][0]["reaction"] == "diarrhoea"
    assert "not incidence" in out["caveat"]


async def test_recalls_found_and_empty(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        met = payload(await session.call_tool("recent_recalls", {"drug": "metformin"}))
        war = payload(await session.call_tool("recent_recalls", {"drug": "warfarin"}))
    assert met["found"] and met["recalls"][0]["classification"] == "Class II"
    assert war["found"] is False


@pytest.mark.parametrize("a,b", [("warfarin", "ibuprofen"), ("advil", "warfarin")])
async def test_interaction_found_in_both_labels(offline_client, a, b):
    async with create_connected_server_and_client_session(mcp) as session:
        out = payload(await session.call_tool("check_interaction", {"drug_a": a, "drug_b": b}))
    assert out["mentioned_in_labels"]
    owners = {e["label_of"] for e in out["evidence"]}
    assert len(owners) == 2  # documented in both labels
    assert all(e["section"] for e in out["evidence"])


async def test_no_interaction_mentioned(offline_client):
    async with create_connected_server_and_client_session(mcp) as session:
        out = payload(await session.call_tool("check_interaction", {"drug_a": "metformin", "drug_b": "ibuprofen"}))
    assert out["mentioned_in_labels"] is False
    assert "does not mean" in out["note"]
