"""Agent end-to-end: LangGraph graph + real MCP server over stdio (offline data) + scripted LLM."""

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

from drug_safety_mcp.agent import build_graph, load_mcp_tools


class ScriptedLLM(BaseChatModel):
    """Plays back a fixed list of AI messages; records what it was shown."""

    script: list[AIMessage]
    seen: list[list[Any]] = Field(default_factory=list)
    bound_tools: list[str] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        self.bound_tools = [t.name for t in tools]
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.seen.append(list(messages))
        # Copy: add_messages assigns ids in place, and a reused id would overwrite the earlier turn.
        msg = self.script[min(len(self.seen) - 1, len(self.script) - 1)].model_copy(deep=True)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def call(name, args, id_):
    return {"name": name, "args": args, "id": id_, "type": "tool_call"}


async def test_agent_uses_mcp_tools_and_finalizes():
    tools = await load_mcp_tools(offline=True)
    llm = ScriptedLLM(
        script=[
            AIMessage(
                content="",
                tool_calls=[
                    call("check_interaction", {"drug_a": "warfarin", "drug_b": "ibuprofen"}, "1"),
                    call("get_drug_label", {"drug": "warfarin", "sections": ["boxed_warning"]}, "2"),
                ],
            ),
            AIMessage(content="Yes - both labels warn about additive bleeding risk [label: warfarin]."),
        ]
    )
    graph = build_graph(llm, tools)
    state = await graph.ainvoke({"messages": [("user", "Can I take ibuprofen with warfarin?")], "tool_rounds": 0})

    assert set(llm.bound_tools) >= {"check_interaction", "get_drug_label"}
    tool_msgs = [m for m in state["messages"] if isinstance(m, ToolMessage)]
    assert {m.name for m in tool_msgs} == {"check_interaction", "get_drug_label"}
    assert any("bleeding" in str(m.content).lower() for m in tool_msgs)  # real data flowed back from MCP
    final = state["messages"][-1].content
    assert "bleeding" in final and "Tools used: check_interaction, get_drug_label" in final
    assert "Not medical advice" in final


async def test_tool_budget_stops_runaway_loops():
    tools = await load_mcp_tools(offline=True)
    loop = AIMessage(content="", tool_calls=[call("recent_recalls", {"drug": "metformin"}, "x")])
    graph = build_graph(ScriptedLLM(script=[loop]), tools, max_tool_rounds=2)
    state = await graph.ainvoke({"messages": [("user", "recalls?")], "tool_rounds": 0})
    assert state["tool_rounds"] == 2
    assert "tool-call budget" in state["messages"][-1].content
