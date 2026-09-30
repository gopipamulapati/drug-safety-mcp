"""LangGraph agent that discovers its tools from the drug-safety MCP server at runtime.

agent --(tool calls)--> tools --> agent ... --> finalize (adds citations/disclaimer)

drug-safety-agent "Is it safe to take ibuprofen with warfarin?"
drug-safety-agent --offline "Summarise metformin safety"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .server import DISCLAIMER

SYSTEM = (
    "You are a drug-safety research assistant for clinicians and pharmacovigilance analysts. "
    "Use the tools to ground every factual statement; never answer from memory. "
    "For interaction questions call check_interaction AND get_drug_label for the relevant sections. "
    "Cite sources inline like [label: warfarin/drug_interactions] or [FAERS: metformin]. "
    "State clearly when data was not found. Be concise."
)


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    tool_rounds: int


def server_connection(offline: bool = False) -> dict[str, Any]:
    env = {**os.environ, "DRUG_SAFETY_OFFLINE": "1" if offline else os.getenv("DRUG_SAFETY_OFFLINE", "0")}
    return {
        "drug-safety": {
            "transport": "stdio",
            "command": sys.executable,
            "args": ["-m", "drug_safety_mcp.server"],
            "env": env,
        }
    }


async def load_mcp_tools(offline: bool = False) -> list[BaseTool]:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    return await MultiServerMCPClient(server_connection(offline)).get_tools()


def build_graph(llm: BaseChatModel, tools: list[BaseTool], max_tool_rounds: int = 6):
    model = llm.bind_tools(tools)

    async def agent(state: AgentState) -> dict:
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(SYSTEM), *msgs]
        return {"messages": [await model.ainvoke(msgs)]}

    async def count_round(state: AgentState) -> dict:
        return {"tool_rounds": state.get("tool_rounds", 0) + 1}

    def finalize(state: AgentState) -> dict:
        last = state["messages"][-1]
        used = sorted({m.name for m in state["messages"] if isinstance(m, ToolMessage) and m.name})
        text = last.content if isinstance(last, AIMessage) and not last.tool_calls else ""
        if not text:
            text = "I could not complete the research within the tool-call budget."
        footer = f"\n\n_Tools used: {', '.join(used) or 'none'}. {DISCLAIMER}_"
        return {"messages": [AIMessage(content=str(text) + footer)]}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and state.get("tool_rounds", 0) < max_tool_rounds:
            return "tools"
        return "finalize"

    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    g.add_node("count_round", count_round)
    g.add_node("finalize", finalize)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", "finalize": "finalize"})
    g.add_edge("tools", "count_round")
    g.add_edge("count_round", "agent")
    g.add_edge("finalize", END)
    return g.compile()


def default_llm() -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.getenv("AGENT_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
    )


async def ask(question: str, llm: BaseChatModel | None = None, offline: bool = False) -> dict:
    tools = await load_mcp_tools(offline)
    graph = build_graph(llm or default_llm(), tools)
    return await graph.ainvoke({"messages": [HumanMessage(question)], "tool_rounds": 0})


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Drug-safety research agent (LangGraph + MCP)")
    p.add_argument("question")
    p.add_argument("--offline", action="store_true", help="use bundled sample data instead of live openFDA")
    p.add_argument("--trace", action="store_true", help="print every tool call")
    args = p.parse_args(argv)
    state = asyncio.run(ask(args.question, offline=args.offline))
    if args.trace:
        for m in state["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                for tc in m.tool_calls:
                    print(f"-> {tc['name']}({tc['args']})")
    print(state["messages"][-1].content)


if __name__ == "__main__":
    main()
