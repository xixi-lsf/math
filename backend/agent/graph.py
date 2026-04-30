"""
LangGraph graph definition for the problem generation agent.
"""
from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.knowledge_retrieval import knowledge_retrieval_node
from agent.nodes.problem_generation import problem_generation_node, param_extraction_node
from agent.nodes.validation import validation_node
from agent.nodes.drawing import drawing_node
from agent.nodes.solution_generation import solution_generation_node
from models.problem import Problem

_MAX_RETRIES = 3
_MAX_DRAWING_RETRIES = 3


# ── Conditional edges ─────────────────────────────────────────────────────────

def should_retry_generation(state: AgentState) -> Literal["retry", "draw"]:
    result = state.get("validation_result")
    retry = state.get("retry_count", 0)
    if result and not result.is_valid and retry < _MAX_RETRIES:
        return "retry"
    return "draw"


def should_retry_drawing(state: AgentState) -> Literal["retry_draw", "done"]:
    error = state.get("drawing_error")
    retry = state.get("drawing_retry_count", 0)
    if error and retry < _MAX_DRAWING_RETRIES:
        return "retry_draw"
    return "done"


# ── Finalize node ─────────────────────────────────────────────────────────────

def finalize_node(state: AgentState) -> dict:
    params = state.get("params")
    problem = Problem(
        problem_id=params.problem_id if params else "unknown",
        params=params,
        latex_problem=state.get("latex_problem", ""),
        image_base64=state.get("image_base64") or "",
        reasoning_trace=state.get("reasoning_trace", []),
        solution_latex=state.get("solution_latex"),
        generation_config=state.get("llm_config", {}),
    )
    return {"final_problem": problem}


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("knowledge_retrieval", knowledge_retrieval_node)
    g.add_node("problem_generation", problem_generation_node)
    g.add_node("param_extraction", param_extraction_node)
    g.add_node("validation", validation_node)
    g.add_node("drawing", drawing_node)
    g.add_node("finalize", finalize_node)

    g.set_entry_point("knowledge_retrieval")
    g.add_edge("knowledge_retrieval", "problem_generation")
    g.add_edge("problem_generation", "param_extraction")
    g.add_edge("param_extraction", "validation")

    g.add_conditional_edges(
        "validation",
        should_retry_generation,
        {"retry": "problem_generation", "draw": "drawing"},
    )

    g.add_conditional_edges(
        "drawing",
        should_retry_drawing,
        {"retry_draw": "drawing", "done": "finalize"},
    )

    g.add_edge("finalize", END)
    return g.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
