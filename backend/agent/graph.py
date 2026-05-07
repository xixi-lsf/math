"""
LangGraph graph definition for the problem generation agent.
定义整个 Agent 的工作流图：节点、边、条件分支、重试逻辑、最终输出。
描述“数据怎么流动、每一步做什么”
每个 _node 函数接收 AgentState，返回部分更新的 dict，LangGraph 会自动合并到状态中
"""
from __future__ import annotations
from typing import Literal
from langgraph.graph import StateGraph, END

from agent.state import AgentState
from agent.nodes.knowledge_retrieval import knowledge_retrieval_node
from agent.nodes.problem_generation import problem_generation_node, param_extraction_node
from agent.nodes.solve_and_validate import solve_and_validate_node
from agent.nodes.fallback import fallback_node
from agent.nodes.drawing import drawing_node
from models.problem import Problem

#题目生成/绘图 重试次数
_MAX_RETRIES = 3
_MAX_DRAWING_RETRIES = 3


# ── Conditional edges（条件边函数）：决策逻辑 ─────────────────────────────────────────────────────────
#决定求解验证失败后是重新生成题目，进入保底，还是进入绘图阶段
def should_retry_generation(state: AgentState) -> Literal["problem_generation", "fallback", "drawing"]:
    result = state.get("validation_result")
    retry = state.get("generation_retry", 0)
    if result and not result.is_valid:
        if retry < _MAX_RETRIES:
            return "problem_generation"
        return "fallback"
    return "drawing"

#决定绘图失败后是重试绘图，还是结束
def should_retry_drawing(state: AgentState) -> Literal["retry_draw", "done"]:
    error = state.get("drawing_error")
    retry = state.get("drawing_retry_count", 0)
    if error and retry < _MAX_DRAWING_RETRIES:
        return "retry_draw"
    return "done"


# ── Finalize node：最终节点 ─────────────────────────────────────────────────────────────
#它将状态中的各个字段组装成一个完整的 Problem 对象，存入 final_problem 字段
def finalize_node(state: AgentState) -> dict:
    params = state.get("params")
    problem = Problem(
        problem_id=params.problem_id if params else "unknown",
        params=params,
        latex_problem=state.get("latex_problem", ""),
        image_base64=state.get("image_base64") or "",
        reasoning_trace=state.get("reasoning_trace", []),
        solution=state.get("solution"),
        solution_latex=state.get("solution_latex"),
        generation_config=state.get("llm_config", {}),
        is_fallback=bool(state.get("is_fallback", False)),
    )
    return {"final_problem": problem}


# ── Build graph ：图构建函数───────────────────────────────────────────────────────────────
#添加节点（顺序无关，只是注册）
def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    #参数：节点标识名，可调用对象（实际函数）
    g.add_node("knowledge_retrieval", knowledge_retrieval_node)
    g.add_node("problem_generation", problem_generation_node)
    g.add_node("param_extraction", param_extraction_node)
    g.add_node("solve_and_validate", solve_and_validate_node)
    g.add_node("fallback", fallback_node)
    g.add_node("drawing", drawing_node)
    g.add_node("finalize", finalize_node)

    #入口
    g.set_entry_point("knowledge_retrieval")
    #加线性执行边（知识检索，问题生成，参数提取）
    g.add_edge("knowledge_retrieval", "problem_generation")
    g.add_edge("problem_generation", "param_extraction")
    g.add_edge("param_extraction", "solve_and_validate")

    g.add_conditional_edges(
        "solve_and_validate",
        should_retry_generation,
        {
            "problem_generation": "problem_generation",
            "drawing": "drawing",
            "fallback": "fallback",
        },
    )

    g.add_edge("fallback", "finalize")

    g.add_conditional_edges(
        "drawing",
        should_retry_drawing,
        {"retry_draw": "drawing", "done": "finalize"},
    )

    #结束边
    g.add_edge("finalize", END)
    return g.compile()


# ── Singleton compiled graph ──────────────────────────────────────────────────
_graph = None

#单例模式：确保整个应用中只有一个编译好的图实例，节省内存和初始化时间
def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
