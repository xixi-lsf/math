from __future__ import annotations

from agent.state import AgentState
from agent.logger import logger, log_step
from knowledge.vectordb import get_store
from models.problem import ReasoningStep


def fallback_node(state: AgentState) -> dict:
    """
    从题库取一道保底题，直接跳过绘图走 finalize。
    """
    topic = state["topic"]
    difficulty = state["difficulty"]
    step_id = state.get("step_counter", 0)

    logger.info(
        "[fallback] requesting fallback problem topic=%s difficulty=%s",
        topic,
        difficulty,
    )
    fallback = get_store().get_fallback_problem(topic, difficulty)

    if fallback:
        logger.info(
            "[fallback] fallback problem found topic=%s difficulty=%s",
            topic,
            difficulty,
        )
        step = ReasoningStep(
            step_id=step_id,
            node_name="fallback",
            action=f"题干/配图生成失败，从题库中随机选取一道{topic}题目（难度{difficulty}）作为兜底",
            tool_called="knowledge_store.get_fallback_problem",
            tool_output_summary=f"source={fallback.get('source', '题库')}, has_image={bool(fallback.get('image_base64'))}",
        )
        q = state.get("step_queue")
        if q is not None:
            q.put_nowait(step)
        log_step(step)
        return {
            "latex_problem": fallback["problem"],
            "solution": fallback["solution"],
            "solution_latex": fallback["solution"],
            "is_fallback": True,
            "params": None,
            "image_base64": fallback.get("image_base64", ""),
            "drawing_path": None,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
        }

    logger.info(
        "[fallback] no fallback problem found topic=%s difficulty=%s",
        topic,
        difficulty,
    )
    step = ReasoningStep(
        step_id=step_id,
        node_name="fallback",
        action=f"题库中未找到匹配的{topic}题目（难度{difficulty}），返回错误提示",
        tool_called="knowledge_store.get_fallback_problem",
        tool_output_summary="no result found",
    )
    q = state.get("step_queue")
    if q is not None:
        q.put_nowait(step)
    log_step(step)
    return {
        "latex_problem": f"抱歉，暂时无法生成符合要求的{topic}题目，请稍后重试。",
        "solution": "",
        "is_fallback": True,
        "params": None,
        "image_base64": "",
        "drawing_path": None,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
