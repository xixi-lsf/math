from __future__ import annotations

from agent.state import AgentState
from agent.logger import logger
from knowledge.vectordb import get_store


def fallback_node(state: AgentState) -> dict:
    """
    从题库取一道保底题，直接跳过绘图走 finalize。
    """
    topic = state["topic"]
    difficulty = state["difficulty"]

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
        return {
            "latex_problem": fallback["problem"],
            "solution": fallback["solution"],
            "solution_latex": fallback["solution"],
            "is_fallback": True,
            "params": None,
            "image_base64": "",
            "drawing_path": None,
        }

    logger.info(
        "[fallback] no fallback problem found topic=%s difficulty=%s",
        topic,
        difficulty,
    )
    return {
        "latex_problem": f"抱歉，暂时无法生成符合要求的{topic}题目，请稍后重试。",
        "solution": "",
        "is_fallback": True,
        "params": None,
        "image_base64": "",
        "drawing_path": None,
    }
