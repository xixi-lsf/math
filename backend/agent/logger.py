"""
Structured console logging for agent reasoning steps.
Prints each step immediately as it completes, including full code blocks.
"""
from __future__ import annotations
import logging
import sys
import textwrap

# Configure a dedicated logger that writes to stdout with a clean format
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(message)s"))

logger = logging.getLogger("agent.trace")
logger.setLevel(logging.DEBUG)
logger.addHandler(_handler)
logger.propagate = False

_NODE_LABELS = {
    "knowledge_retrieval": "知识检索",
    "problem_generation":  "题目生成",
    "param_extraction":    "参数提取",
    "solve_and_validate":  "求解验证",
    "drawing":             "绘图",
    "finalize":            "最终化",
}

_SEP = "─" * 60


def log_step(step, *, code: str | None = None, full_output: str | None = None) -> None:
    """
    Print a reasoning step to stdout.

    Parameters
    ----------
    step        : ReasoningStep instance
    code        : optional generated code to print verbatim (drawing node)
    full_output : optional long text output (e.g. full LLM response)
    """
    label = _NODE_LABELS.get(step.node_name, step.node_name)
    lines = [
        f"\n{_SEP}",
        f"[Step {step.step_id}] {label}",
        f"  动作   : {step.action}",
    ]
    if step.tool_called:
        lines.append(f"  工具   : {step.tool_called}")
    if step.tool_input_summary:
        lines.append(f"  输入   : {step.tool_input_summary}")
    if step.tool_output_summary:
        lines.append(f"  输出   : {step.tool_output_summary}")
    if step.drawing_path:
        lines.append(f"  路径   : {step.drawing_path}")

    if full_output:
        lines.append("  完整输出:")
        for ln in full_output.splitlines():
            lines.append(f"    {ln}")

    if code:
        lines.append("  生成代码:")
        lines.append("  ```python")
        for ln in code.splitlines():
            lines.append(f"  {ln}")
        lines.append("  ```")

    lines.append(_SEP)
    logger.info("\n".join(lines))
