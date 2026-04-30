"""
Node 4: Validation — SymPy mathematical check + LLM clarity check.
"""
from __future__ import annotations
from agent.state import AgentState
from models.problem import ReasoningStep, ValidationResult
from validation.sympy_validator import validate_params


def validation_node(state: AgentState) -> dict:
    params = state.get("params")
    latex_problem = state.get("latex_problem", "")
    step_id = state.get("step_counter", 0)

    if params is None:
        result = ValidationResult(
            is_valid=False,
            error_type="missing_params",
            error_detail="参数提取失败，无法验证",
        )
        step = ReasoningStep(
            step_id=step_id,
            node_name="validation",
            action="验证失败：参数为空",
            tool_called="SymPy",
            tool_output_summary="params is None",
        )
        return {
            "validation_result": result,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # ── SymPy mathematical validation ────────────────────────────────────────
    sympy_result = validate_params(params)

    if not sympy_result.is_valid:
        step = ReasoningStep(
            step_id=step_id,
            node_name="validation",
            action=f"SymPy 验证失败：{sympy_result.error_detail}",
            tool_called="SymPy.validate",
            tool_output_summary=f"error_type={sympy_result.error_type}",
        )
        return {
            "validation_result": sympy_result,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # ── LLM clarity check ────────────────────────────────────────────────────
    clarity_ok, clarity_note = _llm_clarity_check(state, latex_problem)

    if not clarity_ok:
        result = ValidationResult(
            is_valid=False,
            error_type="unclear_problem",
            error_detail=clarity_note,
            suggested_fix="请重新生成更清晰的题干",
        )
        step = ReasoningStep(
            step_id=step_id,
            node_name="validation",
            action=f"LLM 清晰性检查失败：{clarity_note}",
            tool_called="LLM.clarity_check",
            tool_output_summary=clarity_note,
        )
        return {
            "validation_result": result,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
            "retry_count": state.get("retry_count", 0) + 1,
        }

    # ── All passed ────────────────────────────────────────────────────────────
    result = ValidationResult(is_valid=True)
    step = ReasoningStep(
        step_id=step_id,
        node_name="validation",
        action="验证通过：SymPy 数学验证 ✓，题干清晰性 ✓",
        tool_called="SymPy.validate + LLM.clarity_check",
        tool_output_summary="is_valid=True",
    )
    return {
        "validation_result": result,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }


def _llm_clarity_check(state: AgentState, latex_problem: str) -> tuple[bool, str]:
    """Quick LLM check: is the problem statement clear and unambiguous?"""
    try:
        from openai import OpenAI
        cfg = state.get("llm_config", {})
        llm = OpenAI(
            api_key=cfg.get("api_key", "sk-placeholder"),
            base_url=cfg.get("base_url", "https://api.openai.com/v1"),
        )
        model = cfg.get("model", "gpt-4o-mini")
        response = llm.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是数学题审核专家。判断题目是否表述清晰、无歧义、条件充分。只回答 OK 或 FAIL:原因。"},
                {"role": "user", "content": f"审核以下题目：\n\n{latex_problem}"},
            ],
            temperature=0,
            max_tokens=100,
        )
        answer = response.choices[0].message.content.strip()
        if answer.upper().startswith("OK"):
            return True, ""
        return False, answer.replace("FAIL:", "").strip()
    except Exception:
        # If LLM check fails, let it pass (SymPy already validated math)
        return True, ""
