"""
Node 4: Solve and validate — use SymPy for fast checks, then LLM to solve.
"""
from __future__ import annotations
from openai import OpenAI

from agent.state import AgentState
from agent.logger import log_step
from models.problem import ReasoningStep, ValidationResult
from validation.sympy_validator import validate_params


def _make_client(llm_config: dict) -> OpenAI:
    return OpenAI(
        api_key=llm_config.get("api_key", "sk-placeholder"),
        base_url=llm_config.get("base_url", "https://api.openai.com/v1"),
    )


def solve_and_validate_node(state: AgentState) -> dict:
    """
    让 LLM 真正求解题目。
    - 能解出来 → is_valid=True，把解题过程存入 solution
    - 出现矛盾/无解 → is_valid=False，触发重新生成
    """
    params = state.get("params")
    latex_problem = state.get("latex_problem", "")
    llm_config = state["llm_config"]
    step_id = state.get("step_counter", 0)

    if params is not None:
        sympy_result = validate_params(params)
        if not sympy_result.is_valid:
            step = ReasoningStep(
                step_id=step_id,
                node_name="solve_and_validate",
                action=f"SymPy 结构检查失败：{sympy_result.error_detail}",
                tool_called="SymPy.validate",
                tool_output_summary=f"error_type={sympy_result.error_type}",
            )
            q = state.get("step_queue")
            if q is not None:
                q.put_nowait(step)
            log_step(step)
            return {
                "validation_result": sympy_result,
                "solution": None,
                "reasoning_trace": state.get("reasoning_trace", []) + [step],
                "step_counter": step_id + 1,
            }

    system_prompt = """你是严格的高中数学解题专家。
请尝试完整求解以下题目。如果题目较复杂，只需给出关键步骤和结论，不需要展示每一行代数化简过程，控制总长度在2000字以内
如果题目较复杂，优先给出关键步骤和最终各点坐标数值，
    省略繁琐的中间代数化简，控制总长度在3000字以内。
    特别重要：所有在题目中出现的点，如果可以求出坐标，
    必须在解题过程末尾单独列出：
    【关键点坐标汇总】
    A = (x, y)
    B = (x, y)
    M = (x, y)
    ...
    这样绘图时可以直接读取。

要求：
1. 按步骤写出完整解题过程，使用 LaTeX 格式
2. 如果在求解过程中发现条件矛盾、题目无解、
   或者某个几何对象不存在（如内部点无法作切线），
   必须在最后一行单独输出：INVALID: [具体原因]
3. 如果成功求出答案，最后一行单独输出：VALID
4. 不要编造答案，宁可判 INVALID 也不要强行给出错误解"""

    user_prompt = f"请求解以下题目：\n\n{latex_problem}"

    client = _make_client(llm_config)
    response = client.chat.completions.create(
        model=llm_config.get("model", "deepseek-chat"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
        max_tokens=5000,
    )

    solution_text = (response.choices[0].message.content or "").strip()
    non_empty_lines = [l.strip() for l in solution_text.split("\n") if l.strip()]
    last_line = non_empty_lines[-1] if non_empty_lines else ""
    # INVALID may appear before a trailing summary block — scan all lines
    invalid_line = next((l for l in non_empty_lines if l.startswith("INVALID")), None)
    if invalid_line:
        last_line = invalid_line

    q = state.get("step_queue")
    model = llm_config.get("model", "deepseek-chat")

    if last_line.startswith("INVALID"):
        reason = last_line.replace("INVALID:", "", 1).strip() or "题目无解或条件矛盾"
        step = ReasoningStep(
            step_id=step_id,
            node_name="solve_and_validate",
            action=f"LLM 求解判定题目无效：{reason}",
            tool_called=f"LLM.solve ({model})",
            tool_output_summary=reason,
        )
        if q is not None:
            q.put_nowait(step)
        log_step(step, full_output=solution_text)
        return {
            "validation_result": ValidationResult(
                is_valid=False,
                error_type="unsolvable",
                error_detail=reason,
                suggested_fix=f"题目无解或条件矛盾：{reason}，请重新生成",
            ),
            "solution": None,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
        }

    lines = solution_text.split("\n")
    clean_solution = "\n".join(line for line in lines if line.strip() != "VALID").strip()
    step = ReasoningStep(
        step_id=step_id,
        node_name="solve_and_validate",
        action="LLM 完成求解并通过验证",
        tool_called=f"LLM.solve ({llm_config.get('model', 'deepseek-chat')})",
        tool_output_summary=f"解题过程 {len(clean_solution)} 字符",
    )
    if q is not None:
        q.put_nowait(step)
    log_step(step, full_output=clean_solution)
    return {
        "validation_result": ValidationResult(is_valid=True),
        "solution": clean_solution,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
