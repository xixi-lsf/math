"""
Node 6: Solution generation (on-demand).
在题目生成、参数提取（以及可能的绘图之后），
调用 LLM 生成详细的解题过程（LaTeX 格式），
并将结果存入状态中的 solution_latex 字段。
"""
from __future__ import annotations
from agent.state import AgentState
from models.problem import ReasoningStep


def solution_generation_node(state: AgentState) -> dict:
    #读取参数，文本问题，计数器
    params = state.get("params")
    latex_problem = state.get("latex_problem", "")
    step_id = state.get("step_counter", 0)

    from openai import OpenAI
    cfg = state.get("llm_config", {})
    llm = OpenAI(
        api_key=cfg.get("api_key", "sk-placeholder"),
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
    )
    model = cfg.get("model", "gpt-4o-mini")

    # 构建参数上下文（供 LLM 参考），仅提取圆锥曲线的基础参数
    param_ctx = ""
    if params:
        c = params.conic
        parts = [f"曲线类型：{c.curve_type}"]
        if c.a:
            parts.append(f"a={c.a}")
        if c.b:
            parts.append(f"b={c.b}")
        if c.c:
            parts.append(f"c={c.c}")
        if c.p:
            parts.append(f"p={c.p}")
        param_ctx = "，".join(parts)

    system_prompt = """你是解析几何解题专家。请给出完整、清晰的解题过程。
要求：
1. 分步骤解题，每步有简短说明
2. 所有数学表达式用 LaTeX 格式（行内用 $...$ ，独立公式用 $$...$$）
3. 最终答案用 \\boxed{} 标注
4. 解题过程要严谨，不跳步"""

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"题目：{latex_problem}\n\n已知参数：{param_ctx}\n\n请给出完整解题过程。"},
        ],
        temperature=0.1,
        max_tokens=1500,
    )
    solution = response.choices[0].message.content.strip()

    step = ReasoningStep(
        step_id=step_id,
        node_name="solution_generation",
        action="生成解题步骤",
        tool_called=f"LLM.chat ({model})",
        tool_output_summary=f"解题过程 {len(solution)} 字符",
    )

    return {
        "solution_latex": solution,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
