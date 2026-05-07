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
    raw_solution = state.get("solution", "")   # 验证节点已解出的答案
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

    system_prompt = """你是解析几何解题专家。请将给定的解题过程整理成清晰、规范的分步解答。
要求：
1. 保留原解题过程的所有步骤和结论，不要改变推导逻辑
2. 分步骤呈现，每步前加简短说明（如"第一步：设定坐标系"）
3. 所有数学表达式用 LaTeX 格式：行内公式用 $...$，独立公式用 $$...$$
4. 最终答案用 \\boxed{} 标注
5. 只输出整理后的解题过程，不要重新推导"""

    if raw_solution:
        user_content = f"题目：{latex_problem}\n\n已知参数：{param_ctx}\n\n原始解题过程（请整理格式，不要改变推导）：\n{raw_solution}"
    else:
        user_content = f"题目：{latex_problem}\n\n已知参数：{param_ctx}\n\n请给出完整解题过程。"

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
        max_tokens=3000,
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
