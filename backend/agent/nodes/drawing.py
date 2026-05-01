"""
Node 5: Adaptive drawing dispatch.
Fast path: local template functions.
Slow path: LLM generates Matplotlib code → sandbox execution.
为生成的数学题目自动配图（例如椭圆、双曲线、抛物线及其关键点、直线等），
将图片以 Base64 编码的 PNG 存入状态
自适应双路径策略
"""
from __future__ import annotations
from agent.state import AgentState
from models.problem import ReasoningStep
from drawing.local_tools import try_local_draw
from drawing.sandbox import execute_drawing_code

_MAX_DRAWING_RETRIES = 3


def drawing_node(state: AgentState) -> dict:
    #获取参数，当前已重试次数，上次错误记录
    params = state.get("params")
    step_id = state.get("step_counter", 0)
    drawing_retry = state.get("drawing_retry_count", 0)
    prev_error = state.get("drawing_error")

    # ── Fast path: try local template ────────────────────────────────────────
    #仅在第一次进入绘图节点时（drawing_retry == 0）尝试快速路径
    if drawing_retry == 0:
        try:
            image_b64 = try_local_draw(params)
            step = ReasoningStep(
                step_id=step_id,
                node_name="drawing",
                action="本地绘图工具成功生成配图（快速路径）",
                tool_called="local_tools.try_local_draw",
                tool_output_summary="base64 PNG 生成成功",
                drawing_path="fast",
            )
            return {
                "image_base64": image_b64,
                "drawing_path": "fast",
                "reasoning_trace": state.get("reasoning_trace", []) + [step],
                "step_counter": step_id + 1,
            }
        except NotImplementedError:
            pass  # Fall through to slow path
        except Exception as e:
            pass  # Local draw failed for other reason, try slow path

    # ── Slow path: LLM generates drawing code ────────────────────────────────
    if drawing_retry >= _MAX_DRAWING_RETRIES:
        step = ReasoningStep(
            step_id=step_id,
            node_name="drawing",
            action=f"绘图失败：已重试 {drawing_retry} 次，使用占位图",
            tool_called="sandbox",
            tool_output_summary="max retries exceeded",
            drawing_path="slow",
        )
        return {
            "image_base64": _placeholder_image(params),
            "drawing_path": "slow",
            "drawing_error": None,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
        }

    # Generate drawing code via LLM
    #利用 LLM 生成 Python/Matplotlib 代码
    code = _generate_drawing_code(state, prev_error)
    #沙箱执行代码
    image_b64, error = execute_drawing_code(code, params)

    if error:
        step = ReasoningStep(
            step_id=step_id,
            node_name="drawing",
            action=f"绘图代码执行失败（第{drawing_retry+1}次），准备重试",
            tool_called="sandbox.execute",
            tool_output_summary=f"error: {error[:100]}",
            drawing_path="slow",
        )
        return {
            "drawing_code": code,
            "drawing_error": error,
            "drawing_retry_count": drawing_retry + 1,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
        }

    step = ReasoningStep(
        step_id=step_id,
        node_name="drawing",
        action=f"LLM 生成绘图代码并执行成功（慢速路径，第{drawing_retry+1}次）",
        tool_called="LLM.code_gen + sandbox.execute",
        tool_output_summary="base64 PNG 生成成功",
        drawing_path="slow",
    )
    return {
        "image_base64": image_b64,
        "drawing_path": "slow",
        "drawing_code": code,
        "drawing_error": None,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }

#生成绘图代码函数
def _generate_drawing_code(state: AgentState, prev_error: str | None) -> str:
    from openai import OpenAI
    cfg = state.get("llm_config", {})
    llm = OpenAI(
        api_key=cfg.get("api_key", "sk-placeholder"),
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
    )
    model = cfg.get("model", "gpt-4o-mini")
    params = state.get("params")
    latex_problem = state.get("latex_problem", "")

    error_hint = ""
    if prev_error:
        error_hint = f"\n\n上次代码执行报错：\n{prev_error}\n请修正上述错误。"

    system_prompt = """你是 Matplotlib 绘图专家。根据解析几何题目生成精确的配图代码。

重要规则：
1. 代码中【禁止出现任何数字字面量】，只能使用已提供的变量（a, b, c_focal, p, F1_x, F1_y 等）
2. 必须绘制：曲线本体、所有关键点（含标注）、所有直线
3. 使用 matplotlib，设置 aspect='equal'，添加坐标轴
4. 不要调用 plt.show()，不要保存文件（由外部框架处理输出）
5. 只输出 Python 代码，不要任何解释"""

    user_prompt = f"""题目：{latex_problem}

可用变量（已由外部注入，直接使用）：
- a, b, c_focal（椭圆/双曲线参数）
- p（抛物线参数）
- e（离心率）
- 各关键点：F1_x, F1_y, F2_x, F2_y, A_x, A_y, B_x, B_y 等（根据题目中的点名）
- line0_slope, line0_intercept（直线参数）
- plot_x_min, plot_x_max, plot_y_min, plot_y_max（视口范围）

请生成绘图代码：{error_hint}"""

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=1000,
    )
    code = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return code


def _placeholder_image(params) -> str:
    """Generate a minimal placeholder figure when all drawing attempts fail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import io, base64
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.text(0.5, 0.5, "配图生成失败\n请检查参数", ha="center", va="center",
            transform=ax.transAxes, fontsize=14, color="red")
    ax.set_axis_off()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()
