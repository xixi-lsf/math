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
from agent.logger import log_step
from models.problem import ReasoningStep
from drawing.local_tools import try_local_draw
from drawing.sandbox import execute_drawing_code, _build_param_block

_MAX_DRAWING_RETRIES = 3


def drawing_node(state: AgentState) -> dict:
    #获取参数，当前已重试次数，上次错误记录
    params = state.get("params")
    step_id = state.get("step_counter", 0)
    drawing_retry = state.get("drawing_retry_count", 0)
    prev_error = state.get("drawing_error")

    # ── Fast path: try local template ────────────────────────────────────────
    #仅在第一次进入绘图节点时（drawing_retry == 0）尝试快速路径
    def _params_complete(p) -> bool:
        c = p.conic
        if c.curve_type == "ellipse":
            return c.a is not None and c.b is not None
        elif c.curve_type == "hyperbola":
            return c.a is not None and c.b is not None
        elif c.curve_type == "parabola":
            return c.p is not None
        elif c.curve_type == "polar_conic":
            return c.eccentricity is not None
        return False

    if drawing_retry == 0 and _params_complete(params):
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
            q = state.get("step_queue")
            if q is not None:
                q.put_nowait(step)
            log_step(step)
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
        q = state.get("step_queue")
        if q is not None:
            q.put_nowait(step)
        log_step(step)
        return {
            "image_base64": _placeholder_image(params),
            "drawing_path": "slow",
            "drawing_error": None,
            "reasoning_trace": state.get("reasoning_trace", []) + [step],
            "step_counter": step_id + 1,
        }

    # Generate drawing code via LLM
    #利用 LLM 生成 Python/Matplotlib 代码
    log_step(ReasoningStep(
        step_id=step_id,
        node_name="drawing",
        action="进入慢速路径：准备调用 LLM 生成绘图代码",
        tool_called="LLM.code_gen",
        tool_output_summary="drawing_path=slow",
        drawing_path="slow",
    ))
    code = _generate_drawing_code(
        state,
        prev_error,
        solution=state.get("solution"),
        injected_vars=_build_param_block(params)[1],
    )
    #沙箱执行代码
    image_b64, error = execute_drawing_code(code, params)

    if error:
        step = ReasoningStep(
            step_id=step_id,
            node_name="drawing",
            action=f"绘图代码执行失败（第{drawing_retry+1}次），准备重试",
            tool_called="sandbox.execute",
            tool_output_summary=f"error: {error}",
            drawing_path="slow",
        )
        q = state.get("step_queue")
        if q is not None:
            q.put_nowait(step)
        log_step(step, code=code, full_output=f"执行错误:\n{error}")
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
    q = state.get("step_queue")
    if q is not None:
        q.put_nowait(step)
    log_step(step, code=code)
    return {
        "image_base64": image_b64,
        "drawing_path": "slow",
        "drawing_code": code,
        "drawing_error": None,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }

#生成绘图代码函数
def _generate_drawing_code(
    state: AgentState,
    prev_error: str | None,
    solution: str | None = None,
    injected_vars: list[str] | None = None,
) -> str:
    from openai import OpenAI
    cfg = state.get("llm_config", {})
    llm = OpenAI(
        api_key=cfg.get("api_key", "sk-placeholder"),
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
    )
    model = cfg.get("model", "gpt-4o-mini")
    params = state.get("params")
    latex_problem = state.get("latex_problem", "")
    available_vars = ", ".join(injected_vars or []) or "（无可用注入变量）"

    error_hint = ""
    if prev_error:
        error_hint = f"\n\n上次代码执行报错：\n{prev_error}\n请修正上述错误。"

    solution_hint = ""
    if solution:
        solution_hint = f"""
参考解题过程（已经求解完毕，直接从中提取数值，不要重新推导）：
{solution[:1000]}

重要：上面的解题过程已经给出了所有关键点坐标和参数数值，
请直接从中读取这些数值用于绘图，不要在代码里重新计算。
"""

    system_prompt = f"""你是 Matplotlib 绘图专家。根据解析几何题目生成精确的配图代码。

重要规则：
1. 【禁止在代码里推导或计算任何数学结论】，所有数值直接从"参考解题过程"中读取，
   或使用已注入的变量（a, b, c_focal, p, F1_x, F1_y 等）
2. 必须绘制：曲线本体、所有关键点（含标注）、所有直线/线段
3. 使用 matplotlib，设置 aspect='equal'，添加坐标轴，不要显示图例
4. 不要调用 plt.show()，不要保存文件
5. 只输出完整可运行的 Python 代码，不要注释掉关键语句，不要任何解释文字
6. 代码必须能够完整运行到最后一行，不要写到一半的注释或未完成的语句
7. 中文字体配置已由外部注入，不要重新设置与其冲突的字体，也不要关闭中文标题/标签
8. 点位可以用内部数值坐标绘制，但如果某点的 `<点名>_show_coordinates` 为 False，则标签里只能显示 `<点名>_label`，不能显示坐标
9. 如果 `display_equation_latex` 非空，则标题中的曲线方程必须直接使用它；若它是符号形式，就保持符号形式，不要替换成求解后的数值方程
{solution_hint}"""

    user_prompt = f"""题目：{latex_problem}

可用变量（已由外部成功注入，直接使用，不要假设其他变量存在）：
{available_vars}

注意：本题的曲线参数（如a、b、p）可能是题目待求量，数值未知。
请直接从参考解题过程中读取已经求出的参数和点坐标，不要重新推导。
只能使用上面列出的已成功注入变量；如果某个变量不在列表中，说明它未成功注入，不要使用该变量名。
如果存在 `display_equation_latex`，请把它直接用于标题中的曲线方程显示。
如果某个点存在 `<点名>_label` 和 `<点名>_show_coordinates`，请严格按它们控制标签文本；不要为 `_show_coordinates=False` 的点补出坐标。
不要调用 legend，也不要输出任何图例相关代码。
画出的图必须满足题目所有已知条件。

请直接开始写绘图代码，第一行就是 import 语句，最后一行是 plt.tight_layout()，中间只有绘图语句，不要有任何"经计算"、"根据题目"、"此处省略"等注释。{error_hint}"""

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4000,
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
