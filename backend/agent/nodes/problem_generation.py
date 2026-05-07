"""
题目生成+参数提取
Node 2: Problem generation — 调用 LLM 根据主题、难度、检索到的知识片段，自由生成一道数学题目的 LaTeX 题干
Node 3: Parameter extraction — 调用LLM，从上一步生成的题干中提取结构化参数，组装成 ProblemParams 对象
"""
from __future__ import annotations
import json
import uuid
from openai import OpenAI

from agent.state import AgentState
from agent.logger import log_step
from knowledge.vectordb import get_store
from models.problem import (
    ProblemParams, ConicParams, Point, LineParams, ReasoningStep
)
from models.knowledge import KnowledgeChunk

#难度提示词
_DIFFICULTY_HINTS = {
    1: "基础题，直接代入公式即可，无需复杂推导",
    2: "简单题，需要一步推导，答案为整数或简单分数",
    3: "中等题，需要联立方程或综合两个知识点",
    4: "较难题，需要多步推导，可能涉及参数讨论",
    5: "竞赛难度，需要巧妙构造或多个知识点综合",
}

_DIFFICULTY_DESCRIPTIONS = {
    1: "一步推导，直接套公式即可",
    2: "两步推导，涉及基本性质",
    3: "需要综合两个知识点，有一定计算量",
    4: "条件隐蔽，需要多步推导和换元",
    5: "竞赛水平，需要构造辅助元素或多种方法结合",
}

_TOPIC_NAMES = {
    "ellipse": "椭圆",
    "hyperbola": "双曲线",
    "parabola": "抛物线",
    "polar": "极坐标圆锥曲线",
}

#从状态中读取 llm_config，动态创建 OpenAI 客户端实例
def _get_llm(state: AgentState) -> OpenAI:
    cfg = state.get("llm_config", {})
    return OpenAI(
        api_key=cfg.get("api_key", "sk-placeholder"),
        base_url=cfg.get("base_url", "https://api.openai.com/v1"),
    )

#将从知识库检索到的 KnowledgeChunk 对象（包含内容文本和 LaTeX 公式）格式化为字符串，供 LLM 参考
#最多取前 6 个片段，避免超出上下文窗口
def _knowledge_context(chunks: list[KnowledgeChunk]) -> str:
    if not chunks:
        return "（无检索结果）"
    parts = []
    for c in chunks[:6]:
        parts.append(f"- {c.content}\n  LaTeX: ${c.latex_formula}$")
    return "\n".join(parts)


# ── Node: problem_generation ──────────────────────────────────────────────────

def problem_generation_node(state: AgentState) -> dict:
    topic = state["topic"]
    difficulty = state["difficulty"]
    chunks = state.get("retrieved_knowledge", [])
    generation_retry = state.get("generation_retry", 0)
    step_id = state.get("step_counter", 0)
    llm = _get_llm(state)
    cfg = state.get("llm_config", {})
    model = cfg.get("model", "gpt-4o-mini")

    #获取topic,难度提示词，从知识库检索的知识（字符串）
    topic_cn = _TOPIC_NAMES.get(topic, topic)
    difficulty_label = _DIFFICULTY_HINTS.get(difficulty, "")
    difficulty_desc = _DIFFICULTY_DESCRIPTIONS[difficulty]
    knowledge_ctx = _knowledge_context(chunks)

    # 检索例题（所有难度都检索，用户例题优先，内置例题补足）
    example_section = ""
    selected_problem_ids: list[str] = state.get("selected_problem_ids", [])
    store = get_store()

    example_texts: list[str] = []

    # 用户例题（仅当有勾选时）
    if selected_problem_ids:
        query = f"{topic_cn} 难度{difficulty} {' '.join(state.get('subtopics', []))}"
        user_examples = store.retrieve_user_problems(
            query, selected_problem_ids, n_results=2
        )
        example_texts.extend(user_examples)

    # 内置例题补足（凑够 2 条）
    remaining = max(0, 2 - len(example_texts))
    if remaining > 0:
        builtin_examples = store.retrieve_examples(topic=topic, difficulty=difficulty, n=remaining)
        example_texts.extend(builtin_examples)

    if example_texts:
        example_section = (
            "\n\n参考例题（基于以下例题的结构进行改编，"
            "必须修改数值和问法，不得直接复制）：\n"
            + "\n\n".join(example_texts)
        )

    #重试提示：将 validation_result 中的错误详情和建议修正加入提示，引导 LLM 改进
    retry_hint = ""
    if generation_retry > 0:
        prev_result = state.get("validation_result")
        if prev_result and prev_result.error_detail:
            retry_hint = f"\n\n【上次生成的题目存在问题，请重新生成】\n错误：{prev_result.error_detail}\n建议：{prev_result.suggested_fix or '请调整参数'}"

    system_prompt = f"""你是一位专业的高中/竞赛数学出题专家，擅长解析几何。
请根据要求生成一道关于【{topic_cn}】的解析几何题目。

难度要求：{difficulty}/5 — {difficulty_label}

相关知识点参考：
{knowledge_ctx}

出题要求：
1. 题目必须有明确的解，答案应为"漂亮"的数（整数、简单分数或简单根式）
2. 题干用标准数学语言描述，所有数学表达式用 LaTeX 格式（用 $ 包裹行内公式）
3. 题目应包含：已知条件 + 求解目标
4. 参数选取要合理（如椭圆 a>b>0，抛物线 p>0）
5. 只输出题干文本，不要输出解答过程{retry_hint}"""

    user_prompt = f"""
请生成一道关于【{topic_cn}】的解析几何题目。
难度等级：{difficulty}/5（{difficulty_label}）
子知识点：{state.get("subtopics", [])}
参考知识点：{knowledge_ctx}{example_section}

要求：
- 题目必须数学严谨，条件不矛盾，有唯一确定的答案
- 难度{difficulty_label}意味着：{difficulty_desc}
- 使用标准 LaTeX 格式输出题干
"""

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        max_tokens=600,
    )
    #response.choices：一个列表，包含模型生成的多个候选回复
    #choices[0]：取第一个候选
    #.message：该候选中的 message 对象，包含 role（assistant）和 content（文本内容）
    latex_problem = response.choices[0].message.content.strip()

    step = ReasoningStep(
        step_id=step_id,
        node_name="problem_generation",
        action=f"LLM 生成题干（难度{difficulty}，{topic_cn}）",
        tool_called=f"LLM.chat ({model})",
        tool_input_summary=f"topic={topic}, difficulty={difficulty}",
        tool_output_summary=latex_problem,
    )

    q = state.get("step_queue")
    if q is not None:
        q.put_nowait(step)
    log_step(step, full_output=latex_problem)

    #生成的题目文本，追加一条推理记录，步骤计数器+1
    return {
        "latex_problem": latex_problem,
        "solution": None,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
        "generation_retry": generation_retry + 1,
    }


# ── Node: param_extraction ────────────────────────────────────────────────────
#描述期望提取的参数格式
_PARAM_SCHEMA = {
    "type": "object",
    "properties": {
        "curve_type": {"type": "string", "enum": ["ellipse", "hyperbola", "parabola", "polar_conic"]},
        "a": {"type": "string", "description": "半长轴，sympy表达式，如 '2' 或 'sqrt(5)'"},
        "b": {"type": "string"},
        "c": {"type": "string", "description": "半焦距，由a,b计算"},
        "p": {"type": "string", "description": "抛物线参数p（y²=2px中的p）"},
        "parabola_direction": {"type": "string", "enum": ["right", "left", "up", "down"]},
        "eccentricity": {"type": "string", "description": "极坐标圆锥曲线离心率"},
        "focal_distance": {"type": "string", "description": "极坐标中的d"},
        "orientation": {"type": "string", "enum": ["horizontal", "vertical"], "default": "horizontal"},
        "key_points": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "x": {"type": "string"},
                    "y": {"type": "string"},
                    "show_coordinates": {"type": "boolean"},
                    "display_label": {"type": "string"},
                },
                "required": ["name", "x", "y"],
            },
        },
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "slope": {"type": "string"},
                    "intercept": {"type": "string"},
                    "x_fixed": {"type": "string"},
                    "x1": {"type": "string", "description": "线段起点x坐标，sympy表达式"},
                    "y1": {"type": "string", "description": "线段起点y坐标，sympy表达式"},
                    "x2": {"type": "string", "description": "线段终点x坐标，sympy表达式"},
                    "y2": {"type": "string", "description": "线段终点y坐标，sympy表达式"},
                },
            },
        },
        "answer": {"type": "string", "description": "最终答案，sympy表达式"},
        "problem_type": {"type": "string", "description": "题型，如 focal_chord, tangent_line, area"},
        "display_equation_latex": {"type": "string", "description": "图上应显示的曲线方程；若题干未给出具体方程，则保留符号形式如 \\frac{x^2}{a^2}+\\frac{y^2}{b^2}=1，不要暴露求解后的具体数值"},
        "plot_range_x": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
        "plot_range_y": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
    },
    "required": ["curve_type", "problem_type"],
}


def param_extraction_node(state: AgentState) -> dict:
    latex_problem = state.get("latex_problem", "")
    step_id = state.get("step_counter", 0)
    llm = _get_llm(state)
    cfg = state.get("llm_config", {})
    model = cfg.get("model", "gpt-4o-mini")

    system_prompt = """你是数学参数提取专家。从给定的解析几何题目中提取所有数学参数，输出严格的 JSON。

规则：
1. 所有数值用 sympy 表达式字符串表示（如 "2", "sqrt(3)", "1/2"）
2. key_points 包含题目中所有命名点（焦点、顶点、交点等）及其坐标
3. lines 包含题目中所有直线的参数：
   - 如果是有端点的线段（如焦点弦、切线段），请同时填写 x1/y1/x2/y2 端点坐标
   - 如果是延伸到视口边界的完整直线，只填 slope/intercept 或 x_fixed
   - 如果只知道斜率和直线上某一点（点斜式），填写 slope 和 x1/y1，不填 intercept
4. answer 是题目要求求的量的答案（如果能从题目推断）
5. plot_range_x/y 根据曲线参数合理设置视口范围
6. curve_type 必须是以下之一：ellipse, hyperbola, parabola, polar_conic（极坐标圆锥曲线用 polar_conic，不要用 polar）
7. key_points 处理规则：
   - 如果点的坐标可以从题目已知条件直接算出（如焦点、顶点、题目给定坐标的点），必须计算出具体数值填入 x/y
   - 如果点的坐标是题目的待求量或不定点（如"椭圆上满足某条件的点P"但坐标未知），不要把它加入 key_points，因为无法画出准确位置
   - 对于类似"P在椭圆上，|PF1|=3"，"P在椭圆内部"，"P在椭圆外侧"的情况，P的坐标可以通过联立方程算出值或者范围，请计算后填入
8. 每个点必须返回 show_coordinates 字段：
   - 默认值为 false
   - 只有以下情况才设为 true：
     题干里用文字明确写出了该点的具体数字坐标，
     例如'点P(3,1)'、'点Q(0,2)'
   - 以下情况一律设为 false：
     焦点F、顶点A/B、交点、动点、题目中求解的点、
     仅给出了参数坐标如F(p/2, 0)的点
   - 宁可漏标也不要多标，多标坐标会暴露答案"
9. display_equation_latex 表示图上应该显示的曲线方程：
   - 如果题干明确给出了具体数值方程，可保留该数值形式
   - 如果题干没有直接给出具体方程，即使内部已经算出参数，也要返回符号形式（如 \\frac{x^2}{a^2}+\\frac{y^2}{b^2}=1 或 y^2=2px），不要暴露求解出的具体数字"""

    response = llm.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请从以下题目中提取参数：\n\n{latex_problem}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=800,
    )

    #解析 JSON 并构建对象
    raw_json = response.choices[0].message.content.strip()
    data = json.loads(raw_json)

    # Map topic "polar" → "polar_conic" for ConicParams
    _topic_to_curve = {"polar": "polar_conic"}
    fallback_curve = _topic_to_curve.get(state["topic"], state["topic"])

    # Build ProblemParams from extracted data
    raw_curve = data.get("curve_type", fallback_curve)
    # Normalize in case LLM returns "polar" instead of "polar_conic"
    raw_curve = _topic_to_curve.get(raw_curve, raw_curve)
    conic = ConicParams(
        curve_type=raw_curve,
        a=data.get("a"),
        b=data.get("b"),
        c=data.get("c"),
        p=data.get("p"),
        parabola_direction=data.get("parabola_direction"),
        eccentricity=data.get("eccentricity"),
        focal_distance=data.get("focal_distance"),
        orientation=data.get("orientation", "horizontal"),
        display_equation_latex=data.get("display_equation_latex"),
    )

    raw_points = data.get("key_points", [])
    if not isinstance(raw_points, list):
        raw_points = []
    key_points = []
    for pt in raw_points:
        if not isinstance(pt, dict):
            continue
        try:
            key_points.append(Point(
                name=pt["name"],
                x=str(pt["x"]),
                y=str(pt["y"]),
                show_coordinates=bool(pt.get("show_coordinates", False)),
                display_label=pt.get("display_label"),
            ))
        except Exception:
            pass

    raw_lines = data.get("lines", [])
    if not isinstance(raw_lines, list):
        raw_lines = []
    lines = []
    for ln in raw_lines:
        if not isinstance(ln, dict):
            continue
        try:
            lines.append(LineParams(
                label=ln.get("label", "l"),
                slope=ln.get("slope"),
                intercept=ln.get("intercept"),
                x_fixed=ln.get("x_fixed"),
                x1=str(ln["x1"]) if ln.get("x1") is not None else None,
                y1=str(ln["y1"]) if ln.get("y1") is not None else None,
                x2=str(ln["x2"]) if ln.get("x2") is not None else None,
                y2=str(ln["y2"]) if ln.get("y2") is not None else None,
            ))
        except Exception:
            pass

    plot_x = data.get("plot_range_x")
    plot_y = data.get("plot_range_y")
    if not isinstance(plot_x, list) or len(plot_x) < 2:
        plot_x = [-6.0, 6.0]
    if not isinstance(plot_y, list) or len(plot_y) < 2:
        plot_y = [-5.0, 5.0]

    # 自动扩展视口以包含所有关键点
    if key_points:
        try:
            from sympy import sympify as _sympify
            all_x = [float(_sympify(pt.x)) for pt in key_points]
            all_y = [float(_sympify(pt.y)) for pt in key_points]
            margin = 1.5
            auto_x_min = min(all_x) - margin
            auto_x_max = max(all_x) + margin
            auto_y_min = min(all_y) - margin
            auto_y_max = max(all_y) + margin
            final_x = (min(float(plot_x[0]), auto_x_min),
                       max(float(plot_x[1]), auto_x_max))
            final_y = (min(float(plot_y[0]), auto_y_min),
                       max(float(plot_y[1]), auto_y_max))
        except Exception:
            final_x = (float(plot_x[0]), float(plot_x[1]))
            final_y = (float(plot_y[0]), float(plot_y[1]))
    else:
        final_x = (float(plot_x[0]), float(plot_x[1]))
        final_y = (float(plot_y[0]), float(plot_y[1]))

    params = ProblemParams(
        problem_id=str(uuid.uuid4())[:8],
        topic=state["topic"],
        difficulty=state["difficulty"],
        problem_type=data.get("problem_type", "general"),
        conic=conic,
        key_points=key_points,
        lines=lines,
        answer=str(data.get("answer", "") or ""),
        plot_range_x=final_x,
        plot_range_y=final_y,
    )

    step = ReasoningStep(
        step_id=step_id,
        node_name="param_extraction",
        action=f"从题干提取结构化参数（题型：{params.problem_type}）",
        tool_called=f"LLM.json_mode ({model})",
        tool_input_summary="题干 → JSON 参数",
        tool_output_summary=f"curve={conic.curve_type}, a={conic.a}, b={conic.b}, points={len(key_points)}",
    )

    q = state.get("step_queue")
    if q is not None:
        q.put_nowait(step)
    log_step(step, full_output=raw_json)

    return {
        "params": params,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
