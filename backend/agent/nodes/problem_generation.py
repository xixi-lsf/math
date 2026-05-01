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
    retry_count = state.get("retry_count", 0)
    step_id = state.get("step_counter", 0)
    llm = _get_llm(state)
    cfg = state.get("llm_config", {})
    model = cfg.get("model", "gpt-4o-mini")

    #获取topic,难度提示词，从知识库检索的知识（字符串）
    topic_cn = _TOPIC_NAMES.get(topic, topic)
    difficulty_hint = _DIFFICULTY_HINTS.get(difficulty, "")
    knowledge_ctx = _knowledge_context(chunks)

    #重试提示：将 validation_result 中的错误详情和建议修正加入提示，引导 LLM 改进
    retry_hint = ""
    if retry_count > 0:
        prev_result = state.get("validation_result")
        if prev_result and prev_result.error_detail:
            retry_hint = f"\n\n【上次生成的题目存在问题，请重新生成】\n错误：{prev_result.error_detail}\n建议：{prev_result.suggested_fix or '请调整参数'}"

    system_prompt = f"""你是一位专业的高中/竞赛数学出题专家，擅长解析几何。
请根据要求生成一道关于【{topic_cn}】的解析几何题目。

难度要求：{difficulty}/5 — {difficulty_hint}

相关知识点参考：
{knowledge_ctx}

出题要求：
1. 题目必须有明确的解，答案应为"漂亮"的数（整数、简单分数或简单根式）
2. 题干用标准数学语言描述，所有数学表达式用 LaTeX 格式（用 $ 包裹行内公式）
3. 题目应包含：已知条件 + 求解目标
4. 参数选取要合理（如椭圆 a>b>0，抛物线 p>0）
5. 只输出题干文本，不要输出解答过程{retry_hint}"""

    user_prompt = f"请生成一道难度为 {difficulty}/5 的{topic_cn}解析几何题目。"

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
        tool_output_summary=latex_problem[:120] + ("..." if len(latex_problem) > 120 else ""),
    )

    #生成的题目文本，追加一条推理记录，步骤计数器+1
    return {
        "latex_problem": latex_problem,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
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
                },
            },
        },
        "answer": {"type": "string", "description": "最终答案，sympy表达式"},
        "problem_type": {"type": "string", "description": "题型，如 focal_chord, tangent_line, area"},
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
3. lines 包含题目中所有直线的参数
4. answer 是题目要求求的量的答案（如果能从题目推断）
5. plot_range_x/y 根据曲线参数合理设置视口范围
6. curve_type 必须是以下之一：ellipse, hyperbola, parabola, polar_conic（极坐标圆锥曲线用 polar_conic，不要用 polar）"""

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
    )

    raw_points = data.get("key_points", [])
    if not isinstance(raw_points, list):
        raw_points = []
    key_points = []
    for pt in raw_points:
        if not isinstance(pt, dict):
            continue
        try:
            key_points.append(Point(name=pt["name"], x=str(pt["x"]), y=str(pt["y"])))
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
            ))
        except Exception:
            pass

    plot_x = data.get("plot_range_x")
    plot_y = data.get("plot_range_y")
    if not isinstance(plot_x, list) or len(plot_x) < 2:
        plot_x = [-6.0, 6.0]
    if not isinstance(plot_y, list) or len(plot_y) < 2:
        plot_y = [-5.0, 5.0]

    params = ProblemParams(
        problem_id=str(uuid.uuid4())[:8],
        topic=state["topic"],
        difficulty=state["difficulty"],
        problem_type=data.get("problem_type", "general"),
        conic=conic,
        key_points=key_points,
        lines=lines,
        answer=str(data.get("answer", "") or ""),
        plot_range_x=(float(plot_x[0]), float(plot_x[1])),
        plot_range_y=(float(plot_y[0]), float(plot_y[1])),
    )

    step = ReasoningStep(
        step_id=step_id,
        node_name="param_extraction",
        action=f"从题干提取结构化参数（题型：{params.problem_type}）",
        tool_called=f"LLM.json_mode ({model})",
        tool_input_summary="题干 → JSON 参数",
        tool_output_summary=f"curve={conic.curve_type}, a={conic.a}, b={conic.b}, points={len(key_points)}",
    )

    return {
        "params": params,
        "reasoning_trace": state.get("reasoning_trace", []) + [step],
        "step_counter": step_id + 1,
    }
