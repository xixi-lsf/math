from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, field_validator
import uuid

"""
题目系统中的核心数据结构，从点、直线、圆锥曲线参数，到完整的题目对象
"""
#带标签的坐标点
class Point(BaseModel):
    """A labeled point in the coordinate plane."""
    #名字
    name: str
    #坐标
    x: str  # sympy expression string, e.g. "-1", "sqrt(3)/2"
    y: str
    #绘图时标注文字的偏移量
    label_offset: tuple[float, float] = (0.15, 0.15)
    show_coordinates: bool = False
    display_label: Optional[str] = None

    #将 sympy 表达式求值并转换为浮点数
    def to_float(self) -> tuple[float, float]:
        from sympy import sympify
        return float(sympify(self.x)), float(sympify(self.y))

#直线参数，支持斜截式、两点式、竖直线
class LineParams(BaseModel):
    """Parameters for a line in the figure."""
    label: str = "l"
    #斜率
    slope: Optional[str] = None       # None means vertical line
    #y轴交点
    intercept: Optional[str] = None   # y-intercept (when slope is not None)
    x_fixed: Optional[str] = None     # x value for vertical line
    # Explicit endpoints override slope/intercept for bounded segments
    x1: Optional[str] = None
    y1: Optional[str] = None
    x2: Optional[str] = None
    y2: Optional[str] = None

# 圆锥曲线参数
class ConicParams(BaseModel):
    """Parameters for the conic section."""
    curve_type: Literal["ellipse", "hyperbola", "parabola", "polar_conic"]
    # Ellipse / Hyperbola: x²/a² ± y²/b² = 1
    a: Optional[str] = None
    b: Optional[str] = None
    c: Optional[str] = None  # redundant, stored for SymPy cross-check
    orientation: Literal["horizontal", "vertical"] = "horizontal"
    # Parabola: y² = 2px (right) or x² = 2py (up), etc.
    p: Optional[str] = None
    parabola_direction: Optional[Literal["right", "left", "up", "down"]] = None
    # Polar conic: r = ed/(1 ± e·cosθ)
    eccentricity: Optional[str] = None
    focal_distance: Optional[str] = None  # d in the polar form
    display_equation_latex: Optional[str] = None

#题目对象
class ProblemParams(BaseModel):
    """
    Single source of truth for all mathematical parameters.
    Both the LaTeX problem statement and the Matplotlib figure
    must be derived exclusively from this object.
    """
    problem_id: str
    topic: str          # "ellipse" | "hyperbola" | "parabola" | "polar"
    difficulty: int     # 1–5
    problem_type: str   # e.g. "focal_chord", "tangent_line", "area_calculation"


    conic: ConicParams#圆锥曲线
    key_points: list[Point] = []#点列表
    lines: list[LineParams] = []#直线列表
    constraints: list[str] = []  # extra sympy expression strings
    answer: str = ""             # sympy expression string for the final answer

    # Figure viewport
    plot_range_x: tuple[float, float] = (-6.0, 6.0)
    plot_range_y: tuple[float, float] = (-5.0, 5.0)
    show_grid: bool = False

    @classmethod
    def new(cls, **kwargs) -> "ProblemParams":
        kwargs.setdefault("problem_id", str(uuid.uuid4())[:8])
        return cls(**kwargs)


class ReasoningStep(BaseModel):
    """One step in the agent's reasoning trace, streamed to the frontend."""
    step_id: int
    node_name: str
    action: str        #人类可读的动作描述
    tool_called: Optional[str] = None
    #输入输出的简短摘要
    tool_input_summary: Optional[str] = None
    tool_output_summary: Optional[str] = None
    drawing_path: Optional[Literal["fast", "slow"]] = None  # adaptive drawing


class ValidationResult(BaseModel):
    is_valid: bool
    error_type: Optional[str] = None   # "param_inconsistency" | "no_solution" | "unclear"
    error_detail: Optional[str] = None
    suggested_fix: Optional[str] = None


class Problem(BaseModel):
    """Final output object returned to the frontend."""
    problem_id: str
    params: Optional[ProblemParams]
    latex_problem: str
    image_base64: str
    reasoning_trace: list[ReasoningStep] = []
    solution: Optional[str] = None
    solution_latex: Optional[str] = None  # populated on demand
    generation_config: dict = {}
    is_fallback: bool = False
