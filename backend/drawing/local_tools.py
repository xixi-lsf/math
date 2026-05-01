"""
Local drawing tools (fast path).
Each function takes a ProblemParams and returns a base64-encoded PNG string,
or raises NotImplementedError if the problem type is not covered.
"""
from __future__ import annotations
import base64
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# Use a CJK-capable font for Chinese labels in figures
for _font_name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]:
    if any(f.name == _font_name for f in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _font_name
        break
plt.rcParams["axes.unicode_minus"] = False
import matplotlib.patches as mpatches
from sympy import sympify, sqrt, cos, sin, pi, lambdify, symbols

from models.problem import ProblemParams, LineParams, Point

"""
绘图模块的快速路径实现，
负责使用本地模板函数直接生成常见圆锥曲线的配图，避免调用 LLM 生成代码
"""
# ── Helpers ───────────────────────────────────────────────────────────────────
"""
将 Matplotlib 图形保存到内存 BytesIO，编码为 PNG，再转为 Base64 字符串
"""
def _fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()

"""
创建图形和坐标轴
"""
def _setup_axes(params: ProblemParams) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(7, 6))
    #设置 x/y 范围
    ax.set_xlim(*params.plot_range_x)
    ax.set_ylim(*params.plot_range_y)
    #绘制黑色坐标轴
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    #纵横比一致
    ax.set_aspect("equal")
    #是否显示虚线网格
    if params.show_grid:
        ax.grid(True, linestyle="--", alpha=0.4)
    ax.tick_params(labelsize=9)
    #fig是画布，ax是坐标系
    return fig, ax

"""
遍历 params.key_points,画点
"""
def _draw_points(ax: plt.Axes, points: list[Point]) -> None:
    for pt in points:
        px, py = pt.to_float()
        ax.plot(px, py, "ro", markersize=5, zorder=5)
        ax.annotate(
            f"$\\mathit{{{pt.name}}}$",
            (px, py),
            xytext=(px + pt.label_offset[0], py + pt.label_offset[1]),
            fontsize=11,
        )

"""
画线
"""
def _draw_lines(ax: plt.Axes, lines: list[LineParams], x_range: tuple) -> None:
    x_arr = np.linspace(x_range[0], x_range[1], 400)
    for line in lines:
        if line.x1 and line.x2:
            # Explicit segment
            x1, y1 = float(sympify(line.x1)), float(sympify(line.y1))
            x2, y2 = float(sympify(line.x2)), float(sympify(line.y2))
            ax.plot([x1, x2], [y1, y2], "b-", linewidth=1.5, label=f"${line.label}$")
        elif line.x_fixed is not None:
            xv = float(sympify(line.x_fixed))
            ax.axvline(xv, color="blue", linewidth=1.5, label=f"${line.label}$")
        elif line.slope is not None:
            k = float(sympify(line.slope))
            b = float(sympify(line.intercept or "0"))
            y_arr = k * x_arr + b
            ax.plot(x_arr, y_arr, "b-", linewidth=1.5, label=f"${line.label}$")


# ── Ellipse ───────────────────────────────────────────────────────────────────

def draw_ellipse(params: ProblemParams) -> str:
    c = params.conic
    a = float(sympify(c.a))
    b = float(sympify(c.b))

    #画坐标轴，创建图形
    fig, ax = _setup_axes(params)
    theta = np.linspace(0, 2 * np.pi, 1000)
    if c.orientation == "horizontal":
        ax.plot(a * np.cos(theta), b * np.sin(theta), "k-", linewidth=2)
    else:
        ax.plot(b * np.cos(theta), a * np.sin(theta), "k-", linewidth=2)

    """
    调用 _draw_lines 和 _draw_points画点和线，这样调用draw_ellipse得到的不只是椭圆，是组合图形
    """
    _draw_lines(ax, params.lines, params.plot_range_x)
    _draw_points(ax, params.key_points)
    ax.set_title(f"椭圆 $\\frac{{x^2}}{{{_fmt(a**2)}}}+\\frac{{y^2}}{{{_fmt(b**2)}}}=1$", fontsize=12)
    return _fig_to_b64(fig)


# ── Hyperbola ─────────────────────────────────────────────────────────────────
#双曲线（也画出了渐近线）
def draw_hyperbola(params: ProblemParams) -> str:
    c = params.conic
    a = float(sympify(c.a))
    b = float(sympify(c.b))

    fig, ax = _setup_axes(params)
    t = np.linspace(-2.5, 2.5, 800)

    #通过 params.conic.orientation 字段区分水平和竖直
    if c.orientation == "horizontal":
        # Right branch
        ax.plot(a * np.cosh(t), b * np.sinh(t), "k-", linewidth=2)
        # Left branch
        ax.plot(-a * np.cosh(t), b * np.sinh(t), "k-", linewidth=2)
        # Asymptotes (dashed)
        x_a = np.linspace(*params.plot_range_x, 400)
        ax.plot(x_a, (b / a) * x_a, "gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(x_a, -(b / a) * x_a, "gray", linestyle="--", linewidth=1, alpha=0.7)
    else:
        ax.plot(b * np.sinh(t), a * np.cosh(t), "k-", linewidth=2)
        ax.plot(b * np.sinh(t), -a * np.cosh(t), "k-", linewidth=2)
        x_a = np.linspace(*params.plot_range_x, 400)
        ax.plot(x_a, (a / b) * x_a, "gray", linestyle="--", linewidth=1, alpha=0.7)
        ax.plot(x_a, -(a / b) * x_a, "gray", linestyle="--", linewidth=1, alpha=0.7)

    _draw_lines(ax, params.lines, params.plot_range_x)
    _draw_points(ax, params.key_points)
    ax.set_title(f"双曲线 $\\frac{{x^2}}{{{_fmt(a**2)}}}-\\frac{{y^2}}{{{_fmt(b**2)}}}=1$", fontsize=12)
    return _fig_to_b64(fig)


# ── Parabola ──────────────────────────────────────────────────────────────────

def draw_parabola(params: ProblemParams) -> str:
    c = params.conic
    p = float(sympify(c.p))
    direction = c.parabola_direction or "right"

    fig, ax = _setup_axes(params)
    t = np.linspace(*params.plot_range_y, 800)

    #开口方向
    if direction == "right":
        y_arr = t
        x_arr = y_arr ** 2 / (2 * p)
        # Clip to x range
        mask = (x_arr >= params.plot_range_x[0]) & (x_arr <= params.plot_range_x[1])
        ax.plot(x_arr[mask], y_arr[mask], "k-", linewidth=2)
        # Directrix
        ax.axvline(-p / 2, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    elif direction == "left":
        y_arr = t
        x_arr = -(y_arr ** 2) / (2 * p)
        mask = (x_arr >= params.plot_range_x[0]) & (x_arr <= params.plot_range_x[1])
        ax.plot(x_arr[mask], y_arr[mask], "k-", linewidth=2)
        ax.axvline(p / 2, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    elif direction == "up":
        x_arr = np.linspace(*params.plot_range_x, 800)
        y_arr = x_arr ** 2 / (2 * p)
        mask = (y_arr >= params.plot_range_y[0]) & (y_arr <= params.plot_range_y[1])
        ax.plot(x_arr[mask], y_arr[mask], "k-", linewidth=2)
        ax.axhline(-p / 2, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    else:  # down
        x_arr = np.linspace(*params.plot_range_x, 800)
        y_arr = -(x_arr ** 2) / (2 * p)
        mask = (y_arr >= params.plot_range_y[0]) & (y_arr <= params.plot_range_y[1])
        ax.plot(x_arr[mask], y_arr[mask], "k-", linewidth=2)
        ax.axhline(p / 2, color="gray", linestyle="--", linewidth=1, alpha=0.7)

    _draw_lines(ax, params.lines, params.plot_range_x)
    _draw_points(ax, params.key_points)
    ax.set_title(f"抛物线 $y^2={_fmt(2*p)}x$", fontsize=12)
    return _fig_to_b64(fig)


# ── Polar conic ───────────────────────────────────────────────────────────────

def draw_polar_conic(params: ProblemParams) -> str:
    c = params.conic
    e = float(sympify(c.eccentricity))
    d = float(sympify(c.focal_distance or "1"))

    fig, ax = _setup_axes(params)
    theta = np.linspace(0, 2 * np.pi, 2000)
    denom = 1 - e * np.cos(theta)
    # Avoid division by zero for hyperbola
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(np.abs(denom) > 1e-6, e * d / denom, np.nan)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    # Clip to viewport
    mask = (
        (x >= params.plot_range_x[0]) & (x <= params.plot_range_x[1]) &
        (y >= params.plot_range_y[0]) & (y <= params.plot_range_y[1])
    )
    ax.plot(x[mask], y[mask], "k-", linewidth=2)
    _draw_points(ax, params.key_points)
    ax.set_title(f"极坐标圆锥曲线 $e={e:.3g}$", fontsize=12)
    return _fig_to_b64(fig)


# ── Dispatcher ────────────────────────────────────────────────────────────────
"""
drawing_retry == 0 时，会首先调用 try_local_draw(params) 尝试本地绘图
如果当前题目的曲线类型和参数满足预设模板,直接绘制并返回 Base64 编码的 PNG 图片
若不支持（例如有旋转、倾斜直线等复杂情况），则抛出 NotImplementedError，由上层降级到慢速路径（LLM 生成代码 + 沙箱执行）
"""
def try_local_draw(params: ProblemParams) -> str:
    """
    Attempt to draw using local template functions.
    Returns base64 PNG on success, raises NotImplementedError if not supported.
    """
    ct = params.conic.curve_type
    if ct == "ellipse":
        return draw_ellipse(params)
    elif ct == "hyperbola":
        return draw_hyperbola(params)
    elif ct == "parabola":
        return draw_parabola(params)
    elif ct == "polar_conic":
        return draw_polar_conic(params)
    raise NotImplementedError(f"No local drawer for curve_type={ct!r}")


# ── Utility ───────────────────────────────────────────────────────────────────

def _fmt(val: float) -> str:
    """Format a float as integer string if it's a whole number."""
    if val == int(val):
        return str(int(val))
    return f"{val:.4g}"
