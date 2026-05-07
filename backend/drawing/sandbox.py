"""
Sandbox executor for LLM-generated Matplotlib drawing code (slow path).
Runs code in a subprocess with a timeout to prevent hangs or crashes.
绘图模块慢速路径的核心执行器，负责在安全沙箱中运行 LLM 生成的 Matplotlib 代码，
并捕获生成的图片（Base64）或错误信息。
"""
from __future__ import annotations
import base64
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from sympy import sympify

from models.problem import ProblemParams

#子进程最长允许运行 20 秒
_TIMEOUT_SECONDS = 20
#定义了子进程中将要执行的完整 Python 脚本框架
_HARNESS_TEMPLATE = '''
import sys, base64, io, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sympy import sympify
import matplotlib.font_manager as _fm

_candidates = [
    "Microsoft YaHei", "SimHei", "SimSun",
    "Noto Sans CJK SC", "Noto Sans SC",
    "WenQuanYi Micro Hei", "Arial Unicode MS",
    "PingFang SC", "Heiti SC",
]
_available = {{f.name for f in _fm.fontManager.ttflist}}
_matched = [name for name in _candidates if name in _available]
plt.rcParams["font.sans-serif"] = _matched or _candidates
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ── Injected parameters 参数注入模块(do NOT modify these values) ──────────────────────────
{param_block}

# ── LLM-generated drawing codeLLM 生成的绘图代码 ────────────────────────────────────────────────
{user_code}

# ── Capture output将当前图形保存为 PNG 到内存缓冲区，然后 Base64 编码并打印到标准输出─────────
buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
plt.close("all")
buf.seek(0)
#子进程与父进程通信的唯一通道：父进程通过捕获子进程的 stdout 来获取 Base64 字符串
import base64 as _b64
_raw = _b64.b64encode(buf.read()).decode()
print("".join(_raw.split()), end="")
'''

def _compute_viewport(params: ProblemParams) -> tuple[float, float, float, float]:
    """
    Compute a symmetric, complete viewport that:
    1. Shows the full conic curve (symmetric around origin)
    2. Contains all key points
    Returns (x_min, x_max, y_min, y_max).
    """
    c = params.conic
    # Start from curve extent
    try:
        if c.curve_type == "ellipse":
            a = float(sympify(c.a)) if c.a else 3.0
            b = float(sympify(c.b)) if c.b else 2.0
            if c.orientation == "vertical":
                a, b = b, a
            curve_x, curve_y = a, b
        elif c.curve_type == "hyperbola":
            a = float(sympify(c.a)) if c.a else 2.0
            b = float(sympify(c.b)) if c.b else 1.5
            curve_x = a * 2.5
            curve_y = b * 2.5
        elif c.curve_type == "parabola":
            p = float(sympify(c.p)) if c.p else 1.0
            curve_x = p * 6
            curve_y = p * 5
        else:
            curve_x, curve_y = 4.0, 4.0
    except Exception:
        curve_x, curve_y = 4.0, 4.0

    # Collect all key point coordinates
    pt_xs, pt_ys = [], []
    for pt in params.key_points:
        try:
            px, py = float(sympify(pt.x)), float(sympify(pt.y))
            pt_xs.append(px)
            pt_ys.append(py)
        except Exception:
            pass

    # Also consider line endpoints
    for ln in params.lines:
        for coord, lst in [(ln.x1, pt_xs), (ln.y1, pt_ys),
                           (ln.x2, pt_xs), (ln.y2, pt_ys)]:
            if coord is not None:
                try:
                    lst.append(float(sympify(coord)))
                except Exception:
                    pass

    # Symmetric extent from curve
    half_x = max(curve_x * 1.3, 2.0)
    half_y = max(curve_y * 1.3, 2.0)

    # Expand to include all points
    if pt_xs:
        half_x = max(half_x, max(abs(x) for x in pt_xs) * 1.3)
    if pt_ys:
        half_y = max(half_y, max(abs(y) for y in pt_ys) * 1.3)

    # Round up to a clean value
    import math
    half_x = math.ceil(half_x)
    half_y = math.ceil(half_y)

    return -half_x, half_x, -half_y, half_y


def _is_symbolic_equation(latex: str | None) -> bool:
    """Return True if the display equation is symbolic (contains letter params like a, b, p)."""
    if not latex:
        return False
    import re
    # Symbolic if it contains variable letters used as parameters (a, b, p, e)
    # but NOT if it's purely numeric (only digits, operators, sqrt, fractions with numbers)
    return bool(re.search(r'(?<![a-zA-Z])[abpe](?![a-zA-Z0-9_])', latex))


#将 ProblemParams 对象中的数学参数转换为可供 LLM 代码直接使用的 Python 变量赋值语句
#LLM 生成的代码就可以直接使用 a, b, F1_x, F1_y 等变量名，而不必嵌入硬编码数字
def _build_param_block(params: ProblemParams) -> tuple[str, list[str]]:
    """
    Build a Python variable block from ProblemParams so the LLM code
    can reference named variables instead of raw numbers.
    """
    lines: list[str] = []
    injected_vars: list[str] = []
    c = params.conic

    # Only inject numeric conic params (a, b, c, p) when the displayed equation
    # is already numeric (i.e. the problem statement gives the explicit equation).
    # If the equation is symbolic (e.g. x²/a²+y²/b²=1), injecting the solved
    # values would leak the answer through the drawing code.
    suppress_conic_params = _is_symbolic_equation(c.display_equation_latex)

    def _append_numeric_var(name: str, expr: str | None) -> None:
        if not expr:
            return
        try:
            val = float(sympify(expr))
            lines.append(f"{name} = {val}")
            injected_vars.append(name)
        except Exception:
            pass

    if not suppress_conic_params:
        _append_numeric_var("a", c.a)
        _append_numeric_var("b", c.b)
        _append_numeric_var("c_focal", c.c)
        _append_numeric_var("p", c.p)
        _append_numeric_var("e", c.eccentricity)

    for i, pt in enumerate(params.key_points):
        vname = pt.name.replace("₁", "1").replace("₂", "2").replace(" ", "_")
        _append_numeric_var(f"{vname}_x", pt.x)
        _append_numeric_var(f"{vname}_y", pt.y)
        lines.append(f"{vname}_label = {repr(pt.display_label or pt.name)}")
        lines.append(f"{vname}_show_coordinates = {pt.show_coordinates}")
        injected_vars.extend([f"{vname}_label", f"{vname}_show_coordinates"])

    for i, ln in enumerate(params.lines):
        _append_numeric_var(f"line{i}_slope", ln.slope)
        _append_numeric_var(f"line{i}_intercept", ln.intercept)
        _append_numeric_var(f"line{i}_x_fixed", ln.x_fixed)

    lines.append(f"display_equation_latex = {repr(c.display_equation_latex or '')}")
    injected_vars.append("display_equation_latex")
    lines.append(f"plot_x_min, plot_x_max = {params.plot_range_x[0]}, {params.plot_range_x[1]}")
    lines.append(f"plot_y_min, plot_y_max = {params.plot_range_y[0]}, {params.plot_range_y[1]}")
    injected_vars.extend(["plot_x_min", "plot_x_max", "plot_y_min", "plot_y_max"])

    # Pre-computed symmetric viewport (use these for set_xlim/set_ylim)
    vx_min, vx_max, vy_min, vy_max = _compute_viewport(params)
    lines.append(f"viewport_x_min, viewport_x_max = {vx_min}, {vx_max}")
    lines.append(f"viewport_y_min, viewport_y_max = {vy_min}, {vy_max}")
    injected_vars.extend(["viewport_x_min", "viewport_x_max", "viewport_y_min", "viewport_y_max"])

    return "\n".join(lines), injected_vars


def execute_drawing_code(code: str, params: ProblemParams) -> tuple[str, str | None]:
    """
    Execute LLM-generated drawing code in a subprocess.

    Returns:
        (base64_png, error_message)
        On success: (base64_str, None)
        On failure: ("", error_message)
    """
    #调用 _build_param_block(params) 得到变量定义语句
    param_block, _ = _build_param_block(params)
    code = textwrap.dedent(code).strip()
    #将参数块和用户代码嵌入模板
    script = _HARNESS_TEMPLATE.format(
        param_block=param_block,
        user_code=code,
    )

    #生成一个 .py 文件，将脚本写入磁盘
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        tmp_path = f.name

    try:
        #启动子进程
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
        )
        #执行出错
        if result.returncode != 0:
            return "", result.stderr.strip()
        output = result.stdout.strip()
        if not output:
            return "", "绘图代码未输出任何内容"
        # Validate it's valid base64
        base64.b64decode(output)
        return output, None
    except subprocess.TimeoutExpired:
        return "", f"绘图代码执行超时（>{_TIMEOUT_SECONDS}s）"
    except Exception as ex:
        return "", str(ex)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
