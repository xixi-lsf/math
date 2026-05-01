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
print(base64.b64encode(buf.read()).decode(), end="")
'''

#将 ProblemParams 对象中的数学参数转换为可供 LLM 代码直接使用的 Python 变量赋值语句
#LLM 生成的代码就可以直接使用 a, b, F1_x, F1_y 等变量名，而不必嵌入硬编码数字
def _build_param_block(params: ProblemParams) -> str:
    """
    Build a Python variable block from ProblemParams so the LLM code
    can reference named variables instead of raw numbers.
    """
    lines = []
    c = params.conic
    if c.a:
        lines.append(f"a = float(sympify({c.a!r}))")
    if c.b:
        lines.append(f"b = float(sympify({c.b!r}))")
    if c.c:
        lines.append(f"c_focal = float(sympify({c.c!r}))")
    if c.p:
        lines.append(f"p = float(sympify({c.p!r}))")
    if c.eccentricity:
        lines.append(f"e = float(sympify({c.eccentricity!r}))")

    for i, pt in enumerate(params.key_points):
        vname = pt.name.replace("₁", "1").replace("₂", "2").replace(" ", "_")
        lines.append(f"{vname}_x = float(sympify({pt.x!r}))")
        lines.append(f"{vname}_y = float(sympify({pt.y!r}))")

    for i, ln in enumerate(params.lines):
        if ln.slope:
            lines.append(f"line{i}_slope = float(sympify({ln.slope!r}))")
        if ln.intercept:
            lines.append(f"line{i}_intercept = float(sympify({ln.intercept!r}))")
        if ln.x_fixed:
            lines.append(f"line{i}_x_fixed = float(sympify({ln.x_fixed!r}))")

    lines.append(f"plot_x_min, plot_x_max = {params.plot_range_x[0]}, {params.plot_range_x[1]}")
    lines.append(f"plot_y_min, plot_y_max = {params.plot_range_y[0]}, {params.plot_range_y[1]}")
    return "\n".join(lines)


def execute_drawing_code(code: str, params: ProblemParams) -> tuple[str, str | None]:
    """
    Execute LLM-generated drawing code in a subprocess.

    Returns:
        (base64_png, error_message)
        On success: (base64_str, None)
        On failure: ("", error_message)
    """
    #调用 _build_param_block(params) 得到变量定义语句
    param_block = _build_param_block(params)
    #将参数块和用户代码嵌入模板
    script = _HARNESS_TEMPLATE.format(
        param_block=param_block,
        user_code=textwrap.dedent(code),
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
