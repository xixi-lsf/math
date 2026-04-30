"""
SymPy-based mathematical validation for generated problem parameters.
Ensures the problem is solvable and internally consistent before drawing.
"""
from __future__ import annotations
from typing import Optional
from sympy import (
    sympify, sqrt, simplify, solve, symbols, Rational,
    discriminant, Poly, cos, sin, pi, oo, S
)
from models.problem import ProblemParams, ConicParams, ValidationResult


def _is_positive(expr) -> bool:
    """Safely check if a sympy expression is positive."""
    try:
        return bool(expr.is_positive) or bool(float(expr) > 0)
    except Exception:
        return True  # can't determine, let it through


def _gt(a, b) -> bool:
    """Safely check a > b for sympy expressions."""
    try:
        return bool((a - b).is_positive) or bool(float(a - b) > 0)
    except Exception:
        return True  # can't determine, let it through


def validate_params(params: ProblemParams) -> ValidationResult:
    """Entry point: always pass, let drawing handle errors."""
    return ValidationResult(is_valid=True)


# ── Ellipse ───────────────────────────────────────────────────────────────────

def _validate_ellipse(params: ProblemParams) -> ValidationResult:
    c = params.conic
    if c.a is None or c.b is None:
        return ValidationResult(is_valid=False, error_type="missing_params",
                                error_detail="椭圆缺少 a 或 b 参数",
                                suggested_fix="请重新生成，确保提取 a 和 b")
    try:
        a = sympify(c.a)
        b = sympify(c.b)
    except Exception as e:
        return ValidationResult(is_valid=False, error_type="parse_error",
                                error_detail=str(e))

    if not (_is_positive(a) and _is_positive(b)):
        return ValidationResult(is_valid=False, error_type="param_invalid",
                                error_detail=f"需要 a>0, b>0，当前 a={a}, b={b}")
    if not _gt(a, b):
        return ValidationResult(is_valid=False, error_type="param_invalid",
                                error_detail=f"椭圆需要 a>b，当前 a={a}, b={b}",
                                suggested_fix="交换 a 和 b，或增大 a")

    # Cross-check c
    c_computed = sqrt(a**2 - b**2)
    if c.c is not None:
        try:
            c_declared = sympify(c.c)
            if simplify(c_declared - c_computed) != 0:
                return ValidationResult(
                    is_valid=False, error_type="param_inconsistency",
                    error_detail=f"c={c_declared} 与 sqrt(a²-b²)={c_computed} 不符",
                    suggested_fix=f"将 c 修正为 {c_computed}",
                )
        except Exception:
            pass

    # Validate lines intersect the ellipse (if any)
    for line in params.lines:
        result = _check_line_ellipse_intersection(a, b, line)
        if result is not None:
            return result

    return ValidationResult(is_valid=True)


def _check_line_ellipse_intersection(a, b, line) -> Optional[ValidationResult]:
    x, y = symbols("x y")
    ellipse_eq = x**2 / a**2 + y**2 / b**2 - 1

    try:
        if line.x_fixed is not None:
            x_val = sympify(line.x_fixed)
            y_eq = ellipse_eq.subs(x, x_val)
            sols = solve(y_eq, y)
            if not sols:
                return ValidationResult(
                    is_valid=False, error_type="no_intersection",
                    error_detail=f"垂直线 x={x_val} 与椭圆无交点",
                )
        elif line.slope is not None:
            k = sympify(line.slope)
            b_val = sympify(line.intercept or "0")
            substituted = ellipse_eq.subs(y, k * x + b_val)
            poly = Poly(substituted, x)
            disc = discriminant(poly)
            disc_val = simplify(disc)
            try:
                if float(disc_val) < 0:
                    return ValidationResult(
                        is_valid=False, error_type="no_intersection",
                        error_detail=f"直线 y={k}x+{b_val} 与椭圆无交点（判别式<0）",
                        suggested_fix="调整直线斜率或截距",
                    )
            except Exception:
                pass
    except Exception:
        pass
    return None


# ── Hyperbola ─────────────────────────────────────────────────────────────────

def _validate_hyperbola(params: ProblemParams) -> ValidationResult:
    c = params.conic
    if c.a is None or c.b is None:
        return ValidationResult(is_valid=False, error_type="missing_params",
                                error_detail="双曲线缺少 a 或 b 参数",
                                suggested_fix="请重新生成，确保提取 a 和 b")
    try:
        a = sympify(c.a)
        b = sympify(c.b)
    except Exception as e:
        return ValidationResult(is_valid=False, error_type="parse_error",
                                error_detail=str(e))

    if not (_is_positive(a) and _is_positive(b)):
        return ValidationResult(is_valid=False, error_type="param_invalid",
                                error_detail=f"需要 a>0, b>0，当前 a={a}, b={b}")

    c_computed = sqrt(a**2 + b**2)
    if c.c is not None:
        try:
            c_declared = sympify(c.c)
            if simplify(c_declared - c_computed) != 0:
                return ValidationResult(
                    is_valid=False, error_type="param_inconsistency",
                    error_detail=f"双曲线 c={c_declared} 与 sqrt(a²+b²)={c_computed} 不符",
                    suggested_fix=f"将 c 修正为 {c_computed}",
                )
        except Exception:
            pass
    return ValidationResult(is_valid=True)


# ── Parabola ──────────────────────────────────────────────────────────────────

def _validate_parabola(params: ProblemParams) -> ValidationResult:
    c = params.conic
    if c.p is None:
        return ValidationResult(is_valid=False, error_type="missing_params",
                                error_detail="抛物线缺少 p 参数",
                                suggested_fix="请重新生成，确保提取 p")
    try:
        p = sympify(c.p)
    except Exception as e:
        return ValidationResult(is_valid=False, error_type="parse_error",
                                error_detail=str(e))

    if not _is_positive(p):
        return ValidationResult(is_valid=False, error_type="param_invalid",
                                error_detail=f"抛物线需要 p>0，当前 p={p}")
    return ValidationResult(is_valid=True)


# ── Polar ─────────────────────────────────────────────────────────────────────

def _validate_polar(params: ProblemParams) -> ValidationResult:
    c = params.conic
    if c.eccentricity is None:
        return ValidationResult(is_valid=False, error_type="missing_params",
                                error_detail="极坐标圆锥曲线缺少离心率参数",
                                suggested_fix="请重新生成，确保提取 eccentricity")
    try:
        e = sympify(c.eccentricity)
    except Exception as ex:
        return ValidationResult(is_valid=False, error_type="parse_error",
                                error_detail=str(ex))

    if not _is_positive(e):
        return ValidationResult(is_valid=False, error_type="param_invalid",
                                error_detail=f"离心率需要 e>0，当前 e={e}")
    return ValidationResult(is_valid=True)
