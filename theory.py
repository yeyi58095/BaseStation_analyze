# theory.py — analytic curves for WPCN (C=2)
from __future__ import annotations
from typing import Callable, Dict, Any, Optional, Tuple
import math

# ---------- helpers ----------
def stability_lambda_max(mu: float, e: float, C: int) -> float:
    if e <= 0 or mu <= 0:
        return 0.0
    r = mu / e
    if abs(r - 1.0) < 1e-12:
        return mu * (C / (C + 1.0))
    num = 1.0 - r**C
    den = 1.0 - r**(C + 1)
    if abs(den) < 1e-14:
        return min(e, mu)
    return mu * (num / den)

def _safe_pos(x: float, eps: float = 1e-12) -> float:
    return x if x > eps else eps

# ---------- Eq.(45) for C=2 : p_{0,1}(λ,μ,e) ----------
def p01_c2(lmb: float, mu: float, e: float) -> float:
    # 使用以 r = μ/e 表示的數值穩定寫法
    if lmb <= 0 or mu <= 0 or e <= 0:
        return 0.0
    r = mu / e
    # 來源：我們前面推的等價式；避免在 λ→0 或 r→1 時爆掉
    num = (1.0 - r**2) - (lmb / mu) * (1.0 - r**3)
    if lmb <= 0:
        return 0.0
    den = (1.0 - r) + (e / lmb) * (1.0 - r**2)
    if abs(den) < 1e-14:
        return 0.0
    return num / den

# ---------- Eq.(47) : P_es(λ,μ,e) ----------
def pes_c2(lmb: float, mu: float, e: float) -> float:
    if lmb <= 0 or mu <= 0 or e <= 0:
        return float("nan")
    p01 = p01_c2(lmb, mu, e)
    num = (e + mu) * mu * lmb * p01
    den = e * mu * (mu + e) - lmb * (mu*mu + mu*e + e*e)
    if abs(den) < 1e-14:
        return float("nan")
    y = num / den
    # clamp to [0,1]
    return max(0.0, min(1.0, y))

# ---------- Eq.(42) : L_D(λ,μ,e) ----------
def ld_c2(lmb: float, mu: float, e: float) -> float:
    # L_D = λ/(μ-λ) + [ μ^2 λ^2 p01 * ((e+μ)^3 - λ(e^2 + μ^2 + 3eμ)) ]
    #                      / [ (μ-λ) * ( eμ(e+μ) - λ(e^2+eμ+μ^2) )^2 ]
    if lmb <= 0 or mu <= 0 or e <= 0:
        return float("nan")
    if lmb >= mu:  # 服務率界
        return float("nan")
    p01 = p01_c2(lmb, mu, e)
    top1 = lmb / (mu - lmb)

    a = (e + mu)
    b = (e*e + e*mu + mu*mu)
    den1 = (mu - lmb)
    den2 = e*mu*a - lmb*b
    if abs(den2) < 1e-14:
        return float("nan")

    top2 = (mu*mu) * (lmb*lmb) * p01 * (a*a*a - lmb*(e*e + mu*mu + 3*e*mu))
    val = top1 + top2 / (den1 * (den2**2))
    return val

# ---------- Eq.(46) : W = L_D / λ ----------
def w_delay_c2(lmb: float, mu: float, e: float) -> float:
    if lmb <= 0:
        return float("nan")
    Ld = ld_c2(lmb, mu, e)
    return Ld / lmb if math.isfinite(Ld) else float("nan")

# ---------- Fig.6 / Fig.7 stability side constraints (48)(49) ----------
def mu_lower_bound_for_L(e: float, lmb: float) -> float:
    # μ > e * ( -1 + sqrt(1 + 4λ/(e-λ)) + 1 ) / 2  == e * (1 + sqrt(...)) / 2
    if e <= lmb:  # 避免負號或除零
        return float("inf")
    return e * (1.0 + math.sqrt(1.0 + 4.0*lmb / (e - lmb))) / 2.0

def e_lower_bound_for_L(mu: float, lmb: float) -> float:
    if mu <= lmb:
        return float("inf")
    return mu * (1.0 + math.sqrt(1.0 + 4.0*lmb / (mu - lmb))) / 2.0

# ---------- Router for make_figure.py ----------
def _domain_clip_by_xvar(xvar: str, fixed: Dict[str, Any]) -> Tuple[float, float]:
    # 回傳一個保守 domain，實際會被 make_figure 以「模擬點範圍」再裁一次
    mu = float(fixed.get("mu"))
    e  = float(fixed.get("e"))
    C  = int(fixed.get("C", 2))
    lmb = float(fixed.get("lambda", 0.0))

    if xvar == "lambda":
        lam_max = stability_lambda_max(mu, e, C)
        return (max(1e-6, 0.0), 0.999*lam_max)
    elif xvar == "mu":
        lo = max(mu_lower_bound_for_L(e, lmb), lmb + 1e-6)
        return (lo*1.001, max(lo*2.0, mu*2.0))
    elif xvar == "e":
        lo = max(e_lower_bound_for_L(mu, lmb), 1e-6)
        return (lo*1.001, max(lo*2.0, e*2.0))
    else:
        return (0.0, 1.0)

def get_curve(y_metric: str, fig_xvar: str, fixed: Dict[str, Any]):
    """
    Return (f, (xmin,xmax), label_suffix) for C=2; otherwise None.
    y_metric in {"L","W","P_es"}.
    """
    C = int(fixed.get("C", 2))
    if C != 2:
        return None

    mu = float(fixed.get("mu"))
    e  = float(fixed.get("e"))
    lmb = float(fixed.get("lambda", 0.0))

    # choose f(x)
    if y_metric == "P_es":
        if fig_xvar == "lambda":
            f = lambda x: pes_c2(x, mu, e)
        elif fig_xvar == "mu":
            f = lambda x: pes_c2(lmb, x, e)
        elif fig_xvar == "e":
            f = lambda x: pes_c2(lmb, mu, x)
        else:
            return None
        dom = _domain_clip_by_xvar(fig_xvar, fixed)
        return f, dom, "(theory)"

    if y_metric == "L":
        if fig_xvar == "lambda":
            f = lambda x: ld_c2(x, mu, e)
        elif fig_xvar == "mu":
            f = lambda x: ld_c2(lmb, x, e)
        elif fig_xvar == "e":
            f = lambda x: ld_c2(lmb, mu, x)
        else:
            return None
        dom = _domain_clip_by_xvar(fig_xvar, fixed)
        return f, dom, "(theory)"

    if y_metric == "W":
        if fig_xvar == "lambda":
            f = lambda x: w_delay_c2(x, mu, e)
        elif fig_xvar == "mu":
            f = lambda x: w_delay_c2(lmb, x, e)
        elif fig_xvar == "e":
            f = lambda x: w_delay_c2(lmb, mu, x)
        else:
            return None
        dom = _domain_clip_by_xvar(fig_xvar, fixed)
        return f, dom, "(theory)"

    return None
