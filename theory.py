# theory.py — analytic curves (C=2) for RA/Result
from __future__ import annotations
import math
from typing import Dict, Tuple, Callable, Optional

# ---- p_{0,1} for C=2 (Eq. 45 closed-form) ----
def p01_c2(lmb: float, mu: float, e: float) -> float:
    if lmb <= 0 or mu <= 0 or e <= 0:
        return 0.0
    r = mu / e
    num = (1.0 - r*r) - (lmb/mu)*(1.0 - r*r*r)
    den = (1.0 - r) + (e/lmb)*(1.0 - r*r)
    if abs(den) < 1e-14:
        return 0.0
    return num / den

# ---- P_es (Eq. 47) ----
def pes_c2(lmb: float, mu: float, e: float) -> float:
    if lmb <= 0 or mu <= 0 or e <= 0:
        return float("nan")
    p01 = p01_c2(lmb, mu, e)
    num = (e + mu) * mu * lmb * p01
    den = e * mu * (mu + e) - lmb * (mu*mu + e*mu + e*e)
    if abs(den) < 1e-14:
        return float("nan")
    y = num / den
    return max(0.0, min(1.0, y))

# ---- L_D (Eq. 42) ----
def ld_c2(lmb: float, mu: float, e: float) -> float:
    if lmb <= 0 or mu <= 0 or e <= 0 or lmb >= mu:
        return float("nan")
    p01 = p01_c2(lmb, mu, e)
    a = e + mu
    b = e*e + e*mu + mu*mu
    den1 = (mu - lmb)
    den2 = e*mu*a - lmb*b
    if abs(den2) < 1e-14:
        return float("nan")
    top1 = lmb / den1
    top2 = (mu*mu) * (lmb*lmb) * p01 * (a*a*a - lmb*(e*e + mu*mu + 3*e*mu))
    return top1 + top2 / (den1 * (den2**2))

# ---- Stability boundaries used by Fig6/7 (Eq. 48, 49) ----
def mu_min_for_stability(e: float, lmb: float) -> float:
    # μ > e * ( -1 + sqrt(1 + 4λ/(e-λ)) )
    if e <= lmb:
        return float("inf")
    return e * (-1.0 + math.sqrt(1.0 + 4.0*lmb/(e - lmb)))

def e_min_for_stability(mu: float, lmb: float) -> float:
    # e > μ * ( -1 + sqrt(1 + 4λ/(μ-λ)) ) / 2
    if mu <= lmb:
        return float("inf")
    return mu * (-1.0 + math.sqrt(1.0 + 4.0*lmb/(mu - lmb))) / 2.0

# ---- Convenience wrappers for each figure axis ----
def ld_vs_mu(mu_x: float, lmb: float, e: float) -> float:
    if not math.isfinite(mu_x) or mu_x <= mu_min_for_stability(e, lmb):
        return float("nan")
    return ld_c2(lmb, mu_x, e)

def ld_vs_e(e_x: float, lmb: float, mu: float) -> float:
    if not math.isfinite(e_x) or e_x <= e_min_for_stability(mu, lmb):
        return float("nan")
    return ld_c2(lmb, mu, e_x)

def pes_vs_mu(mu_x: float, lmb: float, e: float) -> float:
    return pes_c2(lmb, mu_x, e)

def pes_vs_e(e_x: float, lmb: float, mu: float) -> float:
    return pes_c2(lmb, mu, e_x)

# ---- Router for make_figure.py ----
def get_curve(y_metric: str, fig_xvar: str, fixed: Dict[str, float]):
    # 只支援 C=2 的封閉式
    C = int(fixed.get("C", 2))
    if C != 2:
        return None

    wide = (float("-inf"), float("inf"))

    if y_metric == "P_es":
        if fig_xvar == "lambda":
            mu = float(fixed["mu"]); e = float(fixed["e"])
            return (lambda x: pes_c2(x, mu, e), wide, "(theory)")
        if fig_xvar == "mu":
            lam = float(fixed["lambda"]); e = float(fixed["e"])
            return (lambda x: pes_c2(lam, x, e), wide, "(theory)")
        if fig_xvar == "e":
            lam = float(fixed["lambda"]); mu = float(fixed["mu"])
            return (lambda x: pes_c2(lam, mu, x), wide, "(theory)")
        return None

    if y_metric == "L":
        if fig_xvar == "lambda":
            mu = float(fixed["mu"]); e = float(fixed["e"])
            return (lambda x: ld_c2(x, mu, e), wide, "(theory)")
        if fig_xvar == "mu":
            lam = float(fixed["lambda"]); e = float(fixed["e"])
            return (lambda x: ld_vs_mu(x, lam, e), wide, "(theory)")
        if fig_xvar == "e":
            lam = float(fixed["lambda"]); mu = float(fixed["mu"])
            return (lambda x: ld_vs_e(x, lam, mu), wide, "(theory)")
        return None

    if y_metric == "W":
        if fig_xvar == "lambda":
            mu = float(fixed["mu"]); e = float(fixed["e"])
            return (lambda x: (ld_c2(x, mu, e)/x if x>0 else float("nan")), wide, "(theory)")
        if fig_xvar == "mu":
            lam = float(fixed["lambda"]); e = float(fixed["e"])
            return (lambda x: (ld_vs_mu(x, lam, e)/lam if lam>0 else float("nan")), wide, "(theory)")
        if fig_xvar == "e":
            lam = float(fixed["lambda"]); mu = float(fixed["mu"])
            return (lambda x: (ld_vs_e(x, lam, mu)/lam if lam>0 else float("nan")), wide, "(theory)")
        return None

    return None
