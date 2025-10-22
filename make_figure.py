#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_figure.py — Batch simulator + plotter for WPCN paper figures (Fig.4~Fig.11)

Directory layout (relative to repo root `RA/`):
- Projects/BaseStation/Win32/Debug/Project2.exe  # simulator (already built)
- Result/
    make_figure.py                                # this file
    theory.py                                     # (optional) analytic curves plug-in
    PaperX/
      FigY/
        config.yaml                               # figure recipe
        out/                                      # auto-created, raw CSVs
        figY.png                                  # auto-generated plot

Usage:
    # If you run from RA/Result, set sim.exe to "../Projects/.../Project2.exe"
    python3 make_figure.py Paper1/Fig4/config.yaml

    # Or run from RA/ with original paths
    python3 Result/make_figure.py Result/Paper1/Fig4/config.yaml

    # Run all configs under Result/**/Fig*/config.yaml
    python3 Result/make_figure.py --all

Notes:
- Stability guard uses λ_max from inequality (10): λ < μ * (1 - (μ/e)^C) / (1 - (μ/e)^(C+1)).
- From now on, simulation points are *not connected* (points only). Error bars/bands still supported.
- Theory lines are drawn by default (no YAML change). Put closed-form/solvers into Result/theory.py.
  If theory is unavailable for a curve, nothing is drawn (no error). You can also pass --no-theory.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import subprocess
import sys
import time
import uuid
import statistics
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- Optional dependency ----------
try:
    import yaml  # PyYAML
except Exception:
    print("[make_figure] Missing dependency: PyYAML (pip install pyyaml)")
    raise

# ---------- Plot ----------
import matplotlib.pyplot as plt
from math import sqrt

# ---------- Optional theory plug-in ----------
_theory_mod = None
try:
    import importlib
    _theory_mod = importlib.import_module("theory")  # look for Result/theory.py
    print("[theory] plug-in loaded.")
except Exception:
    _theory_mod = None
    print("[theory] no plug-in found; only simulation points will be drawn where theory is missing.")

# ------------------------- Utilities -------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def stability_lambda_max(mu: float, e: float, C: int) -> float:
    """Compute λ_max from inequality (10):
        λ < μ * (1 - (μ/e)^C) / (1 - (μ/e)^(C+1))
        Handles edge cases to avoid division-by-zero.
    """
    if e <= 0 or mu <= 0:
        return 0.0
    ratio = mu / e
    if abs(ratio - 1.0) < 1e-9:
        return mu * (C / (C + 1.0))
    num = 1.0 - (ratio ** C)
    den = 1.0 - (ratio ** (C + 1))
    if abs(den) < 1e-12:
        return min(e, mu)
    return mu * (num / den)

def mu_min_for_stability(e: float, lmb: float) -> float:
    # Eq.(48): μ > e * (-1 + sqrt(1 + 4λ/(e-λ)))
    if e <= lmb:
        return float("inf")
    return e * (-1.0 + math.sqrt(1.0 + 4.0*lmb/(e - lmb)))

def e_min_for_stability(mu: float, lmb: float) -> float:
    # Eq.(49): e > μ * (-1 + sqrt(1 + 4λ/(μ-λ))) / 2
    if mu <= lmb:
        return float("inf")
    return mu * (-1.0 + math.sqrt(1.0 + 4.0*lmb/(mu - lmb))) / 2.0


def frange(start: float, end: float, step: float) -> List[float]:
    xs = []
    if step == 0:
        return xs
    x = start
    if start <= end:
        while x <= end + 1e-12:
            xs.append(round(float(x), 10))
            x += step
    else:
        while x >= end - 1e-12:
            xs.append(round(float(x), 10))
            x -= abs(step)
    return xs

def merged_params(sim: SimConfig, curve: CurveSpec) -> Dict[str, Any]:
    """Merge sim.base_args and curve.fixed for theory; curve.fixed overrides."""
    params: Dict[str, Any] = {}
    for k in ("mu", "e", "C", "lambda"):
        if k in sim.base_args and sim.base_args[k] is not None:
            params[k] = sim.base_args[k]
    for k in ("mu", "e", "C", "lambda"):
        if k in curve.fixed and curve.fixed[k] is not None:
            params[k] = curve.fixed[k]
    if "C" not in params or params["C"] is None:
        params["C"] = 2
    return params


def dyadic_refine_left(start: float, x_asym: float, levels: int, terminal_clip: float) -> List[float]:
    """Generate points approaching x_asym from the left with dyadic spacing."""
    if not math.isfinite(start) or not math.isfinite(x_asym) or x_asym <= start:
        return []
    d0 = x_asym - start
    xs = []
    for k in range(1, levels + 1):
        xk = x_asym - d0 / (2.0 ** k)
        xs.append(xk)
    if terminal_clip is not None:
        xs = [x for x in xs if x <= x_asym * terminal_clip]
    return xs

def dyadic_refine_right(end: float, x_asym: float, levels: int, terminal_clip: float) -> List[float]:
    """Generate points approaching x_asym from the right with dyadic spacing."""
    if not math.isfinite(end) or not math.isfinite(x_asym) or x_asym >= end:
        return []
    d0 = end - x_asym
    xs = []
    for k in range(1, levels + 1):
        xk = x_asym + d0 / (2.0 ** k)
        xs.append(xk)
    if terminal_clip is not None and terminal_clip > 1.0:
        xs = [x for x in xs if x >= x_asym * terminal_clip]
    return xs

def run_sim_once(exe: str, args_map: Dict[str, Any], out_csv: str) -> None:
    ensure_dir(os.path.dirname(out_csv))
    cmd = [exe, "--headless"]
    for k, v in args_map.items():
        if isinstance(v, bool):
            if v:
                cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")
    cmd.append(f"--out={out_csv}")
    print("[run]", " ".join(cmd))
    try:
        cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        raise RuntimeError(f"Simulator not found: {exe}")
    if cp.returncode != 0:
        print(cp.stdout)
        print(cp.stderr, file=sys.stderr)
        raise RuntimeError(f"Simulator exited with {cp.returncode}")

def read_metric_from_csv(csv_path: str, metric: str) -> float:
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        if not rows:
            raise ValueError(f"No data rows in {csv_path}")
        val = rows[-1].get(metric)
        if val is None:
            raise KeyError(f"Metric '{metric}' not found in header of {csv_path}")
        return float(val)

# ------------------------- Data classes -------------------------

@dataclass
class CurveSpec:
    label: str
    fixed: Dict[str, Any]
    x_values: Optional[List[float]] = None
    x_auto: Optional[Dict[str, Any]] = None

@dataclass
class FigureSpec:
    paper: str
    id: str
    title: str
    x_var: str
    y_metric: str
    ylabel: str
    xlabel: str
    x_ticks: List[float] | None
    legend: bool
    save_as: str
    aggregate_runs: int
    stability_guard: bool
    auto_defaults: Dict[str, Any]
    asymptote: Dict[str, Any]
    stats: Dict[str, Any]
    parallel_workers: int = 0
    # theory 控制：預設 auto（畫理論；不用 YAML）
    theory: Dict[str, Any] = None

@dataclass
class SimConfig:
    exe: str
    base_args: Dict[str, Any]

# ------------------------- Core Loader -------------------------

def load_yaml(path: str) -> Tuple[FigureSpec, SimConfig, List[CurveSpec], str]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fig = FigureSpec(
        paper=cfg["figure"]["paper"],
        id=cfg["figure"]["id"],
        title=cfg["figure"].get("title", cfg["figure"]["id"]),
        x_var=cfg["figure"]["x_var"],
        y_metric=cfg["figure"]["y_metric"],
        ylabel=cfg["figure"]["ylabel"],
        xlabel=cfg["figure"]["xlabel"],
        x_ticks=cfg["figure"].get("x_ticks"),
        legend=bool(cfg["figure"].get("legend", True)),
        save_as=cfg["figure"].get("save_as", f"{cfg['figure']['id'].lower()}.png"),
        aggregate_runs=int(cfg["figure"].get("aggregate_runs", 1)),
        stability_guard=bool(cfg["figure"].get("stability_guard", True)),
        auto_defaults=cfg["figure"].get("auto_defaults", {}),
        asymptote=cfg["figure"].get("asymptote", {"draw": False}),
        stats=cfg["figure"].get("stats", {"ci_level": 0.95, "error_style": "bar", "save_table": True}),
        parallel_workers=int(cfg["figure"].get("parallel_workers", 0)),
        # 如果 YAML 沒有 theory 欄位，預設為 auto（會畫理論）
        theory=cfg["figure"].get("theory", {"mode": "auto"}),
    )

    sim = SimConfig(
        exe=cfg["sim"]["exe"],
        base_args=cfg["sim"].get("base_args", {}),
    )

    curves: List[CurveSpec] = []
    for i, it in enumerate(cfg.get("sweeps", [])):
        curves.append(CurveSpec(
            label=it.get("label", f"curve_{i}"),
            fixed=it["fixed"],
            x_values=it.get("x_values"),
            x_auto=it.get("x_auto"),
        ))

    fig_dir = os.path.dirname(path)
    return fig, sim, curves, fig_dir

# ------------------------- Asymptote helpers -------------------------

def determine_asymptote_x(fig: FigureSpec, curve: CurveSpec) -> Optional[float]:
    mode = None; value = None; clip_ratio = None
    if curve.x_auto and curve.x_auto.get("asymptote"):
        a = curve.x_auto["asymptote"]
        mode = a.get("mode"); value = a.get("value"); clip_ratio = a.get("clip_ratio")
    if mode is None and fig.asymptote:
        mode = fig.asymptote.get("mode"); value = fig.asymptote.get("value")
        if clip_ratio is None:
            clip_ratio = fig.auto_defaults.get("clip_ratio")

    if mode == "value" and isinstance(value, (int, float)):
        return float(value)

    if mode == "auto_stability":
        if fig.x_var == "lambda":
            mu = float(curve.fixed.get("mu"))
            e  = float(curve.fixed.get("e"))
            C  = int(curve.fixed.get("C"))
            return stability_lambda_max(mu, e, C)
        elif fig.x_var == "mu":
            lmb = float((curve.fixed.get("lambda")
                         if curve.fixed.get("lambda") is not None
                         else (fig.auto_defaults.get("lambda") if fig.auto_defaults else None)
                         or (sim.base_args.get("lambda") if 'sim' in globals() else 0.0)))
            e = float(curve.fixed.get("e"))
            return mu_min_for_stability(e, lmb)
        elif fig.x_var == "e":
            lmb = float((curve.fixed.get("lambda")
                         if curve.fixed.get("lambda") is not None
                         else (fig.auto_defaults.get("lambda") if fig.auto_defaults else None)
                         or (sim.base_args.get("lambda") if 'sim' in globals() else 0.0)))
            mu = float(curve.fixed.get("mu"))
            return e_min_for_stability(mu, lmb)
    return None

def build_auto_xs(fig: FigureSpec, curve: CurveSpec) -> Optional[List[float]]:
    if not curve.x_auto and not fig.auto_defaults:
        return None
    approach = (curve.x_auto or {}).get("approach", fig.auto_defaults.get("approach"))
    step = (curve.x_auto or {}).get("step", fig.auto_defaults.get("step", 0.2))
    clip_ratio = (curve.x_auto or {}).get("asymptote", {}).get("clip_ratio", fig.auto_defaults.get("clip_ratio", 0.995))

    refine_cfg = (curve.x_auto or {}).get("refine", fig.auto_defaults.get("refine", {})) or {}
    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_levels = int(refine_cfg.get("levels", 4))

    x_asym = determine_asymptote_x(fig, curve)
    xs: List[float] = []

    if approach == "left":
        start = fig.auto_defaults.get("common_start")
        if start is None:
            start = (curve.x_auto or {}).get("start", 0.0)
        if x_asym is not None:
            end = x_asym * float(clip_ratio)
        else:
            end = (curve.x_auto or {}).get("end", fig.auto_defaults.get("common_end"))
        if end is None:
            return None
        coarse = frange(float(start), float(end), float(step))
        xs.extend(coarse)
        if refine_enabled and x_asym is not None and refine_levels > 0:
            xs.extend(dyadic_refine_left(float(start), float(x_asym), refine_levels, float(clip_ratio)))

    elif approach == "right":
        end = fig.auto_defaults.get("common_end")
        if end is None:
            end = (curve.x_auto or {}).get("end", 0.0)
        if x_asym is not None:
            start = x_asym * (1.0 + max(1e-3, (1.0/float(clip_ratio) - 1.0) if clip_ratio else 0.005))
        else:
            start = (curve.x_auto or {}).get("start", fig.auto_defaults.get("common_start"))
        if start is None:
            return None
        coarse = frange(float(start), float(end), float(step))
        xs.extend(coarse)
        if refine_enabled and x_asym is not None and refine_levels > 0:
            xs.extend(dyadic_refine_right(float(end), float(x_asym), refine_levels, float(clip_ratio)))
    else:
        return None

    xs = sorted(set(round(x, 10) for x in xs))
    return xs

# ------------------------- Stats helpers -------------------------

def compute_stats(values: List[float], ci_level: float) -> Dict[str, float]:
    n = len(values)
    mean_val = sum(values) / n
    std_val = statistics.stdev(values) if n > 1 else 0.0
    var_val = std_val ** 2
    z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_map.get(round(ci_level, 2), 1.96)
    ci_half = z * (std_val / sqrt(n)) if n > 1 else 0.0
    return {
        "mean": mean_val,
        "var": var_val,
        "std": std_val,
        "ci_lower": mean_val - ci_half,
        "ci_upper": mean_val + ci_half,
        "ci_half": ci_half,
    }

# ------------------------- Plotting core -------------------------

def plot_figure(fig: FigureSpec, sim: SimConfig, curves: List[CurveSpec], fig_dir: str, no_theory: bool=False) -> str:
    out_dir = os.path.join(fig_dir, "out")
    ensure_dir(out_dir)

    plt.figure()
    plt.title(fig.title)
    plt.xlabel(fig.xlabel)
    plt.ylabel(fig.ylabel)

    # Auto-compute common_end for approach=="right" when requested
    if fig.auto_defaults.get("approach") == "right" and not fig.auto_defaults.get("common_end"):
        end_policy = fig.auto_defaults.get("end_policy") or {}
        if end_policy.get("mode") == "threshold_plus":
            threshold = float(end_policy.get("threshold", 10.0))
            margin = float(end_policy.get("margin", 0.5))
            probe_step = float(end_policy.get("probe_step", 0.1))

            def measure_y(xval: float, base_args: Dict[str, Any], curve_label: str) -> float:
                tmp_csv = os.path.join(out_dir, f"__probe__{curve_label.replace(' ','_')}__{fig.x_var}{xval}.csv")
                run_args = dict(base_args)
                run_args[fig.x_var] = xval
                run_sim_once(sim.exe, run_args, tmp_csv)
                try:
                    yv = read_metric_from_csv(tmp_csv, fig.y_metric)
                finally:
                    try:
                        os.remove(tmp_csv)
                    except Exception:
                        pass
                return yv

            crossings: List[float] = []
            for curve in curves:
                base_args = dict(sim.base_args)
                for k in ("mu", "e", "C", "lambda"):
                    if k in curve.fixed and curve.fixed[k] is not None:
                        base_args[k] = curve.fixed[k]
                x_asym = determine_asymptote_x(fig, curve)
                if x_asym is None:
                    continue
                start = x_asym * (1.0 + max(1e-3, (1.0/float(fig.auto_defaults.get("clip_ratio", 0.995)) - 1.0)))
                x = start
                found = None
                for _ in range(200):
                    yv = measure_y(x, base_args, curve.label)
                    if yv < threshold:
                        found = x
                        break
                    x += probe_step
                if found is not None:
                    crossings.append(found)
            if crossings:
                common_end = max(crossings) + margin
                fig.auto_defaults["common_end"] = common_end
                print(f"[auto] approach=right common_end set to {common_end:.4f} (threshold {threshold} + margin {margin})")

    # Collect stats rows when aggregate_runs>1
    stats_rows: List[Dict[str, Any]] = []
    use_error = fig.aggregate_runs > 1
    ci_level = float(fig.stats.get("ci_level", 0.95))
    error_style = fig.stats.get("error_style", "bar")

    # --- unified small marker style ---
    base_marker = dict(marker='o', markersize=4, markerfacecolor='none', markeredgewidth=0.8)

    for curve in curves:
        # Build x grid
        if curve.x_values:
            x_list = list(curve.x_values)
        else:
            x_list = build_auto_xs(fig, curve)
            if not x_list:
                print(f"[warn] No x grid for curve '{curve.label}'. Provide x_values or x_auto/auto_defaults.")
                continue

        # Per-curve vertical asymptote
        if fig.asymptote.get("draw"):
            x_asym = determine_asymptote_x(fig, curve)
            if x_asym is not None:
                plt.axvline(x_asym, linestyle="--", alpha=0.5)

        # --- run simulations for each x ---
        xs: List[float] = []
        ys_mean: List[float] = []
        ys_err: List[float] = []

        for x in x_list:
            args_map = dict(sim.base_args)
            for k in ("mu", "e", "C", "lambda"):
                if k in curve.fixed and curve.fixed[k] is not None:
                    args_map[k] = curve.fixed[k]
            args_map[fig.x_var] = x

            # Stability guard for lambda-axis
            if fig.stability_guard and fig.x_var == "lambda":
                mu = float(args_map["mu"]) if args_map.get("mu") is not None else None
                e = float(args_map["e"]) if args_map.get("e") is not None else None
                C = int(args_map["C"]) if args_map.get("C") is not None else None
                if None not in (mu, e, C):
                    lam_max = stability_lambda_max(mu, e, C)
                    if x >= 0.999 * lam_max:
                        print(f"[skip] {curve.label} x={x} >= stability bound (λ_max≈{lam_max:.4f})")
                        continue

            run_values: List[float] = []
            runs = max(1, int(fig.aggregate_runs))
            workers = int(fig.parallel_workers) if getattr(fig, "parallel_workers", 0) else min(os.cpu_count() or 1, runs)

            def _one_run(r_idx: int) -> Tuple[float, str]:
                safe_label = curve.label.replace(' ', '_').replace('=', '')
                unique = uuid.uuid4().hex[:8]
                out_csv = os.path.join(out_dir, f"{safe_label}__{fig.x_var}{x}__r{r_idx}_{unique}.csv")
                run_args = dict(args_map)
                if "seed" in run_args:
                    try:
                        run_args["seed"] = int(run_args["seed"]) + r_idx
                    except Exception:
                        pass
                run_sim_once(sim.exe, run_args, out_csv)
                val = read_metric_from_csv(out_csv, fig.y_metric)
                return val, out_csv

            if runs == 1 or workers <= 1:
                for r in range(runs):
                    v, _ = _one_run(r)
                    run_values.append(v)
            else:
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures = {ex.submit(_one_run, r): r for r in range(runs)}
                    for fut in as_completed(futures):
                        v, _ = fut.result()
                        run_values.append(v)

            if use_error:
                s = compute_stats(run_values, ci_level)
                stats_rows.append({
                    "curve": curve.label,
                    fig.x_var: x,
                    f"{fig.y_metric}_mean": s["mean"],
                    f"{fig.y_metric}_var": s["var"],
                    f"{fig.y_metric}_std": s["std"],
                    "ci_level": ci_level,
                    "ci_lower": s["ci_lower"],
                    "ci_upper": s["ci_upper"],
                })
                xs.append(x); ys_mean.append(s["mean"]); ys_err.append(s["ci_half"])
            else:
                m = sum(run_values) / len(run_values)
                xs.append(x); ys_mean.append(m); ys_err.append(0.0)

        # ---- Simulation: draw style depends on theory availability ----
        if not xs:
            print(f"[warn] No valid points for curve '{curve.label}'.")
            continue
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        xs = [xs[i] for i in order]
        ys_mean = [ys_mean[i] for i in order]
        ys_err = [ys_err[i] for i in order]

        # whether theory exists for this curve
        has_theory = False
        if _theory_mod and hasattr(_theory_mod, "get_curve"):
            try:
                params_for_theory = merged_params(sim, curve)
                has_theory = _theory_mod.get_curve(fig.y_metric, fig.x_var, params_for_theory) is not None
            except Exception as ex:
                print(f"[theory] probe error '{curve.label}': {ex}")
                has_theory = False



        # draw simulation (A) with theory: points only; (B) without theory: line+points
        if use_error:
            if error_style == "band":
                # band should not enter legend
                plt.fill_between(xs,
                                 [m - e for m, e in zip(ys_mean, ys_err)],
                                 [m + e for m, e in zip(ys_mean, ys_err)],
                                 alpha=0.15, zorder=2, label=None)
                if has_theory:
                    plt.plot(xs, ys_mean, linestyle='none', label=f"{curve.label} (sim)", zorder=3, **base_marker)
                else:
                    plt.plot(xs, ys_mean, '-', linewidth=1.2, label=f"{curve.label} (sim)", zorder=3, **base_marker)
            else:
                plt.errorbar(xs, ys_mean, yerr=ys_err,
                             fmt='o' if has_theory else '-o',
                             linestyle='none' if has_theory else '-',
                             capsize=3, markersize=base_marker["markersize"],
                             label=f"{curve.label} (sim)", zorder=3)
        else:
            if has_theory:
                plt.plot(xs, ys_mean, linestyle='none', label=f"{curve.label} (sim)", zorder=3, **base_marker)
            else:
                plt.plot(xs, ys_mean, '-', linewidth=1.2, label=f"{curve.label} (sim)", zorder=3, **base_marker)

                # ---- Theory line (auto ON unless --no-theory or theory.mode=="off") ----
        draw_theory = not no_theory
        if isinstance(fig.theory, dict) and fig.theory.get("mode", "auto") == "off":
            draw_theory = False

        if draw_theory and _theory_mod and hasattr(_theory_mod, "get_curve"):
            try:
                params_for_theory = merged_params(sim, curve)
                ask = _theory_mod.get_curve(fig.y_metric, fig.x_var, params_for_theory)
            except Exception as ex:
                ask = None
                print(f"[theory] error resolving '{curve.label}': {ex}")

            if ask:
                f_theory, _dom_unused, lsuf = ask

                # 直接用「模擬點的範圍」來畫理論線（避免 domain 過度保守）
                x_lo = min(xs); x_hi = max(xs)
                if x_hi > x_lo:
                    step = max((x_hi - x_lo) / 300.0, 1e-4)
                    x_dense = frange(x_lo, x_hi, step)
                    y_dense = [f_theory(xv) for xv in x_dense]

                    # 至少要有兩個有效點才畫
                    xy = [(xd, yd) for xd, yd in zip(x_dense, y_dense) if yd == yd and math.isfinite(yd)]
                    if len(xy) >= 2:
                        x_dense2, y_dense2 = zip(*xy)

                        # 用與模擬點相同顏色
                        tmp, = plt.plot(xs, ys_mean, linestyle='none')
                        color = tmp.get_color()
                        tmp.remove()

                        plt.plot(x_dense2, y_dense2, '-', linewidth=2, color=color,
                                 label=f"{curve.label} {lsuf}", zorder=2)
                    else:
                        print(f"[theory] '{curve.label}' has <2 finite points in span [{x_lo:.4g},{x_hi:.4g}] → skip")
            else:
                print(f"[theory] no curve for '{curve.label}' (y={fig.y_metric}, x={fig.x_var}, fixed={curve.fixed})")


    # ----- axes cosmetics and legend (deduplicated) -----
    if fig.x_ticks:
        plt.xticks(fig.x_ticks)

    if fig.legend:
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        seen = set()
        new_h, new_l = [], []
        for h, l in zip(handles, labels):
            if not l or l in seen:
                continue
            seen.add(l)
            new_h.append(h); new_l.append(l)
        ax.legend(new_h, new_l)

    plt.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(fig_dir, fig.save_as)
    plt.tight_layout(); plt.savefig(save_path, dpi=200)
    print(f"[ok] Saved: {save_path}")

    if fig.aggregate_runs > 1 and fig.stats.get("save_table", True):
        stats_csv = os.path.join(fig_dir, f"{fig.id.lower()}_stats.csv")
        if 'stats_rows' in locals() and stats_rows:
            with open(stats_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(stats_rows[0].keys()))
                w.writeheader(); w.writerows(stats_rows)
            print(f"[ok] Stats saved: {stats_csv}")

    return save_path

# ------------------------- Discovery -------------------------

def discover_all_configs() -> List[str]:
    # 支援：從 RA/ 或 RA/Result/ 甚至更深層呼叫
    here = os.getcwd()
    candidates = []

    patterns = [
        os.path.join("Result", "**", "Fig*", "config.yaml"),  # 從 RA/ 執行時有效
        os.path.join("**", "Fig*", "config.yaml"),            # 從 Result/ 執行時有效
    ]

    for pat in patterns:
        candidates.extend(glob.glob(pat, recursive=True))

    # 去重、排序
    return sorted(set(candidates))

# ------------------------- CLI -------------------------

def main():
    ap = argparse.ArgumentParser(description="Auto-run BaseStation simulator and plot figures from YAML recipes.")
    ap.add_argument("config", nargs="?", help="Path to a single config.yaml")
    ap.add_argument("--all", action="store_true", help="Run all configs under Result/**/Fig*/config.yaml")
    ap.add_argument("--no-theory", action="store_true", help="Do not draw theory lines even if available")
    args = ap.parse_args()

    cfg_paths: List[str] = []
    if args.all:
        cfg_paths = discover_all_configs()
        if not cfg_paths:
            print("[make_figure] No configs found under Result/**/Fig*/config.yaml"); return 1
    elif args.config:
        cfg_paths = [args.config]
    else:
        ap.print_help(); return 2

    for p in cfg_paths:
        fig, sim, curves, fig_dir = load_yaml(p)
        print(f"[figure] {fig.paper}/{fig.id} — {fig.title}")
        print(f"[sim] exe={sim.exe}")
        plot_figure(fig, sim, curves, fig_dir, no_theory=args.no_theory)

    return 0

if __name__ == "__main__":
    sys.exit(main())
