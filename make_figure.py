#!/usr/bin/env python3
"""
make_figure.py — Batch simulator + plotter for WPCN paper figures (Fig.4~Fig.11)

Directory layout (relative to repo root `RA/`):
- Projects/BaseStation/Win32/Debug/Project2.exe  # simulator (already built)
- Result/
    make_figure.py                                # this file
    PaperX/
      FigY/
        config.yaml                               # figure recipe
        out/                                      # auto-created, raw CSVs
        figY.png                                  # auto-generated plot

Usage:
    # If you run from RA/Result, set sim.exe to "../Projects/.../Project2.exe"
    python make_figure.py Result/Paper1/Fig4/config.yaml
    
    # Or run from RA/ with original paths
    python Result/make_figure.py Result/Paper1/Fig4/config.yaml

    # Run all configs under Result/**/Fig*/config.yaml
    python Result/make_figure.py --all

YAML schema (key additions for optional vertical asymptotes & auto ranges):
---
figure:
  paper: "Paper1"
  id: "Fig4"
  title: "Average number of DPs vs λ (μ=3, e=2.4)"
  x_var: "lambda"              # one of: lambda, mu, e
  y_metric: "L"                # CSV columns: L, W, avg_delay_ms, loss_rate, EP_mean, ...
  ylabel: "Average number of DPs (L_D)"
  xlabel: "λ"
  legend: true
  save_as: "fig4.png"
  aggregate_runs: 1
  stability_guard: true        # keeps legacy protection
  # --- NEW: figure-level defaults for auto range toward a vertical asymptote ---
  auto_defaults:
    step: 0.2                  # spacing for auto-generated x grid
    approach: "left"           # "left" = approach asymptote from smaller x; "right" from larger x
    common_start: 0.4          # if approach==left, all curves share this start (optional)
    common_end: 2.2            # if approach==right, all curves share this end (optional)
    clip_ratio: 0.995          # how close to asymptote (0.995 = 99.5% of x_asym) for left; 1.005 for right
  # --- NEW: vertical asymptote rendering (optional) ---
  asymptote:
    draw: true                 # show dashed vertical lines
    mode: "auto_stability"     # "auto_stability" (only when x_var==lambda) or "value"
    value: null                # required when mode=="value"

sim:
  exe: "Projects/BaseStation/Win32/Debug/Project2.exe"
  base_args: { T: 1000010, seed: 12345, N: 1, rtx: 1, slots: 0, alwaysCharge: true, version: "BaseStation" }

sweeps:
  - label: "C=1"
    fixed: { mu: 3.0, e: 2.4, C: 1 }
    # Use EITHER x_values OR x_auto. If both are present, x_values wins.
    # x_values: [0.4, 0.6, 0.8, 1.0, 1.2, 1.3]
    x_auto:
      approach: "left"         # optional override of figure.auto_defaults.approach
      step: 0.2                # optional override of figure.auto_defaults.step
      asymptote:
        mode: "auto_stability" # or "value"
        value: null            # set when mode=="value"
        clip_ratio: 0.995      # optional override of figure.auto_defaults.clip_ratio
  - label: "C=5"
    fixed: { mu: 3.0, e: 2.4, C: 5 }
    x_auto: { }
...

Notes:
- Stability guard uses λ_max from inequality (10): λ < μ * (1 - (μ/ε)^C) / (1 - (μ/ε)^(C+1)).
- NEW: You can let the script auto-generate x grids that stop ε-close to a vertical asymptote, and also draw dashed vertical lines per curve. This matches your request: curves start from a common start and approach their own asymptote (left), or start near their own asymptote and extend to a common end (right).
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
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple, Optional

try:
    import yaml  # PyYAML
except Exception as e:
    print("[make_figure] Missing dependency: PyYAML (pip install pyyaml)")
    raise

import matplotlib.pyplot as plt

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


def frange(start: float, end: float, step: float) -> List[float]:
    xs = []
    x = start
    # include end with small epsilon tolerance depending on direction
    if step == 0:
        return xs
    if start <= end:
        while x <= end + 1e-12:
            xs.append(round(x, 10))
            x += step
    else:
        while x >= end - 1e-12:
            xs.append(round(x, 10))
            x -= abs(step)
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

        # 支援虛擬 metric：例如 EP_short = 1 - EP_mean
        if metric == "EP_short":
            base_val = rows[-1].get("EP_mean")
            if base_val is None:
                raise KeyError(f"Base metric 'EP_mean' not found in header of {csv_path}")
            return 1.0 - float(base_val)

        # 一般情況：直接讀取對應欄位
        val = rows[-1].get(metric)
        if val is None:
            raise KeyError(f"Metric '{metric}' not found in header of {csv_path}")
        return float(val)



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


@dataclass
class SimConfig:
    exe: str
    base_args: Dict[str, Any]


# ------------------------- Core Runner -------------------------

def dyadic_refine_left(start: float, x_asym: float, levels: int, terminal_clip: float) -> List[float]:
    """Generate points approaching x_asym from the left with dyadic spacing.
       Distance sequence d_k = d0 / 2^k until k=levels, then clip by terminal_clip (e.g., 0.995 of asym).
    """
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
    """Generate points approaching x_asym from the right with dyadic spacing.
       Distance sequence d_k = d0 / 2^k until k=levels, then clip by terminal_clip (e.g., 1.005 of asym).
    """
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
    )
    sim = SimConfig(
        exe=cfg["sim"]["exe"],
        base_args=cfg["sim"].get("base_args", {}),
    )
    curves = []
    for i, it in enumerate(cfg.get("sweeps", [])):
        curves.append(CurveSpec(
            label=it.get("label", f"curve_{i}"),
            fixed=it["fixed"],
            x_values=it.get("x_values"),
            x_auto=it.get("x_auto"),
        ))
    fig_dir = os.path.dirname(path)
    return fig, sim, curves, fig_dir


def determine_asymptote_x(fig: FigureSpec, curve: CurveSpec) -> Optional[float]:
    # Per-curve x_auto has highest priority
    mode = None
    value = None
    clip_ratio = None
    if curve.x_auto and curve.x_auto.get("asymptote"):
        a = curve.x_auto["asymptote"]
        mode = a.get("mode")
        value = a.get("value")
        clip_ratio = a.get("clip_ratio")
    if mode is None and fig.asymptote:
        mode = fig.asymptote.get("mode")
        value = fig.asymptote.get("value")
        if clip_ratio is None:
            clip_ratio = fig.auto_defaults.get("clip_ratio")
    # Compute if requested
    if mode == "value" and isinstance(value, (int, float)):
        return float(value)
    if mode == "auto_stability" and fig.x_var == "lambda":
        mu = float(curve.fixed.get("mu"))
        e = float(curve.fixed.get("e"))
        C = int(curve.fixed.get("C"))
        return stability_lambda_max(mu, e, C)
    return None


def build_auto_xs(fig: FigureSpec, curve: CurveSpec) -> Optional[List[float]]:
    if not curve.x_auto and not fig.auto_defaults:
        return None
    approach = (curve.x_auto or {}).get("approach", fig.auto_defaults.get("approach"))
    step = (curve.x_auto or {}).get("step", fig.auto_defaults.get("step", 0.2))
    clip_ratio = (curve.x_auto or {}).get("asymptote", {}).get("clip_ratio", fig.auto_defaults.get("clip_ratio", 0.995))

    # NEW: optional dyadic refinement near asymptote
    refine_cfg = (curve.x_auto or {}).get("refine", fig.auto_defaults.get("refine", {})) or {}
    refine_enabled = bool(refine_cfg.get("enabled", False))
    refine_levels = int(refine_cfg.get("levels", 4))  # 4 levels → final distance d0/16

    x_asym = determine_asymptote_x(fig, curve)
    xs: List[float] = []

    if approach == "left":
        start = fig.auto_defaults.get("common_start")
        if start is None:
            start = (curve.x_auto or {}).get("start", 0.0)
        end = None
        if x_asym is not None:
            end = x_asym * float(clip_ratio)
        else:
            end = (curve.x_auto or {}).get("end", fig.auto_defaults.get("common_end"))
        if end is None:
            return None
        # coarse grid up to clipped end
        coarse = frange(float(start), float(end), float(step))
        xs.extend(coarse)
        # dyadic refinement approaching asymptote
        if refine_enabled and x_asym is not None and refine_levels > 0:
            xs.extend(dyadic_refine_left(float(start), float(x_asym), refine_levels, float(clip_ratio)))
    elif approach == "right":
        end = fig.auto_defaults.get("common_end")
        if end is None:
            end = (curve.x_auto or {}).get("end", 0.0)
        start = None
        if x_asym is not None:
            # approach from right: just beyond the asymptote
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

    # unique + sorted
    xs = sorted(set(round(x, 10) for x in xs))
    return xs


def aggregate_runs(csv_paths: List[str], metric: str) -> float:
    vals = [read_metric_from_csv(p, metric) for p in csv_paths]
    return sum(vals) / len(vals)


def plot_figure(fig: FigureSpec, sim: SimConfig, curves: List[CurveSpec], fig_dir: str) -> str:
    out_dir = os.path.join(fig_dir, "out")
    ensure_dir(out_dir)

    plt.figure()
    plt.title(fig.title)
    plt.xlabel(fig.xlabel)
    plt.ylabel(fig.ylabel)

    for curve in curves:
        xs: List[float] = []
        ys: List[float] = []

        # Decide x grid
        if curve.x_values:
            x_list = list(curve.x_values)
        else:
            x_list = build_auto_xs(fig, curve)
            if not x_list:
                print(f"[warn] No x grid for curve '{curve.label}'. Provide x_values or x_auto/auto_defaults.")
                continue

        # draw per-curve vertical asymptote if requested
        if fig.asymptote.get("draw"):
            x_asym = determine_asymptote_x(fig, curve)
            if x_asym is not None:
                plt.axvline(x_asym, linestyle="--", alpha=0.5)

        for x in x_list:
            args_map = dict(sim.base_args)
            for k in ("mu", "e", "C", "lambda"):
                if k in curve.fixed and curve.fixed[k] is not None:
                    args_map[k] = curve.fixed[k]
            args_map[fig.x_var] = x


            # Legacy stability guard (still useful when user supplies x_values)
            if fig.stability_guard and fig.x_var == "lambda":
                mu = float(args_map["mu"]) if args_map.get("mu") is not None else None
                e = float(args_map["e"]) if args_map.get("e") is not None else None
                C = int(args_map["C"]) if args_map.get("C") is not None else None
                if None not in (mu, e, C):
                    lam_max = stability_lambda_max(mu, e, C)
                    if x >= 0.999 * lam_max:
                        print(f"[skip] {curve.label} x={x} >= stability bound (λ_max≈{lam_max:.4f})")
                        continue

            # Multiple runs
            run_csvs = []
            for r in range(max(1, fig.aggregate_runs)):
                stamp = f"{int(time.time()*1000)%1_000_000:06d}"
                safe_label = curve.label.replace(' ', '_').replace('=', '')
                out_csv = os.path.join(out_dir, f"{safe_label}__{fig.x_var}{x}__r{r}_{stamp}.csv")
                run_args = dict(args_map)
                if "seed" in run_args:
                    try:
                        run_args["seed"] = int(run_args["seed"]) + r
                    except Exception:
                        pass
                run_sim_once(sim.exe, run_args, out_csv)
                run_csvs.append(out_csv)

            y = aggregate_runs(run_csvs, fig.y_metric)
            xs.append(x)
            ys.append(y)

        if not xs:
            print(f"[warn] No valid points for curve '{curve.label}'.")
            continue
        xs, ys = zip(*sorted(zip(xs, ys)))
        plt.plot(list(xs), list(ys), marker="o", label=curve.label)

    if fig.x_ticks:
        plt.xticks(fig.x_ticks)
    if fig.legend:
        plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)

    save_path = os.path.join(fig_dir, fig.save_as)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print(f"[ok] Saved: {save_path}")
    return save_path


# ------------------------- CLI -------------------------

def discover_all_configs() -> List[str]:
    return sorted(glob.glob(os.path.join("Result", "**", "Fig*", "config.yaml"), recursive=True))


def main():
    ap = argparse.ArgumentParser(description="Auto-run BaseStation simulator and plot figures from YAML recipes.")
    ap.add_argument("config", nargs="?", help="Path to a single config.yaml")
    ap.add_argument("--all", action="store_true", help="Run all configs under Result/**/Fig*/config.yaml")
    args = ap.parse_args()

    cfg_paths: List[str] = []
    if args.all:
        cfg_paths = discover_all_configs()
        if not cfg_paths:
            print("[make_figure] No configs found under Result/**/Fig*/config.yaml")
            return 1
    elif args.config:
        cfg_paths = [args.config]
    else:
        ap.print_help(); return 2

    for p in cfg_paths:
        fig, sim, curves, fig_dir = load_yaml(p)
        print(f"\n[figure] {fig.paper}/{fig.id} — {fig.title}")
        print(f"[sim] exe={sim.exe}")
        plot_figure(fig, sim, curves, fig_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
