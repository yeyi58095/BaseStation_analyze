import os, math, subprocess
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_CSV       = os.path.join(HERE, "raw_runs.csv")
SUMMARY_CSV   = os.path.join(HERE, "summary.csv")

# 指標
X_AXIS   = "lambda"
Y_METRIC = "avg_delay_ms"

def mean_ci(series, z=1.96):
    n = len(series)
    m = series.mean()
    s = series.std(ddof=1) if n > 1 else 0.0
    half = z * s / (n ** 0.5) if n > 1 else 0.0
    return m, s, (m - half), (m + half), n

def main():
    df = pd.read_csv(RAW_CSV)
    need = {"C", X_AXIS, Y_METRIC}
    if not need.issubset(df.columns):
        raise RuntimeError(f"CSV missing columns: {need - set(df.columns)}")

    rows = []
    for (Cval, xval), g in df.groupby(["C", X_AXIS]):
        m, s, lo, hi, n = mean_ci(g[Y_METRIC])
        rows.append({"C": Cval, X_AXIS: xval, "mean": m, "std": s, "ci95_lo": lo, "ci95_hi": hi, "n": n})
    summary = pd.DataFrame(rows).sort_values(["C", X_AXIS])
    summary.to_csv(SUMMARY_CSV, index=False)
    print("Wrote:", SUMMARY_CSV)

    # 依 C 分別畫圖
    for Cval, sub in summary.groupby("C"):
        dat_path = os.path.join(HERE, f"fig4_C{Cval}.dat")
        gnu_path = os.path.join(HERE, f"plot_fig4_C{Cval}.gnu")
        out_png  = os.path.join(HERE, f"Fig4_C{Cval}.png")

        with open(dat_path, "w") as f:
            f.write(f"# {X_AXIS} mean ci95_lo ci95_hi std n\n")
            for _, r in sub.iterrows():
                f.write(f"{r[X_AXIS]} {r['mean']} {r['ci95_lo']} {r['ci95_hi']} {r['std']} {int(r['n'])}\n")

        with open(gnu_path, "w") as g:
            g.write(f"""
set term pngcairo size 1000,700
set output "{out_png}"
set xlabel "{X_AXIS}"
set ylabel "Average Packet Delay (ms)"
set grid
set title "C = {Cval}"
plot "{dat_path}" using 1:2:3:4 with yerrorbars title "mean ± 95% CI", \
     "{dat_path}" using 1:2 with linespoints title "mean"
""")

        try:
            subprocess.run(["gnuplot", gnu_path], check=True)
            print("Plot saved:", out_png)
        except Exception:
            print("gnuplot not run. Run manually:", gnu_path)

if __name__ == "__main__":
    main()
