# === fig5_lambda_vs_L_all.gnu ===
# 輸出檔
set terminal pngcairo size 1200,900
set output "fig5_lambda_vs_L_all.png"

# 座標與外觀設定
set xlabel "{/Symbol l}"
set ylabel "Mean System Size"
set grid
set key left top
set tics out
set border lw 1.5
set autoscale xfix
set autoscale yfix

# 線型樣式（不同 μ 與 e）
set style line 1 lw 3 dt 1 lc rgb "#1f77b4"  # μ=2, e=2.4
set style line 2 lw 3 dt 2 lc rgb "#ff7f0e"  # μ=3, e=2.4
set style line 3 lw 3 dt 3 lc rgb "#2ca02c"  # μ=3, e=4.0

# 繪圖
plot \
  "fig5_mu2_e2.4.txt" using 1:2 with linespoints ls 1 title "{/Symbol m}=2, e=2.4", \
  "fig5_mu3_e2.4.txt" using 1:2 with linespoints ls 2 title "{/Symbol m}=3, e=2.4", \
  "fig5_mu3_e4.txt"   using 1:2 with linespoints ls 3 title "{/Symbol m}=3, e=4.0"

unset output
