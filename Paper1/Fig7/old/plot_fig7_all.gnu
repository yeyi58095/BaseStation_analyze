# === plot_fig7_all.gnu ===
# 輸出
set terminal pngcairo size 1200,900
set output "fig7_e_vs_L_all.png"

# 座標/外觀
set xlabel "e"
set ylabel "Mean System Size"
set grid
set key right top
set tics out
set border lw 1.5
set autoscale xfix
set autoscale yfix

# 線型樣式（不同 μ）
set style line 1 lw 3 dt 1 lc rgb "#1f77b4"  # μ = 1.6
set style line 2 lw 3 dt 2 lc rgb "#ff7f0e"  # μ = 2.0
set style line 3 lw 3 dt 3 lc rgb "#2ca02c"  # μ = 2.4

# 如果想要用對數 y 軸，取消下一行註解
# set logscale y

# 繪圖（檔名依你貼的）
plot \
  "fig7_mu1.6.txt" using 1:2 with linespoints ls 1 title "{/Symbol m}=1.6", \
  "fig7_mu2.txt"   using 1:2 with linespoints ls 2 title "{/Symbol m}=2.0", \
  "fig7_mu2.4.txt" using 1:2 with linespoints ls 3 title "{/Symbol m}=2.4"

unset output
