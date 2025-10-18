# === plot_fig8_all.gnu ===
# 輸出
set terminal pngcairo size 1200,900
set output "fig8_lambda_vs_delay_all.png"

# 座標與外觀
set xlabel "{/Symbol l}"
set ylabel "Average Packet Delay (ms)"
set grid
set key left top
set tics out
set border lw 1.5
set autoscale xfix
set autoscale yfix

# 線型樣式（不同 μ,e）
set style line 1 lw 3 dt 1 lc rgb "#1f77b4"  # μ=2, e=2.4
set style line 2 lw 3 dt 2 lc rgb "#ff7f0e"  # μ=3, e=2.4
set style line 3 lw 3 dt 3 lc rgb "#2ca02c"  # μ=3, e=4

# 若要看 delay 爆增區明顯，可開啟對數 y 軸
# set logscale y

# 繪圖
plot \
  "fig8_mu2_e2.4.txt" using 1:2 with linespoints ls 1 title "{/Symbol m}=2, e=2.4", \
  "fig8_mu3_e2.4.txt" using 1:2 with linespoints ls 2 title "{/Symbol m}=3, e=2.4", \
  "fig8_mu3_e4.txt"   using 1:2 with linespoints ls 3 title "{/Symbol m}=3, e=4"

unset output
