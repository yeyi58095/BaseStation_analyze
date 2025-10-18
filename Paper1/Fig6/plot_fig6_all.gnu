# === fig6_mu_vs_L_all.gnu ===
# 輸出設定
set terminal pngcairo size 1200,900
set output "fig6_mu_vs_L_all.png"

# 座標與外觀
set xlabel "{/Symbol m}"
set ylabel "Mean System Size"
set grid
set key right top
set tics out
set border lw 1.5
set autoscale xfix
set autoscale yfix

# 線條樣式（不同 e）
set style line 1 lw 3 dt 1 lc rgb "#1f77b4"  # e=1.6
set style line 2 lw 3 dt 2 lc rgb "#ff7f0e"  # e=2.4
set style line 3 lw 3 dt 3 lc rgb "#2ca02c"  # e=3.0

# 繪圖
plot \
  "fig6_e1.6.txt" using 1:2 with linespoints ls 1 title "e = 1.6", \
  "fig6_e2.txt"   using 1:2 with linespoints ls 2 title "e = 2.4", \
  "fig6_e3.txt"   using 1:2 with linespoints ls 3 title "e = 3.0"

unset output
