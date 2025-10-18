# 輸出檔
set terminal pngcairo size 1200,900
set output "fig4_lambda_vs_L_all.png"

# 座標/外觀
set xlabel "{/Symbol l}"
set ylabel "Mean System Size"
set grid
set key left top
set tics out
set border lw 1.5
set autoscale xfix
set autoscale yfix

# 線型（不同線條樣式便於分辨）
set style line 1 lw 3 dt 1      # C=1  實線
set style line 2 lw 3 dt 2      # C=5  虛線
set style line 3 lw 3 dt 3      # C=10 點線

# 直接畫三條線；第二欄就是要畫的值
plot \
  "simDPwithC_mu3_e2.4_C1.txt" using 1:2 with lines ls 1 title "C = 1", \
  "fig4_c5.txt"                using 1:2 with lines ls 2 title "C = 5", \
  "fig4_C10.txt"               using 1:2 with lines ls 3 title "C = 10"

unset output
