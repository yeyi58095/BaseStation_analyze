
set term pngcairo size 1000,700
set output "/mnt/d/daniel/RA/Result/Paper1/Fig4/old/Fig4_C100.png"
set xlabel "lambda"
set ylabel "Average Packet Delay (ms)"
set grid
set title "C = 100"
plot "/mnt/d/daniel/RA/Result/Paper1/Fig4/old/fig4_C100.dat" using 1:2:3:4 with yerrorbars title "mean ± 95% CI",      "/mnt/d/daniel/RA/Result/Paper1/Fig4/old/fig4_C100.dat" using 1:2 with linespoints title "mean"
