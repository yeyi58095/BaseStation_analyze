set datafile separator ","
set term pngcairo size 1000,700
set output "fig8.png"
set xlabel "lambda"
set ylabel "Average Packet Delay (ms)"
set grid
# 注意：summary.csv 欄位：lambda,mean,std,ci95_lo,ci95_hi,n
plot "summary.csv" using 1:2:4:5 with yerrorbars title "mean ± 95% CI", \
     "summary.csv" using 1:2 with linespoints title "mean"
