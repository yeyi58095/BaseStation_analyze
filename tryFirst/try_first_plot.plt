# plot_test.plt
set terminal pngcairo size 800,600 enhanced font 'Arial,12'
set output 'sin_plot.png'
set title 'Test Plot from Gnuplot'
set xlabel 'x'
set ylabel 'sin(x)'
set grid
plot sin(x) title 'sin(x)' with lines linewidth 2 linecolor rgb '#AA00FF'
set output
pause -1 "Press Enter to exit"
