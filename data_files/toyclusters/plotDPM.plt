# start value for H
h1 = 117/360.0
# end value for H
h2 = 1
# creating the palette by specifying H,S,V
set palette model HSV functions (1-gray)*(h2-h1)+h1,1,1

set terminal png
set output "plot.png"

# plot idx.dat
plot "./dpmIDX.dat" with points palette, "./dpmMU.out" pt 6 ps 3 lc rgb 'black'

