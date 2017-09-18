#!/usr/bin/python2.7
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def set_legend_fontsize(legend):
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize(25)

    for label in legend.get_lines():
        label.set_linewidth(5.0)  # the legend line width

wave = np.loadtxt(sys.argv[1])

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.plot(wave [:, 0], wave [:, 1], 'ro-', label =  "ground" , markersize = 3) 
plt.plot(wave [:, 0], wave [:, 10], 'bo-', label =  "the 10'th excited" , markersize = 3) 
plt.plot(wave [:, 0], wave [:, 20], 'go-', label =  "the 20'th excited" , markersize = 3) 
plt.plot(wave [:, 0], wave [:, 50], 'mo-', label =  "the 50'th excited" , markersize = 3) 

plt.xlabel(r"""$x[{\mu}m]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 5, 5))

plt.ylabel(r"""$\psi(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.show()
