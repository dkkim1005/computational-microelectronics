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

def set_xlim(xmin, xmax):
    axes = plt.gca()
    axes.set_xlim([xmin, xmax])

def set_ylim(ymin, ymax):
    axes = plt.gca()
    axes.set_ylim([ymin, ymax])

pi_N_100 = np.loadtxt(sys.argv[1])
pi_N_1000 = np.loadtxt(sys.argv[2])

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.plot(pi_N_100 [:, 0], pi_N_100 [:, 1], 'rs-', label =  "n=100" , markersize = 10) 
plt.plot(pi_N_1000[:, 0], pi_N_1000[:, 1], 'bo-', label =  "n=1001", markersize = 5) 

plt.xlabel(r"""$x$(m)""", fontsize = 25)
plt.xticks(fontsize = 25)
#plt.xticks([2.0, 2.5, 3.0])

plt.ylabel(r"""$V(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.show()
