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

phi_E = np.loadtxt("E-field.dat")

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.plot(phi_E [:, 0], phi_E [:, 1], 'ro-', label =  "analytic" , markersize = 3) 
plt.plot(phi_E [:, 0], phi_E [:, 2], 'bo-', label =  "numerical" , markersize = 3) 

plt.xlabel(r"""$q_{0}\phi[ev]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 1, 5))

plt.ylabel(r"""$E_{s}[N/C]$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.show()
