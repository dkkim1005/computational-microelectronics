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


phi_E5e5 = np.loadtxt("E-field-d5em5")
phi_E1e4 = np.loadtxt("E-field-d1em4")
phi_E2e4 = np.loadtxt("E-field-d2em4")
phi_E5e4 = np.loadtxt("E-field-d5em4")
phi_E1e3 = np.loadtxt("E-field-d1em3")

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.plot(phi_E1e3 [:, 0], phi_E1e3 [:, 1], 'ro-', label =  "analytic(exact)" , markersize = 3) 
plt.plot(phi_E1e3 [:, 0], phi_E1e3 [:, 2], 'bo-', label =  r"""$\Delta x=10^{-3}$""" , markersize = 3) 
plt.plot(phi_E5e4 [:, 0], phi_E5e4 [:, 2], 'go-', label =  r"""$\Delta x=5x10^{-4}$""" , markersize = 3) 
plt.plot(phi_E2e4 [:, 0], phi_E2e4 [:, 2], 'co-', label =  r"""$\Delta x=2x10^{-4}$""" , markersize = 3) 
plt.plot(phi_E1e4 [:, 0], phi_E1e4 [:, 2], 'mo-', label =  r"""$\Delta x=1x10^{-4}$""" , markersize = 3) 
plt.plot(phi_E5e5 [:, 0], phi_E5e5 [:, 2], 'y^-', label =  r"""$\Delta x=5x10^{-5}$""" , markersize = 3) 

plt.xlabel(r"""$q_{0}\phi_{s}[ev]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0.1, 0.5, 5))

plt.ylabel(r"""$E_{s}[N/C]$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.show()
