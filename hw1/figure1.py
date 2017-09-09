#!/usr/bin/python2.7
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


psi_exact = np.loadtxt("exact.dat")
psi_r11 = np.loadtxt("r11.out")
psi_r101 = np.loadtxt("r101.out")
psi_r501 = np.loadtxt("r501.out")
psi_r1001 = np.loadtxt("r1001.out")

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.plot(psi_exact[:, 0], psi_exact[:, 1], 'md-', label =  "exact", markersize = 10) 
plt.plot(psi_r1001[:, 0], psi_r1001[:, 1], 'c<-', label =  "n=1001", markersize = 8) 
plt.plot(psi_r501[:, 0],  psi_r501[:, 1],  'g^-', label =  "n=501",  markersize = 7) 
plt.plot(psi_r101[:, 0],  psi_r101[:, 1],  'bv-', label =  "n=101",  markersize = 6) 
plt.plot(psi_r11[:, 0],   psi_r11[:, 1],   'ro-', label =  "n=11",   markersize = 5) 

plt.xlabel(r"""$x$(m)""", fontsize = 25)
plt.xticks(fontsize = 25)
#plt.xticks([2.0, 2.5, 3.0])

plt.ylabel(r"""$\psi(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.show()
