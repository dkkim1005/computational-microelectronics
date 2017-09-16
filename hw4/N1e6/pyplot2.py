#!/usr/bin/python2.7
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

x_elec = np.loadtxt("x-elec.dat")
x_hole = np.loadtxt("x-hole.dat")

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.plot(x_elec [:, 0], x_elec [:, 1], 'ro-', label =  "" , markersize = 3) 
plt.plot(x_hole [:, 0], x_hole [:, 1], 'bo-', label =  "" , markersize = 3) 

plt.xlabel(r"""$x[{\mu}m]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 2, 5))

plt.ylabel(r"""$n[cm^{-3}]$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

#legend = plt.legend(loc = 'best')
#set_legend_fontsize(legend)

plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.show()
