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

plt.subplot(211)

plt.plot(wave [:, 0], wave [:, 60], 'r-', label =  "ground" , markersize = 0.1) 
plt.plot(wave [:, 0], wave [:, 55], 'b-', label =  "the 5'th excited" , markersize = 0.1) 
plt.plot(wave [:, 0], wave [:, 50], 'g-', label =  "the 10'th excited" , markersize = 0.1) 
plt.plot(wave [:, 0], wave [:, 40], 'm-', label =  "the 15'th excited" , markersize = 0.1) 

plt.xlabel(r"""$x[{\mu}m]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 1, 5))

plt.ylabel(r"""$|\psi(x)|^{2}$""", fontsize = 25)
plt.yticks([], [])


legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

#plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))


Vx = np.loadtxt(sys.argv[2])
Vx[:, 1] = -Vx[:, 1] + 0.56
plt.subplot(212)

plt.plot(Vx [:, 0], Vx [:, 1], 'r-')
plt.xlabel(r"""$x[{\mu}m]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 1, 5))

plt.ylabel(r"""$V(x)[\mathrm{ev}]$""", fontsize = 25)
plt.yticks(fontsize = 25)

#plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

for i, label in enumerate(('(a)', '(b)')):
    ax = fig.add_subplot(2, 1, i+1)
    ax.text(-0.1, 1.15, label, transform = ax.transAxes, fontsize = 25, va = 'top', ha = 'right')


plt.show()
