#!/usr/bin/python2.7
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

filename = 'figure1'
markersize = 5

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


psi = np.loadtxt("r1001.out")

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)
plt.subplot(221)

plt.plot(psi[:, 0], psi[:, 1], 'ro', label =  "ground", markersize = markersize) 

plt.xlabel(r"""$x$""", fontsize = 25)
plt.xticks(fontsize = 25)
#plt.xticks([2.0, 2.5, 3.0])
plt.ylabel(r"""$\psi(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'upper right')
set_legend_fontsize(legend)
plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

plt.subplot(222)

plt.plot(psi[:, 0], psi[:, 2], 'bv', label =  "the 1st", markersize = markersize) 

plt.xlabel(r"""$x$""", fontsize = 25)
plt.xticks(fontsize = 25)
#plt.xticks([2.3, 2.7, 3.1, 3.5])
plt.ylabel(r"""$\psi(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'upper right')
set_legend_fontsize(legend)
plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))


plt.subplot(223)
plt.plot(psi[:, 0], psi[:, 3], 'g^', label =  "the 2nd", markersize = markersize) 

plt.xlabel(r"""$x$""", fontsize = 25)
plt.xticks(fontsize = 25)
#plt.xticks([2.0, 2.5, 3.0])
plt.ylabel(r"""$\psi(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'upper right')
set_legend_fontsize(legend)
plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))


plt.subplot(224)
plt.plot(psi[:, 0], psi[:, 4], 'c<', label =  "the 3rd", markersize = markersize) 

plt.xlabel(r"""$x$""", fontsize = 25)
plt.xticks(fontsize = 25)
#plt.xticks([2.3, 2.7, 3.1, 3.5])
plt.ylabel(r"""$\psi(x)$""", fontsize = 25)
plt.yticks(fontsize = 25)
#plt.yticks([0, 5, 10, 15, 20])

legend = plt.legend(loc = 'upper right')
set_legend_fontsize(legend)
plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))


plt.subplots_adjust(wspace=0.5, hspace=0)

for i, label in enumerate(('(a)', '(b)', '(c)', '(d)')):
    ax = fig.add_subplot(2, 2, i+1)
    ax.text(-0.2, 1.15, label, transform = ax.transAxes, fontsize = 25, va = 'top', ha = 'right')



plt.show()

"""
filename_ps = filename + '.ps'
plt.savefig(filename_ps)

cmd = "ps2eps -f -d %s"%filename_ps

os.system(cmd)

filename_eps = filename + '.eps'

cmd = 'epstopdf %s'%filename_eps

os.system(cmd)
"""
