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

c_density    = np.loadtxt(sys.argv[1])
semi_density = np.loadtxt(sys.argv[2])

print "classical(density):",    sys.argv[1]
print "semi_density(density):", sys.argv[2]

# --: dashes,  s: squares  ^: triangles

fig = plt.figure(1)

plt.subplot(121)

plt.plot(c_density    [:, 0], c_density    [:, 1], 'ro-', label =  "classical"     , markersize = 0.1) 
plt.plot(semi_density [:, 0], semi_density [:, 1], 'b^-', label =  "semiclassical" , markersize = 0.1) 

plt.xlabel(r"""$x[{\mu}m]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 1, 5))

plt.ylabel(r"""$n(x)[\mathrm{cm}^{-3}]$""", fontsize = 25)
plt.yticks(fontsize = 25)

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)


c_phi    = np.loadtxt(sys.argv[3])
semi_phi = np.loadtxt(sys.argv[4])

print "classical(q0phi):",   sys.argv[3]
print "semi_density(q0phi)", sys.argv[4]

plt.subplot(122)

plt.plot(c_phi    [:, 0], c_phi    [:, 1], 'ro-', label =  "classical"     , markersize = 2) 
plt.plot(semi_phi [:, 0], semi_phi [:, 1], 'b^-', label =  "semiclassical" , markersize = 0.1) 

plt.xlabel(r"""$x[{\mu}m]$""", fontsize = 25)
plt.xticks(fontsize = 25)
plt.xticks(np.linspace(0, 1, 5))

plt.ylabel(r"""$q\phi(x)[\mathrm{ev}]$""", fontsize = 25)
plt.yticks(fontsize = 25)

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)


#plt.ticklabel_format(style='sci', axis = 'both', scilimits=(0, 0))

for i, label in enumerate(('(a)', '(b)')):
    ax = fig.add_subplot(1, 2, i+1)
    ax.text(-0.1, 1.1, label, transform = ax.transAxes, fontsize = 25, va = 'top', ha = 'right')
plt.show()
