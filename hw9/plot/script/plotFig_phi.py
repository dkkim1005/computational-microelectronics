import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def set_legend_fontsize(legend):
    # Set the fontsize
    for label in legend.get_texts():
        label.set_fontsize(25)

    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width


def division(vec, ratio):
    totsize = len(vec)
    reducedSize = int(totsize*ratio)
    step = int(1./ratio)
    data = []

    for i in range(reducedSize):
        data.append(vec[i*step])

    return np.array(data)


fig = plt.figure(1)

result    = np.loadtxt("../raw-data/result.dat")
bench_phi = np.loadtxt("../raw-data/x-phi.dat")

markersize = 10

plt.xlabel(r"""$x [\mu m]$""", fontsize = 30)
plt.xticks(fontsize = 30)
#plt.xticks(np.arange(2,3.501,5e-1))

plt.ylabel(r"""$\frac{\phi(x)}{V_{T}}$""", fontsize = 30)
plt.yticks(fontsize = 30)
#plt.yticks([0.01, 0.03, 0.06, 0.08])

x = division(result[:, 0], 0.1)
ddy = division(result[:, 1], 0.1)
poy = division(bench_phi[:, 1], 0.1)

plt.plot(x, ddy, 'ro-', label =  "drift-difusion", markersize = markersize, markerfacecolor = 'None') 
plt.plot(x, poy, 'bs-', label =  "poisson", markersize = markersize, markerfacecolor = 'None') 

legend = plt.legend(loc = 'best')
set_legend_fontsize(legend)

plt.show()
