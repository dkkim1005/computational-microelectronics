#!/usr/bin/python2.7
import numpy as np

IMGPOLE = 1e-30j

def f(x):
    return ((x+IMGPOLE)/(np.exp(x+IMGPOLE) - 1.)).real

def df(x):
    return ((np.exp(x+IMGPOLE) - 1. - (x+IMGPOLE)*np.exp(x+IMGPOLE))/
            (np.exp(x+IMGPOLE) - 1.)**2).real


if __name__ == "__main__":
    x = np.linspace(-3, 3, 101)
    y = f(x)

    data = np.zeros([len(x), 2])
    data[:, 0] = x; data[:, 1] = y

    np.savetxt("bern.dat", data)

    y = df(x)

    data = np.zeros([len(x), 2])
    data[:, 0] = x; data[:, 1] = y

    np.savetxt("dbern.dat-%s"%IMGPOLE, data)
