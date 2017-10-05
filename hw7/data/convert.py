#!/usr/bin/python2.7
import numpy as np
import sys

filename = sys.argv[1]
psi = np.loadtxt(filename)
KbT = 0.025851984732130292 # [ev]

x = psi[:, 0]
y = psi[:, 1]*KbT

data = np.zeros([len(x), 2])
data[:, 0] = x
data[:, 1] = y

np.savetxt(filename, data)
