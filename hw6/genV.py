#!/usr/bin/python2.7
import numpy as np

a = np.linspace(0, 1, 1001)
x = a**1.2
v = 0*x
data = np.zeros([len(a), 2])
data[:, 0] = x; data[:, 1] = v

np.savetxt('v-x.dat', data)
