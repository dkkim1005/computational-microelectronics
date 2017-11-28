#!/usr/bin/python2.7
import numpy as np
import os

V = np.linspace(-0.2, 0.2, 101)

os.chdir('raw-data')

for i, v in enumerate(V):
    cmd = '../sch-poisson %s %s' %(str(v), '1')
    os.system(cmd)
