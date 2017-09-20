#!/usr/bin/python2.7
import sys
import numpy as np
import os

eps = 11.68*8.854187817e-12 # silicon permittivity [C^2/(J*m)]
KbT = 300*1.3806488e-23 # boltzmann constant [J/K] * room temperature(300K)
KbT_ev = 0.025851984732130292
q0 = 1.6021766208e-19
Nam = 1e15*1e6 # [# m^-3]
n_i = 1.5e16

E = lambda phis : np.sqrt(2*eps*KbT*Nam)*\
                  np.sqrt( (np.exp(-q0*phis/KbT_ev) + q0*phis/KbT_ev - 1) +\
                  (n_i/Nam)**2*(np.exp(q0*phis/KbT_ev) - q0*phis/KbT_ev - 1))/eps

qphis = np.linspace(0.1, 0.5, 50)
data = np.zeros([len(qphis), 3], dtype = 'float64')

for i, qphi in enumerate(qphis):
    data[i, 0] = qphi # phi [J/C]
    data[i, 1] = E((qphi-(-0.28715))/q0) # E[N/C]

    cmd = './chw5 %s 1'%(str(qphi))
    os.system(cmd)
    rdata = np.loadtxt("x-qphi-%s.dat"%str(qphi))
    data[i, 2] = -(rdata[1, 1] - rdata[0, 1])/((rdata[1, 0] - rdata[0, 0])*1e-6)

np.savetxt('E-field-new1.dat', data)
