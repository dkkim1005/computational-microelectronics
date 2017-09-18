#!/usr/bin/python2.7
import numpy as np

eps = 11.68*8.854187817e-12 # silicon permittivity [C^2/(J*m)]
KbT = 300*1.3806488e-23 # boltzmann constant [J/K] * room temperature(300K)
Nam = 1e15*1e6 # [# m^-3]
ev = 1.6021766208e-19
phis = 0.3 # [ev]
n_i = 1.5e16

print "a0:", ev*phis/KbT


E = np.sqrt(2*eps*KbT*Nam)*np.sqrt( (np.exp(-ev*phis/KbT) + ev*phis/KbT - 1) +\
                (n_i/Nam)**2*(np.exp(ev*phis/KbT) - ev*phis/KbT - 1))/eps

print E, "[N/C]"

rdata = np.loadtxt("x-phi-0.3.dat")
dx = (rdata[1, 0] - rdata[0, 0])*1e-4 # [m]
dphi = -(rdata[1, 1] - rdata[1, 0])
# W=q*V [ev] (=> V = W/q [ev/C] = W/q * |q| [J/C] = W)

E1 = -dphi/dx

print E1, "[N/C]"

print "ratio:",E/E1

print ev**2*n_i/(eps/11.68*KbT)/(1e12)
