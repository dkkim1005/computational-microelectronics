#!/usr/bin/python2.7
import numpy as np
"""
q = 1.60217662*1e-19 # [C] 
hbar = 6.626070040*1e-34/(2.*np.pi) # [J*s]
m0 = 9.10938356*1e-31
KbT = 1.3806488*1e-23*300 #[J]

print q*hbar**2/(m0**2*KbT)
"""
coeff2 = 3.8099815e-2
KbT   = 0.0258520
Esub  = 0.5
myyR  = 0.91
mzzR  = 0.19
Esub  = 0.5

FD = lambda ky, kz : 1./(np.exp((Esub + coeff2*(ky*ky/myyR + kz*kz/mzzR))/KbT) + 1.)

k = np.linspace(-3., 3., 1001)

fy = FD(k, 0)
fz = FD(0, k)

data1 = np.zeros([len(fy), 2])
data2 = np.zeros([len(fy), 2])

data1[:, 0] = k; data1[:, 1] = fy
data2[:, 0] = k; data2[:, 1] = fz

np.savetxt("f1-dat", data1)
np.savetxt("f2-dat", data2)
