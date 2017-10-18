#!/usr/bin/python2.7
import numpy as np
q = 1.60217662*1e-19 # [C] 
hbar = 6.626070040*1e-34/(2.*np.pi) # [J*s]
m0 = 9.10938356*1e-31
KbT = 1.3806488*1e-23*300 #[J]

#print q*hbar**2/(m0**2*KbT)


print q*1e-12/(m0*0.2)

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
"""


_Esub = 3.; _myyR = 0.9; _mzzR = 0.1; _KbT = 0.38
#FD = lambda ky, kz : 1./(np.exp((_Esub + 2.(ky*ky/_myyR + kz*kz/_mzzR))/_KbT) + 1.)
FD = lambda x : 1./(np.exp(x) + 1.)
dFD = lambda x : -np.exp(x)/(np.exp(x) + 1.)**2

print FD(1), dFD(1)
print FD(1)*(1 - FD(1))
