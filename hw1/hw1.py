#!/usr/bin/python2.7
import numpy as np

#The exact results for the Schrodinger equation in 1-D infinite potential wall.
hbar = 6.62607004e-34/(2*np.pi) #(m^2 kg / s)
m    = 9.10938356e-31 #(kg)
a    = 1e-10
N    = 10000
da   = a/(N - 1.) # da = a/(N-1)
Ncut = 100

x = np.linspace(0, a, N)
E = (np.array(range(1, min(N, Ncut)-1))*np.pi*hbar/a)**2/(2.*m)
k = np.sqrt(2.*m*E)/hbar

exact_data = np.zeros([N, min(N, Ncut)-1], dtype = 'float64')
exact_data[:, 0] = x
for i in range(N):
    exact_data[i, 1:] = np.sin(k*x[i])*np.sqrt(2./a)
np.savetxt('exact.dat', exact_data)
