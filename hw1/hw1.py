#!/usr/bin/python2.7
import numpy as np
from scipy.integrate import simps 

#Eigen solver for solving the Schrodinger equation in 1-D infinite potential wall.
hbar = 6.62607004e-34/(2*np.pi) #(m^2 kg / s)
m    = 9.10938356e-31 #(kg)
a    = 1e-10
N    = 100
da   = a/(N - 1.) # da = a/(N-1)
coef = -hbar**2/(2*m*da**2) # -h^2/2m

H = np.zeros([N-2, N-2], dtype = 'float64')
H += -2*np.eye(N-2)

for i in xrange(N-3):
    H[i, i+1] = 1.
    H[i+1, i] = 1.

H *= coef

E, eig = np.linalg.eig(H)

# sorting the arrays according to the order of the array E.
ind = np.argsort(E); E = E[ind]; eig = eig[:, ind]

x = np.array([i*da for i in range(N)])
psi = np.zeros([N, N-2], dtype = 'float64')
# boundary condition: psi(0) = 0, psi(a) = 0
psi[0] = 0; psi[-1] = 0 
psi[1:-1, :] = eig

data = np.zeros([N, N-1], dtype = 'float64')
data[:, 0] = x
data[:, 1:] = psi

# normalizing the wave function
for i in xrange(1, N-1):
    norm = simps(data[:, i]**2, x)
    data[:, i] /= np.sqrt(norm)

np.savetxt('result.dat', data)

print "Eigen energy:"
print E[:10], "..."
print E[-10:]

k = np.sqrt(2.*m*E)/hbar
exact_data = np.zeros([N, N-1], dtype = 'float64')
exact_data[:, 0] = x
for i in range(N):
    exact_data[i, 1:] = np.sin(k*x[i])*np.sqrt(2./a)
np.savetxt('exact.dat', exact_data)
