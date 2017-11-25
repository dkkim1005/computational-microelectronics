#!/usr/bin/python2.7
import numpy as np
from scipy.optimize import root


"""
-hbar^2/2m
((psi(x_{i+1}) - psi(x_{i}))/(x_{i+1} - x_{i}) - (psi(x_{i}) - psi(x_i{i-1}))/(x_{i} - x_{i-1}))/((x_{i+1} - x_{i})/2.)
+ V(x_{i})*psi(x_{i}) = E*psi(x_{i})

-hbar^2/2m
(psi(x_{i+1}) - psi(x_{i}))/(x_{i+1} - x_{i}) - (psi(x_{i}) - psi(x_i{i-1}))/(x_{i} - x_{i-1})
+ V(x_{i})*((x_{i+1} - x_{i})/2.)*psi(x_{i}) = E*((x_{i+1} - x_{i})/2.)*psi(x_{i})

i,i+1 : -hbar^2/2m*1/(x_{i+1} - x_{i})
i+1,i : -hbar^2/2m*1/(x_{i} - x_{i-1})
i,i   : -hbar^2/2m*(-1/(x_{i+1} - x_{i}) -1/(x_{i} - x_{i-1})) + V(x_{i})*(x_{i+1} - x_{i-1})/2.

"""

class eigensolver:
    def __init__(self, x, scale, meffRatio):
        self._x = x
        self._Nx = len(x)
        self._scale = scale
        _coeff = 3.8099815e-8
        
        self._H = np.zeros([self._Nx-2, self._Nx-2], dtype = 'float64')
        self._V = np.zeros([self._Nx-2], dtype = 'float64')

        for i in range(self._Nx-2):
            self._H[i, i] = -_coeff/(meffRatio*self._scale)*\
                            (-1./(self._x[i+2] - self._x[i+1]) - 1./(self._x[i+1] - self._x[i]))
                            
        for i in range(self._Nx-3):
            self._H[i, i+1] = -_coeff/(meffRatio*self._scale)*\
                            (1./(self._x[i+2] - self._x[i+1]))
            self._H[i+1, i] = -_coeff/(meffRatio*self._scale)*\
                            (1./(self._x[i+1] - self._x[i]))


    def insert_V(self, V):
         for i in range(self._Nx-2):
             self._H[i, i] += (V[i] - self._V[i])*(self._x[i+2] - self._x[i])/2.*self._scale
             self._V[i] = V[i]


    def compute(self, solver, nev):
        energy, psi = solver(self._H, nev)
        psi = self._set_norm(psi);
        psi = self._set_sign(psi)

        for i in range(nev):
            energy[i] /= (self._x[i+2] - self._x[i])/2.*self._scale

        return energy, psi


    def _set_norm(self, psi):
        nev = len(psi[0])
        psiSquare = np.zeros([self._Nx], dtype = 'float64');

        for j in range(nev):
            for i in range(1, self._Nx-1): 
                psiSquare[i] = psi[i-1, j]**2;

            norm = np.trapz(psiSquare, x = self._x);

            for i in range(1, self._Nx-1):
                psi[i-1, j] /= np.sqrt(norm);

        return psi


    def _set_sign(self, psi, sign = 1.):
        nev = len(psi[0])

        for j in range(nev): 
            if(psi[0, j]*sign < 0):
                for i in range(self._Nx-2):
                    psi[i, j] *= -1;

        return psi


def uniformDenseDiagSolver(A, nev):
    eValue, eVector = np.linalg.eig(A)
    arg = np.argsort(eValue)
    eValue = (eValue[arg])[:nev]
    eVector = (eVector[:, arg])[:, :nev]

    return eValue, eVector

x = np.linspace(0, 1, 1001)
solver = eigensolver(x, 1, 1)
energy, psi = solver.compute(uniformDenseDiagSolver, 10)

data = np.zeros([1001, 1 + len(psi[0, :])])
data[:, 0] = x
data[1:1000, 1:] = psi
np.savetxt('psi-save3.dat', data)
