#!/usr/bin/python2.7
import numpy as np
from bernoulli import *
from newton_solver import *

class difusion_drift:
    def __init__(self, x, Nplus, bound, epsf, scale = 1.):
        """
            _C = q0^2 * n_i/(eps0 * KbT) [1/micrometer^2]
                 q0:   unit charge
                 n_i:  the intrinsic carrier density(silicon)
                 eps0: vaccum permittivity
                 KbT:  boltzman constant at a room temperature
        """
        self._C       = 0.0104992634866
        self._x       = x         # [micrometer]
        self._Nplus   = Nplus     # ratio: N+/n_i where N+ is dopping density
        self._bound   = bound     # boundary conditions at x[0] and x[_Ndim-1]
        self._scale   = scale     # [x] = _scale[micrometer]
        self._epsf    = epsf      # susceptibility
        self._Ndim    = (len(x) - 2)*3


    def _index_convert_i(self, root, i):
        psi_ip1 = 0; psi_i = 0; psi_im1 = 0; # psi
        n_ip1   = 0;   n_i = 0;   n_im1 = 0; # n
        p_ip1   = 0;   p_i = 0;   p_im1 = 0; # p

        x_ip1 = self._x[i+2]; x_i = self._x[i+1]; x_im1 = self._x[i];

        if i == 0:
            psi_ip1 = root[3*1 + 0]
            psi_i   = root[3*0 + 0]
            psi_im1 = self._bound[0]   # boundary condition

            n_ip1 = root[3*1 + 1]
            n_i   = root[3*0 + 1]
            n_im1 = self._bound[1]     # boundary condition

            p_ip1 = root[3*1 + 2]
            p_i   = root[3*0 + 2]
            p_im1 = self._bound[2]     # boundary condition

        elif i == (self._Ndim-1)/3:
            psi_ip1 = self._bound[3]   # boundary condition
            psi_i   = root[3*i + 0]
            psi_im1 = root[3*(i-1) + 0]

            n_ip1   = self._bound[4]   # boundary condition
            n_i     = root[3*i + 1]
            n_im1   = root[3*(i-1) + 1]

            p_ip1 = self._bound[5]     # boundary condition
            p_i   = root[3*i + 2]
            p_im1 = root[3*(i-1) + 2]
        else:
            psi_ip1 = root[3*(i+1) + 0]
            psi_i   = root[3*i + 0]
            psi_im1 = root[3*(i-1) + 0]

            n_ip1 = root[3*(i+1) + 1]
            n_i   = root[3*i + 1]
            n_im1 = root[3*(i-1) + 1]

            p_ip1 = root[3*(i+1) + 2]
            p_i   = root[3*i + 2]
            p_im1 = root[3*(i-1) + 2]

        return x_ip1, x_i, x_im1,       \
               psi_ip1, psi_i, psi_im1, \
               n_ip1,   n_i,   n_im1,   \
               p_ip1,   p_i,   p_im1;


    def _F_ij(self, root, i, j):
        x_ip1, x_i, x_im1,               \
        psi_ip1, psi_i, psi_im1,       \
        n_ip1,   n_i,   n_im1,         \
        p_ip1,   p_i,   p_im1          \
        = self._index_convert_i(root, i)

        # poisson equation
        if j == 0 :
            result = -self._epsf((x_ip1+x_i)/2.)*(psi_ip1 - psi_i)/(x_ip1-x_i) + \
                      self._epsf((x_i+x_im1)/2.)*(psi_i - psi_im1)/(x_i-x_im1) + \
                      self._scale**2*self._C*(n_i - p_i - self._Nplus[i])*(x_i - x_im1)

        # <equilibrium current equation for the electrons>
        elif j == 1:
            result = 1./(x_ip1 - x_i)*(n_ip1*f( psi_ip1 - psi_i)  \
                                       - n_i*f(-psi_ip1 + psi_i)) \
                    -1./(x_i - x_im1)*(  n_i*f( psi_i - psi_im1)  \
                                       - n_im1*f(-psi_i + psi_im1))
        # <equilibrium current equation for the holes>
        elif j == 2:
            result = 1./(x_ip1 - x_i)*(p_ip1*f( psi_ip1 - psi_i)  \
                                       - p_i*f(-psi_ip1 + psi_i)) \
                    -1./(x_i - x_im1)*(  p_i*f( psi_i - psi_im1)  \
                                       - p_im1*f(-psi_i + psi_im1));
        else:
            print "  -- Error! j is only in the boundary where |j| < 3."
            assert(False)

        return result;


    def residue(self, root):
        """
                x0      _bound
                x1      root_0 {psi_{0}, n_{0}, p_{0}}
                x2      root_1 {psi_{1}, n_{1}, p_{1}}
                .         .
                .         .
                .         .
                .         .
                x_n     phi_n-1 {psi_{n-1}, n_{n-1}, p_{n-1}}
                x_n+1   _bound
        """

        F = np.zeros([self._Ndim])

        for n in xrange(self._Ndim):
            i = n/3      # (i = 0, 1, .. (_Ndim-1)/3)
            j = n - 3*i  # (j = 0, 1, 2)

            F[n] = self._F_ij(root, i, j)

        return F;


    def _residue(self, root, n):
        i = n/3      # (i = 0, 1, .. (_Ndim-1)/3)
        j = n - 3*i  # (j = 0, 1, 2)

        return self._F_ij(root, i, j)


    def jacobian(self, root):
        h = 1e-7
        rootpls = np.array(root)
        J = np.zeros([self._Ndim, self._Ndim])

        for i in xrange(self._Ndim):
            for j in xrange(self._Ndim):
                rootpls[j] += h
                J[i, j] = (self._residue(rootpls, i) - self._residue(root, i))/h
                rootpls[j] -= h

        return J 

    def size(self):
        return self._Ndim


def permittivityForSilicon(x):
    return 11.68


if __name__ == "__main__":
    x = np.linspace(0, 1, 100)
    Nplus = np.zeros((len(x)-2))
    Nplus[:5] = -1e6/1.5; Nplus[5:] = 1e6/1.5;
    bound = np.array([-13.41,
                       1e16/(1.5*1e10),
                       22500/(1.5*1e10),
                       13.41,
                       22500/(1.5*1e10),
                       1e16/(1.5*1e10)])

    eqs = difusion_drift(x, Nplus, bound, permittivityForSilicon, scale = 1.)

    root = np.zeros([(len(x)-2)*3])


    for i in xrange(len(x)-2):
        root[3*i + 0] = bound[0] + (bound[3] - bound[0])*(1./(len(x) - 1)*(i+1))**2
        root[3*i + 1] = bound[1] + (bound[4] - bound[1])*(1./(len(x) - 1)*(i+1))**2
        root[3*i + 2] = bound[2] + (bound[5] - bound[2])*(1./(len(x) - 1)*(i+1))**2


    J = eqs.jacobian(root)

    msg, root = newton_method(eqs, root, lu_solver, niter = 100, tol = 1e-7)

    data   = np.zeros([len(x), 2])
    data[:, 0] = x
    data[0, 1] = bound[0]; data[-1, 1] = bound[3]
    for i in xrange(1, len(x)-1):
        data[i, 1] = root[3*(i-1)]

    np.savetxt("dat", data)
