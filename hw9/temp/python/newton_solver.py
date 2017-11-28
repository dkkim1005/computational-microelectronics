#!/usr/bin/python2.7
import numpy as np
from scipy.linalg import lu_factor, lu_solve

def newton_method(eqObject, root, solver, niter = 100, tol = 1e-7):
    assert(eqObject.size() == len(root))

    Ndim = eqObject.size()
    up = np.array([Ndim]); f = np.array([Ndim])
    isConverge = False

    for n in xrange(niter):
        f = eqObject.residue(root)

        cost = np.linalg.norm(f)

        print "  iter: ", n, "\t\tcost: ", cost

        if cost < tol:
            isConverge = True
            break;

        up = f;
        J = eqObject.jacobian(root)
        step = solver(J, up)

        root -= step;

    return isConverge, root


def lu_solver(A, b):
    lu_info = lu_factor(A)
    return lu_solve(lu_info, b)




if __name__ == "__main__":

    class eq1:
        def residue(self, root):
            F = np.zeros([2])
            F[0] = root[0]**2 + root[1]**2 - 1./2.
            F[1] = root[0] + root[1] - 1.
            return F

        def jacobian(self, root):
            J = np.zeros([2, 2])
            J[0, 0] = 2*root[0]
            J[0, 1] = 2*root[1]
            J[1, 0] = 1
            J[1, 1] = 1
            return J

        def size(self):
            return 2

    def solver(J, up):
        return np.dot(np.linalg.inv(J), up)

    eq = eq1()
    root = [0, 1]
    msg, root = newton_method(eq, root, lu_solver, niter = 100, tol = 1e-10)

    print root
