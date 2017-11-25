#include <iostream>
#include <Eigen/Sparse>
#include <vector>
#include <assert.h>
#include <functional>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <complex>
#include <cstdlib>

#define BERNOLLI_FUNCTION_POLE_ROUND 1e-30

namespace LINALG
{

	template<class T1, class T2>
	double inner_product(const T1& v1, const T2& v2, const int& N)
	{
		double result = 0;
		for(int i=0; i<N; ++i) {
			result += v1[i]*v2[i];
		}
		return result;
	}

}



namespace SPARSE_SOLVER
{
	namespace EIGEN
	{

		using SparseDouble = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;
		using COLAMDO = Eigen::COLAMDOrdering<long long>;
		using nvector = Eigen::VectorXd;

		// Functor for a symmetrc sparse matrix solver
		class CholeskyDecompSolver
		{
		public:
			CholeskyDecompSolver() {}

			void operator()(const SparseDouble& A, nvector& b) const
			{
				const int Ndim = b.size();
				const nvector x(b);

				Eigen::SimplicialCholesky<SparseDouble> solver(A);

				if(solver.info() != Eigen::Success) 
				{
					std::cout<< "!Error : SPARSE_SOLVER::EIGEN::CholeskyDecompSolver"
						 << std::endl;
					assert(false);
				}

				b = solver.solve(x);
			}
		};



		// Functor for a general sparse matrix solver
		class LUdecompSolver
		{
		public:
			LUdecompSolver() {}

			void operator()(const SparseDouble& A, nvector& b) const
			{
				const int Ndim = b.size();
				const nvector x(b);

				Eigen::SparseLU<SparseDouble, COLAMDO> solver;

				solver.analyzePattern(A);

				solver.factorize(A);

				if(solver.info() != Eigen::Success)
				{
					std::cout<< "!Error : SPARSE_SOLVER::EIGEN::LUdecompSolver"
						 << std::endl;
					assert(false);
				}

				b = solver.solve(x);
			}
		};

	}
}



namespace ROOT_FINDING
{

	template<class JacobianObj, class VectorObj>
	class ResidualBase
	{
	public:
		explicit ResidualBase(const int Ndim) : _Ndim(Ndim) {}
		virtual ~ResidualBase() {}

		void residue(const VectorObj& root, VectorObj& f) const;

		virtual void jacobian(const VectorObj& root) = 0;

		virtual JacobianObj& get_J() = 0;

		int size() const {
			return _Ndim;
		}

	protected:
		const int _Ndim;

		virtual double _residual(const VectorObj& root, const int& i) const = 0;
	};

	template<class JacobianObj, class VectorObj>
	void ResidualBase<JacobianObj, VectorObj>::residue(const VectorObj& root, VectorObj& f) const
	{
		for(int i=0; i<_Ndim; ++i) {
			f[i] = _residual(root, i);
		}
	}
	


	template<class JacobianObj, class VectorObj, class SolverObj>
	bool newton_method(ResidualBase<JacobianObj, VectorObj>& object, VectorObj& root,
			   const SolverObj& solver, const size_t niter = 100, const double tol = 1e-7)
	{
		assert(object.size() == root.size());

		const int Ndim = object.size();
		VectorObj up(Ndim), f(Ndim);
		bool isConverge = false;

		for(size_t n=0; n<niter; ++n)
		{
			object.residue(root, f);

			double cost = std::sqrt(LINALG::inner_product(f, f, Ndim));

			std::cout<<"  iter: "<<n<<"\t\tcost: "<<cost<<std::endl;

			if(cost < tol) 
			{
				isConverge = true;
				break;
			}

			up = f;
			object.jacobian(root);
			solver(object.get_J(), up);

			for(int i=0; i<Ndim; ++i) {
				root[i] -= up[i];
			}
		}

		return isConverge;
	}

}

namespace BERNOULLI
{
	using dcomplex = std::complex<double>;
	static const dcomplex IMGPOLE = dcomplex(0, BERNOLLI_FUNCTION_POLE_ROUND);

	double f(const double x) {
		return ((x+IMGPOLE)/(std::exp(x+IMGPOLE) - 1.)).real();
	}

	double df(const double x)
	{
                return ((std::exp(x+IMGPOLE) - 1. - (x+IMGPOLE)*std::exp(x+IMGPOLE))/
                        std::pow((std::exp(x+IMGPOLE)-1.), 2)).real();
	}
}


using sparseMatrix = SPARSE_SOLVER::EIGEN::SparseDouble;
using denseVector  = SPARSE_SOLVER::EIGEN::nvector;
using dvector = std::vector<double>;

/*
f1 = 2*x + 3*y + 2*z = 3
f2 =   x + 2*y -   z = 1
f3 = 4*x -   y + 5*z = 10
*/

/*
f1 = x**2 + y**2 = 1.
f2 = (x-1/2)**2 + y**2 = 1
*/


class testsolver_client : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
{
	using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
public:
	explicit testsolver_client(const size_t Ndim)
	: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	  _J(Ndim, Ndim) {}

	virtual ~testsolver_client() {}

	virtual void jacobian(const denseVector& psi);

	virtual sparseMatrix& get_J() {
		return _J;
	}
private:
	virtual double _residual(const denseVector& psi, const int& i) const;

	sparseMatrix _J;
};


void testsolver_client::jacobian(const denseVector& psi)
{
	_J.coeffRef(0, 0) = 2.*psi(0);
	_J.coeffRef(0, 1) = 2.*psi(1);

	_J.coeffRef(1, 0) = 2.*(psi(0)-1./2.);
	_J.coeffRef(1, 1) = 2.*psi(1);

	_J.makeCompressed();
}


double testsolver_client::_residual(const denseVector& psi, const int& i) const
{
	double res = 0;

	if(i==0) {
		res = std::pow(psi(0), 2) + std::pow(psi(1), 2) - 1.;
	}
	else if(i==1) {
		res = std::pow(psi(0)-1/2., 2) + std::pow(psi(1), 2) - 1.;
	}
	else {
		std::abort();
	}

	return res;
}


int main(int argc, char* argv[])
{
	denseVector root(2);

	root[0] = 1;
	root[1] = 2;

	testsolver_client tester(root.size());

	ROOT_FINDING::newton_method(tester, root, SPARSE_SOLVER::EIGEN::LUdecompSolver(), 100, 1e-15);

	std::cout << " == roots ==" << std::endl;
	std::cout << root << std::endl;
	std::cout << " ===========" << std::endl;

	return 0;
}
