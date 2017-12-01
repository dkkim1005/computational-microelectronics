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

//#define USE_NUMERIC_JACOBIAN 

#ifdef USE_NUMERIC_JACOBIAN
#define JACOBIAN_2WIDTH 1e-5
#endif

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

					//std::cout << A << std::endl;

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

			double cost = std::sqrt(LINALG::inner_product(f, f, Ndim))/Ndim;

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


template<class PermittivityObj>
class DriftDifussionEquation : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
{
	using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
	using dvector = std::vector<double>;
public:
	/*
	    1. roots = {psi, n, p}

	    array assignment : 
		{roots[0], roots[1], roots[2], roots[3*1 + 0], roots[3*1 + 1], ...}
	      = {  psi[0],     n[0],     p[0],   psi[1],           n[1], ...}

            -----------------------------------------------------------------------

	    2. boundaries = {psi, n, p}

	    array assignment : 
		{bound[0], bound[1], bound[2], bound[3], bound[4], bound[5]}
	      = {psi0, n0, p0, psi1, n1, p1}

            -----------------------------------------------------------------------

	    3. residue

	    n = 3*i + j  where the i is the i'th position of the x and the j is the notation for the variables (psi, n, p)
	    F(n) = F(3*i+j) = F_i,j
	   
	    <poisson equation> : F_i,0

	    <equilibrium current equation for the electrons> : F_i,1

	    <equilibrium current equation for the holes> : F_i,2
	*/

	DriftDifussionEquation(const int Ndim, const dvector& dopping,
			const dvector& x, const dvector& boundaries, const double scale = 1.0)
	: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	 _J(Ndim, Ndim), _dopping(dopping), _x(x), _bound(boundaries), _scale(scale)
	{
		assert(_Ndim/3 == _dopping.size());
		assert(_Ndim/3 + 2 == _x.size());
	}

	virtual ~DriftDifussionEquation() {}

	virtual void jacobian(const denseVector& psi);

	virtual sparseMatrix& get_J() {
		return _J;
	}

private:
	virtual double _residual(const denseVector& psi, const int& i) const;

	struct _variables
	{
		double &psi_im1, &psi_i, &psi_ip1,
		         &n_im1,   &n_i,   &n_ip1,
                         &p_im1,   &p_i,   &p_ip1,
                         &x_ip1,   &x_i,   &x_im1;

		_variables(double& psi_im1_, double& psi_i_, double& psi_ip1_,
		           double& n_im1_,   double& n_i_,   double& n_ip1_,
                           double& p_im1_,   double& p_i_,   double& p_ip1_,
                           double& x_ip1_,   double& x_i_,   double& x_im1_)
		: psi_im1(psi_im1_), psi_i(psi_i_), psi_ip1(psi_ip1_),
		    n_im1(n_im1_),     n_i(n_i_),     n_ip1(n_ip1_),
                    p_im1(p_im1_),     p_i(p_i_),     p_ip1(p_ip1_),
                    x_ip1(x_ip1_),     x_i(x_i_),     x_im1(x_im1_) {}
	};

	void _index_rules(_variables& set, const denseVector& psi, const int& i) const;

	sparseMatrix _J;
	/*
	    _coeff : q0^2 * n_i/(eps0 * KbT) [1/micrometer^2]
		q0: unit charge
		n_i: intrinsic carrier density(silicon)
		eps0: vaccum permittivity
		KbT: boltzman constant with room temperature
	*/
	static constexpr double _coeff = 0.0104992634866;
	const double _scale;	      // [x] = _scale[micrometer]
	std::vector<double> _dopping; // ratio: N+/n_i where N+ is dopping density
	std::vector<double> _x;       // [micrometer]
	std::vector<double> _bound;   // boundary conditions at _x[0] and _x[_Ndim-1]
	const PermittivityObj _epsf;  // relative permittivity
};


#ifdef USE_NUMERIC_JACOBIAN
template<class PermittivityObj>
void DriftDifussionEquation<PermittivityObj>::jacobian(const denseVector& root)
{
	const double h = JACOBIAN_2WIDTH;
	denseVector xmh(root), xph(root);

	for(int i=0; i<_Ndim; ++i)
	{
		for(int j=0; j<_Ndim; ++j)
		{
			xmh[j] -= h;
			xph[j] += h;

			_J.coeffRef(i, j) = (_residual(xph, i) - _residual(xmh, i))/(2.*h);

			xmh[j] += h;
			xph[j] -= h;
		}
	}

	_J.makeCompressed();
}


#else
template<class PermittivityObj>
void DriftDifussionEquation<PermittivityObj>::jacobian(const denseVector& root)
{
	double psi_ip1, psi_i, psi_im1, // psi
	       n_ip1, n_i, n_im1,	// n
	       p_ip1, p_i, p_im1,	// p
       	       x_ip1, x_i, x_im1;       // x

	_variables set(psi_im1, psi_i, psi_ip1,
		         n_im1,   n_i,   n_ip1,
           		 p_im1,   p_i,   p_ip1,
			 x_ip1,   x_i,   x_im1);

        for(int i=1; i<_Ndim/3-1; ++i)
        {
		_index_rules(set, root, i);

		_J.coeffRef(3*i, 3*i-3) = -_epsf((x_im1+x_i)/2.)/(x_i - x_im1); // dF_i,0/dpsi_im1
		_J.coeffRef(3*i, 3*i+0) =  _epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i) +
					   _epsf((x_im1+x_i)/2.)/(x_i - x_im1); // dF_i,0/dpsi_i
		_J.coeffRef(3*i, 3*i+3) = -_epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i); // dF_i,0/dpsi_ip1

		_J.coeffRef(3*i, 3*i+1) =  std::pow(_scale, 2)*_coeff*(x_i - x_im1); // dF_i,0/dn_i

		_J.coeffRef(3*i, 3*i+2) = -std::pow(_scale, 2)*_coeff*(x_i - x_im1); // dF_i,0/dp_i


		_J.coeffRef(3*i+1, 3*i-3) =  1./(x_i - x_im1)*(n_i*BERNOULLI::df(psi_i - psi_im1) + n_im1*BERNOULLI::df(-psi_i + psi_im1)); // dF_i,1/dpsi_im1
		_J.coeffRef(3*i+1, 3*i+0) = -1./(x_ip1 - x_i)*(n_ip1*BERNOULLI::df(psi_ip1 - psi_i) + n_i*BERNOULLI::df(-psi_ip1 + psi_i))
					    -1./(x_i - x_im1)*(n_i*BERNOULLI::df(psi_i - psi_im1) + n_im1*BERNOULLI::df(-psi_i + psi_im1)); // dF_i,1/dpsi_i
		_J.coeffRef(3*i+1, 3*i+3) =  1./(x_ip1 - x_i)*(n_ip1*BERNOULLI::df(psi_ip1 - psi_i) + n_i*BERNOULLI::df(-psi_ip1 + psi_i)); // dF_i,1/dpsi_ip1

		_J.coeffRef(3*i+1, 3*i-2) =  1./(x_i - x_im1)*BERNOULLI::f(-psi_i + psi_im1); // dF_i,1/dn_im1
		_J.coeffRef(3*i+1, 3*i+1) = -1./(x_ip1 - x_i)*BERNOULLI::f(-psi_ip1 + psi_i) - 1./(x_i - x_im1)*BERNOULLI::f(psi_i - psi_im1); // dF_i,1/dn_i
		_J.coeffRef(3*i+1, 3*i+4) =  1./(x_ip1 - x_i)*BERNOULLI::f(psi_ip1 - psi_i); // dF_i,1/dn_ip1


		_J.coeffRef(3*i+2, 3*i-3) =  1./(x_i - x_im1)*(-p_i*BERNOULLI::df(-psi_i + psi_im1) - p_im1*BERNOULLI::df(psi_i - psi_im1)); // dF_i,2/dpsi_im1
		_J.coeffRef(3*i+2, 3*i+0) =  1./(x_ip1 - x_i)*(p_ip1*BERNOULLI::df(-psi_ip1 + psi_i) + p_i*BERNOULLI::df(psi_ip1 - psi_i)) +
					     1./(x_i - x_im1)*(p_i*BERNOULLI::df(-psi_i + psi_im1) + p_im1*BERNOULLI::df(psi_i - psi_im1)); // dF_i,2/dpsi_i
		_J.coeffRef(3*i+2, 3*i+3) =  1./(x_ip1 - x_i)*(-p_ip1*BERNOULLI::df(-psi_ip1 + psi_i) - p_i*BERNOULLI::df(psi_ip1 - psi_i)); // dF_i,2/dpsi_ip1

		_J.coeffRef(3*i+2, 3*i-1) =  1./(x_i - x_im1)*BERNOULLI::f(psi_i - psi_im1); // dF_i,2/dp_im1
		_J.coeffRef(3*i+2, 3*i+2) = -1./(x_ip1 - x_i)*BERNOULLI::f(psi_ip1 - psi_i) - 1./(x_i - x_im1)*BERNOULLI::f(-psi_i + psi_im1); // dF_i,2/dp_i
		_J.coeffRef(3*i+2, 3*i+5) =  1./(x_ip1 - x_i)*BERNOULLI::f(-psi_ip1 + psi_i); // dF_i,2/dp_ip1
	}

	// At the x[1]
	_index_rules(set, root, 0);

	_J.coeffRef(0, 0) =  _epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i) + _epsf((x_im1+x_i)/2.)/(x_i - x_im1); // dF_0,0/dpsi_i
	_J.coeffRef(0, 3) = -_epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i); // dF_0,0/dpsi_ip1

	_J.coeffRef(0, 1) =  std::pow(_scale, 2)*_coeff*(x_i - x_im1); // dF_0,0/dn_i

	_J.coeffRef(0, 2) = -std::pow(_scale, 2)*_coeff*(x_i - x_im1); // dF_0,0/dp_i

	// ---------------------------------

	_J.coeffRef(1, 0) = -1./(x_ip1 - x_i)*(n_ip1*BERNOULLI::df(psi_ip1 - psi_i) + n_i*BERNOULLI::df(-psi_ip1 + psi_i))
			    -1./(x_i - x_im1)*(n_i*BERNOULLI::df(psi_i - psi_im1) + n_im1*BERNOULLI::df(-psi_i + psi_im1)); // dF_0,1/dpsi_i

	_J.coeffRef(1, 3) = 1./(x_ip1 - x_i)*(n_ip1*BERNOULLI::df(psi_ip1 - psi_i) + n_i*BERNOULLI::df(-psi_ip1 + psi_i)); // dF_0,1/dpsi_ip1

	_J.coeffRef(1, 1) = -1./(x_ip1 - x_i)*BERNOULLI::f(-psi_ip1 + psi_i) - 1./(x_i - x_im1)*BERNOULLI::f(psi_i - psi_im1); // dF_0,1/dn_i
	_J.coeffRef(1, 4) = 1./(x_ip1 - x_i)*BERNOULLI::f(psi_ip1 - psi_i); // dF_0,1/dn_ip1

	// ---------------------------------

	_J.coeffRef(2, 0) = 1./(x_ip1 - x_i)*(p_ip1*BERNOULLI::df(-psi_ip1 + psi_i) + p_i*BERNOULLI::df(psi_ip1 - psi_i)) +
			    1./(x_i - x_im1)*(p_i*BERNOULLI::df(-psi_i + psi_im1) + p_im1*BERNOULLI::df(psi_i - psi_im1)); // dF_0,2/dpsi_i
	_J.coeffRef(2, 3) = 1./(x_ip1 - x_i)*(-p_ip1*BERNOULLI::df(-psi_ip1 + psi_i) - p_i*BERNOULLI::df(psi_ip1 - psi_i)); // dF_0,2/dpsi_ip1

	_J.coeffRef(2, 2) = -1./(x_ip1 - x_i)*BERNOULLI::f(psi_ip1 - psi_i) - 1./(x_i - x_im1)*BERNOULLI::f(-psi_i + psi_im1); // dF_0,2/dp_i
	_J.coeffRef(2, 5) = 1./(x_ip1 - x_i)*BERNOULLI::f(-psi_ip1 + psi_i); // dF_0,2/dp_ip1



	// At the x[-2]
	_index_rules(set, root, _Ndim/3-1);

	_J.coeffRef(_Ndim-3, _Ndim-3 -3) = -_epsf((x_im1+x_i)/2.)/(x_i - x_im1); // dF_-1,0/dpsi_im1
	_J.coeffRef(_Ndim-3, _Ndim-3) = _epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i) + _epsf((x_im1+x_i)/2.)/(x_i - x_im1); // dF_-1,0/dpsi_i

	_J.coeffRef(_Ndim-3, _Ndim-3 +1) = std::pow(_scale, 2)*_coeff*(x_i - x_im1); // dF_-1,0/dn_i

	_J.coeffRef(_Ndim-3, _Ndim-3 +2) = -std::pow(_scale, 2)*_coeff*(x_i - x_im1); // dF_-1,0/dp_i

	// ---------------------------------

	_J.coeffRef(_Ndim-2, _Ndim-3 -3) = 1./(x_i - x_im1)*(n_i*BERNOULLI::df(psi_i - psi_im1) + n_im1*BERNOULLI::df(-psi_i + psi_im1)); // dF_-1,1/dpsi_im1
	_J.coeffRef(_Ndim-2, _Ndim-3) = -1./(x_ip1 - x_i)*(n_ip1*BERNOULLI::df(psi_ip1 - psi_i) + n_i*BERNOULLI::df(-psi_ip1 + psi_i))
					-1./(x_i - x_im1)*(n_i*BERNOULLI::df(psi_i - psi_im1) + n_im1*BERNOULLI::df(-psi_i + psi_im1)); // dF_-1,1/dpsi_i

	_J.coeffRef(_Ndim-2, _Ndim-3 -2) = 1./(x_i - x_im1)*BERNOULLI::f(-psi_i + psi_im1); // dF_-1,1/dn_im1
	_J.coeffRef(_Ndim-2, _Ndim-3 +1) = -1./(x_ip1 - x_i)*BERNOULLI::f(-psi_ip1 + psi_i) - 1./(x_i - x_im1)*BERNOULLI::f(psi_i - psi_im1); // dF_-1,1/dn_i

	// ---------------------------------

	_J.coeffRef(_Ndim-1, _Ndim-3 -3) = 1./(x_i - x_im1)*(-p_i*BERNOULLI::df(-psi_i + psi_im1) - p_im1*BERNOULLI::df(psi_i - psi_im1)); // dF_-1,2/dpsi_im1
	_J.coeffRef(_Ndim-1, _Ndim-3) = 1./(x_ip1 - x_i)*(p_ip1*BERNOULLI::df(-psi_ip1 + psi_i) + p_i*BERNOULLI::df(psi_ip1 - psi_i)) +
					1./(x_i - x_im1)*(p_i*BERNOULLI::df(-psi_i + psi_im1) + p_im1*BERNOULLI::df(psi_i - psi_im1)); // dF_-1,2/dpsi_i

	_J.coeffRef(_Ndim-1, _Ndim-3 -1) = 1./(x_i - x_im1)*BERNOULLI::f(psi_i - psi_im1); // dF_-1,2/dp_im1
	_J.coeffRef(_Ndim-1, _Ndim-3 +2) = -1./(x_ip1 - x_i)*BERNOULLI::f(psi_ip1 - psi_i) - 1./(x_i - x_im1)*BERNOULLI::f(-psi_i + psi_im1); // dF_-1,2/dp_i


	_J.makeCompressed();
}
#endif


template<class PermittivityObj>
double DriftDifussionEquation<PermittivityObj>::_residual(const denseVector& root, const int& n) const
{
	/*
		x0	_bound
		x1	root_0 {psi_{0}, n_{0}, p_{0}}
		x2	root_1 {psi_{1}, n_{1}, p_{1}}
		.	  .
		.	  .
		.	  .
		.	  .
		x_n	phi_n-1 {psi_{n-1}, n_{n-1}, p_{n-1}}
		x_n+1	_bound
	*/

	// n = 3*i + j
	const int i = n/3;	//(i = 0, 1, .. (_Ndim-1)/3)
	const int j = n - 3*i;	//(j = 0, 1, 2)

	double result,			// F_i,j
	       psi_ip1, psi_i, psi_im1, // psi
	       n_ip1, n_i, n_im1,	// n
	       p_ip1, p_i, p_im1,	// p
               x_ip1, x_i, x_im1;       // x

	_variables set(psi_im1, psi_i, psi_ip1,
		         n_im1,   n_i,   n_ip1,
                         p_im1,   p_i,   p_ip1,
                         x_ip1,   x_i,   x_im1);

	_index_rules(set, root, i);

	switch(j)
	{
		// poisson equation
		case 0 :
			result = -_epsf((x_ip1+x_i)/2.)*(psi_ip1 - psi_i)/(x_ip1-x_i) +
		  	_epsf((x_i+x_im1)/2.)*(psi_i - psi_im1)/(x_i-x_im1) +
	 	  	std::pow(_scale, 2)*_coeff*(n_i - p_i - _dopping[i])*(x_i - x_im1);
			break;

	    	// <equilibrium current equation for the electrons>
		case 1 :
			result = 1./(x_ip1 - x_i)*(n_ip1*BERNOULLI::f( psi_ip1 - psi_i)
						   - n_i*BERNOULLI::f(-psi_ip1 + psi_i))
				-1./(x_i - x_im1)*(  n_i*BERNOULLI::f( psi_i - psi_im1)
						 - n_im1*BERNOULLI::f(-psi_i + psi_im1));
			break;

		// <equilibrium current equation for the holes>
		case 2 :
			result = 1./(x_ip1 - x_i)*(p_ip1*BERNOULLI::f(-psi_ip1 + psi_i)
						   - p_i*BERNOULLI::f(psi_ip1 - psi_i))
				-1./(x_i - x_im1)*(  p_i*BERNOULLI::f(-psi_i + psi_im1)
						 - p_im1*BERNOULLI::f(psi_i - psi_im1));
			break;
		default:
			std::cout << "  -- Error! j is only in the boundary where |j| < 3." << std::endl;
			assert(false);
	}

	return result;
}


template<class PermittivityObj>
void DriftDifussionEquation<PermittivityObj>::_index_rules(_variables& set, const denseVector& root, const int& i) const
{
	double  psi_ip1 = 0, psi_i = 0, psi_im1 = 0, // psi
                n_ip1   = 0,   n_i = 0,   n_im1 = 0, // n 
		p_ip1   = 0,   p_i = 0,   p_im1 = 0, // p 
                x_ip1   = _x[i+2], x_i = _x[i+1], x_im1 = _x[i];

        if(i == 0)
        {
                psi_ip1 = root[3*1 + 0]; 
                psi_i   = root[3*0 + 0]; 
                psi_im1 = _bound[0]; // boundary condition

                n_ip1 = root[3*1 + 1]; 
                n_i   = root[3*0 + 1]; 
                n_im1 = _bound[1]; // boundary condition

                p_ip1 = root[3*1 + 2]; 
                p_i   = root[3*0 + 2]; 
                p_im1 = _bound[2]; // boundary condition

        }
        else if(i == (_Ndim-1)/3)
        {
                psi_ip1 = _bound[3]; // boundary condition
                psi_i   = root[3*i + 0]; 
                psi_im1 = root[3*(i-1) + 0]; 

                n_ip1 = _bound[4]; // boundary condition
                n_i   = root[3*i + 1]; 
                n_im1 = root[3*(i-1) + 1]; 

                p_ip1 = _bound[5]; // boundary condition
                p_i   = root[3*i + 2]; 
                p_im1 = root[3*(i-1) + 2]; 
        }
        else
        {
                psi_ip1 = root[3*(i+1) + 0]; 
                psi_i   = root[3*i + 0]; 
                psi_im1 = root[3*(i-1) + 0]; 

                n_ip1 = root[3*(i+1) + 1]; 
                n_i   = root[3*i + 1]; 
                n_im1 = root[3*(i-1) + 1]; 

                p_ip1 = root[3*(i+1) + 2]; 
                p_i   = root[3*i + 2]; 
                p_im1 = root[3*(i-1) + 2]; 
        }

	set.psi_im1 = psi_im1;
	set.psi_i   = psi_i;
	set.psi_ip1 = psi_ip1;

	set.n_im1   = n_im1;
	set.n_i     = n_i;
	set.n_ip1   = n_ip1;

        set.p_im1   = p_im1;
	set.p_i     = p_i;
	set.p_ip1   = p_ip1;

	set.x_ip1   = x_ip1;
	set.x_i     = x_i;
	set.x_im1   = x_im1;
}


class permittivityForSilicon
{
public:
	permittivityForSilicon() {}

	// relative permittivity for the silicon
	double operator()(const double& x) const {
		constexpr double eps = 11.68;
		return eps;
	}
};


/*
namespace
{
	struct read_environment
	{
		void read(const char filename[])
		{
			std::cout << " --- read file: "<< filename << std::endl;
			std::ifstream rfile(filename);
			if(!rfile.is_open())
			{
				std::cout << "  Error : there is no object file to read in the current directory!" << std::endl;
				std::abort();
			}

			rfile >> Nx;
			rfile >> Tsi;
			rfile >> scale;

			x = std::vector<double>(Nx);

			for(int i=0; i<Nx; ++i) {
				x[i] = Tsi/(Nx - 1)*i;
			}

		}

		std::vector<double> bound, x, Nplus;
		double Nx, Tsi, scale;
	};
}
*/


int main(int argc, char* argv[])
{
	constexpr int Nx = 503;
	constexpr double Tsi = 1.; // [scale * micrometer]
	constexpr double KbT = 0.025851984732130292; //(ev)
	constexpr double n_i = 1.5e10; // [cm^-3]
	constexpr double KbT_J = 300*1.3806488e-23; //(J)
	constexpr double q0 = 1.6021766208e-19; // (C)
	const double scale = 1.;

	dvector dopping(Nx-2, 1e6/1.5), bound(6, 0);
        for(int i=0; i<dopping.size()/2; ++i) {
                dopping[i] *= -1; // (N- range for 0 to 1[micrometer])
        }

	// boundary condition 

	// at the x[0]
	bound[0] = -1.341000000000000014e+01;// psi := q*phi/KbT
	bound[1] = 1.500000000000000038e-06;  //   n := n/n_i (n_i : intrinsic density of the silicon)
	bound[2] = 6.666670000000000000e+05 ; //   p := p/n_i

	// at the x[-1]
	bound[3] = 1.341000000000000014e+01 ; // psi := q*phi/KbT
	bound[4] = 6.666670000000000000e+05 ; //   n := n/n_i 
	bound[5] = 1.500000000000000038e-06 ;  //   p := p/p_i 

	dvector x(Nx, 0);

	for(int i=0; i<Nx; ++i) {
		x[i] = Tsi/(Nx - 1)*i;
	}

	DriftDifussionEquation<permittivityForSilicon> DDEq(3*(Nx-2), dopping, x, bound, scale);
	denseVector root(3*(Nx-2)), residue(3*(Nx-2));

	std::ifstream rfile(argv[1]);

	if(rfile.is_open())
	{
		std::cout<<"  --file: "<<std::string(argv[1])<<std::endl;
		double temp;
		// read x0 and the values at the boundary x[0]
		rfile >> temp; rfile >> temp; rfile >> temp; rfile >> temp;

		for(int i=0; i<(Nx-2); ++i) 
		{
			rfile >> temp;   // read x
			rfile >> root[3*i];   // read psi (q*psi/KbT)
			rfile >> root[3*i+1]; // read n   (n/n_i)
			rfile >> root[3*i+2]; // read p   (p/p_i)
		}
	}
	else
	{
		std::cout << "  --default initial root: linear line for x(root(x) = ax + b)"
			  << std::endl;
		for(int i=0; i<(Nx-2); ++i) {
			root[3*i + 0] = bound[0] + (bound[3] - bound[0])*std::pow(1./(Nx - 1)*(i+1), 1);
			root[3*i + 1] = bound[1] + (bound[4] - bound[1])*std::pow(1./(Nx - 1)*(i+1), 1);
			root[3*i + 2] = bound[2] + (bound[5] - bound[2])*std::pow(1./(Nx - 1)*(i+1), 1);
		}
	}

	rfile.close();

	ROOT_FINDING::newton_method(DDEq, root, SPARSE_SOLVER::EIGEN::LUdecompSolver(), 10, 1e-7);

	std::ofstream wfile("result.dat");
	wfile << x[0] << "\t" << std::setprecision(15) << bound[0] << "\t" << bound[1] << "\t" << bound[2] << std::endl;
	for(int i=0; i<Nx-2; ++i) {
		wfile << x[i+1] << std::setprecision(15) << "\t" << root[3*i] << "\t" << root[3*i+1] << "\t" << root[3*i+2] << std::endl;
	}
	wfile << x[Nx-1] << "\t" << std::setprecision(15) << bound[3] << "\t" << bound[4] << "\t" << bound[5] << std::endl;

	wfile.close();

/*
	std::ofstream wfile("x-qphi.dat");
	wfile << x[0] << "\t" << std::setprecision(15) << bound[0]*KbT << "\n";
	for(int i=0; i<Nx-2; ++i) {
		wfile << x[i+1] << "\t" << root[3*i]*KbT << "\n";
	}
	wfile << x[Nx-1] << "\t" << bound[3]*KbT << "\n";
	wfile.close();

	wfile.open("x-hole.dat");
	wfile << x[0] << "\t" << bound[2] << "\n";
	for(int i=0; i<Nx-2; ++i) {
		wfile << x[i+1] << "\t" << root[3*i+2] << "\n";
	}
	wfile << x[Nx-1] << "\t" << bound[5] << "\n";
	wfile.close();

	wfile.open("x-elec.dat");
	wfile << x[0] << "\t" << bound[1] << "\n";
	for(int i=0; i<Nx-2; ++i) {
		wfile << x[i+1] << "\t" << root[3*i+1] << "\n";
	}
	wfile << x[Nx-1] << "\t" << bound[4] << "\n";
	wfile.close();
*/

	return 0;
}
