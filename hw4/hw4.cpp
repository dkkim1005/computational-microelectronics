#include <iostream>
#include <Eigen/Sparse>
#include <vector>
#include <assert.h>
#include <functional>
#include <cmath>
#include <fstream>


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



using sparseMatrix = SPARSE_SOLVER::EIGEN::SparseDouble;
using denseVector  = SPARSE_SOLVER::EIGEN::nvector;


class chargeNeutrality : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
{
	using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
public:
	explicit chargeNeutrality(const int Ndim)
	: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	  _dopping(Ndim), _J(Ndim, Ndim) {}

	virtual ~chargeNeutrality() {}

        virtual void jacobian(const denseVector& root)
        {
                constexpr double h = 5e-8;
                denseVector root_pdh(root), root_mdh(root);

		for(int i=0; i<_Ndim; ++i)
		{
                	root_pdh[i] += h;
                	root_mdh[i] -= h;

                	_J.coeffRef(i, i) = (_residual(root_pdh, i) - _residual(root_mdh, i))/(2.*h);

                	root_pdh[i] -= h;
                	root_mdh[i] += h;
		}

		_J.makeCompressed();
        }

	virtual sparseMatrix& get_J() {
		return _J;
	}

	void insert_dopping(const std::vector<double>& dopping) {
		_dopping = dopping;
	}

private:

	virtual double _residual(const denseVector& root, const int& i) const {
		return (std::exp(root[i]) - std::exp(-root[i]))/_dopping[i] - 1.;
	}

	std::vector<double> _dopping; // ratio: N+/n_i where N+ is dopping density
	sparseMatrix _J;
};



template<class PermittivityObj>
class PoissonEquation : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
{
	using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
	using dvector = std::vector<double>;
public:
	PoissonEquation(const int Ndim, const dvector& dopping,
			const dvector& x, const dvector& boundaries)
	: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	 _J(Ndim, Ndim), _dopping(dopping), _x(x), _phi0(boundaries)
	{
		assert(_Ndim == _dopping.size());
		assert(_Ndim + 2 == _x.size());
	}

	virtual ~PoissonEquation() {}

	virtual void jacobian(const denseVector& phi);

	virtual sparseMatrix& get_J() {
		return _J;
	}

private:
	virtual double _residual(const denseVector& phi, const int& i) const;

	sparseMatrix _J;
	/*
	    _coeff : q0^2 * n_i/(eps0 * KbT) [1/micrometer^2]
		q0: unit charge
		n_i: intrinsic carrier density(silicon)
		eps0: vaccum permittivity
		KbT: boltzman constant with room temperature
	*/
	static constexpr double _coeff = 0.010499272845780509;
	std::vector<double> _dopping; // ratio: N+/n_i where N+ is dopping density
	std::vector<double> _x;       // [micrometer]
	std::vector<double> _phi0;    // boundary conditions at _x[0] and _x[_Ndim-1]
	const PermittivityObj _epsf;  // relative permittivity
	bool _isUpdated = false;
};


template<class PermittivityObj>
void PoissonEquation<PermittivityObj>::jacobian(const denseVector& phi)
{
	for(int i=0; i<_Ndim; ++i)
	{
		double &x_ip1 = _x[i+2], &x_i = _x[i+1], &x_im1 = _x[i];
		
		_J.coeffRef(i, i) = _epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i) +
				    _epsf((x_i+x_im1)/2.)/(x_i-x_im1) +
				    _coeff*((x_ip1+x_i)/2. - (x_i+x_im1)/2.)*
				    (std::exp(phi[i]) + std::exp(-phi[i]));
		
	}

	// off-diagonal for a jacobian matrix
	if(!_isUpdated)
	{
		for(int i=1; i<_Ndim; ++i)
		{
			double &x_ip1 = _x[i+2], &x_i = _x[i+1], &x_im1 = _x[i];
			_J.coeffRef(i, i-1) = -_epsf((x_i+x_im1)/2.)/(x_i - x_im1);
			_J.coeffRef(i-1, i) = _J.coeffRef(i, i-1);
			
		}
		_isUpdated = true;
	}
}

template<class PermittivityObj>
double PoissonEquation<PermittivityObj>::_residual(const denseVector& phi, const int& i) const
{
	/*
		x0	phi0
		x1	phi_0
		x2	phi_1
		.	  .
		.	  .
		.	  .
		.	  .
		x_n	phi_n-1
		x_n+1	phi0
	*/

	/*
		! equation set:

		-_epsf(x_{i+0.5})(phi_{i+1} - phi_{i})/(x_{i+1} - x_{i}) +
		 _epsf(x_{i-0.5})*(phi_{i} - phi_{i-1})/(x_i - x_{i-1}) +
		 qn_i*(exp(phi_i/Vth) - exp(-phi_i/Vth) - N_i)*(x_{i}- x_{i-1}) = 0
	*/

	double result, phi_ip1, phi_i, phi_im1;
	const double &x_ip1 = _x[i+2], &x_i = _x[i+1], &x_im1 = _x[i];

	if(i == 0)
	{
		phi_ip1 = phi[1];
		phi_i   = phi[0];
		phi_im1 = _phi0[0]; // boundary condition
	}
	else if(i == _Ndim-1)
	{
		phi_ip1 = _phi0[1]; // boundary condition
		phi_i   = phi[_Ndim-1];
		phi_im1 = phi[_Ndim-2];
	}
	else
	{
		phi_ip1 = phi[i+1];
		phi_i   = phi[i];
		phi_im1 = phi[i-1];
	}

	result = -_epsf((x_ip1+x_i)/2.)*(phi_ip1 - phi_i)/(x_ip1-x_i) +
		  _epsf((x_i+x_im1)/2.)*(phi_i - phi_im1)/(x_i-x_im1) +
	 	  _coeff*((std::exp(phi_i) - std::exp(-phi_i)) - _dopping[i])*(x_i - x_im1);

	return result;
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



int main(int argc, char* argv[])
{
	const int Ndim = 1000;
	const double KbT = 0.025851984732130292; //(ev)

	/* dopping density: 1e16 [cm^-3]
           ratio(per n_i): 1e16/(1.5*1e10) = 1e6/1.5 */
	std::vector<double> dopping(Ndim, 1e6/1.5);
	for(int i=0; i<dopping.size()/2; ++i) {
		dopping[i] *= -1; // (N- range for 0 to 1[micrometer])
	}

	// charge neutrality equation
	chargeNeutrality sp_object(Ndim);

	// electric potential phi
	denseVector phi(Ndim);

	sp_object.insert_dopping(dopping);

	for(int i=0; i<Ndim/2; ++i) {
		phi[i] = -std::log(std::abs(dopping[0]));
	}
	for(int i=Ndim/2; i<Ndim; ++i) {
		phi[i] = std::log(std::abs(dopping[Ndim-1]));
	}
	
	ROOT_FINDING::newton_method(sp_object, phi, SPARSE_SOLVER::EIGEN::CholeskyDecompSolver(), 100);

	std::cout<<phi[0]*KbT<<"(ev)"<<std::endl;
	std::cout<<phi[Ndim-1]*KbT<<"(ev)"<<std::endl;
	
	std::vector<double> x(Ndim+2, 0);

	// x: 0.0 to 2.0 [micrometer]
	for(int i=0; i<Ndim+2; ++i) {
		x[i] = 2./(Ndim + 1.)*i;
	}

	// boundary conditions
	std::vector<double> phi0 = {phi[0], phi[Ndim-1]};

	// Poisson equation
	PoissonEquation<permittivityForSilicon> poissonEq(Ndim, dopping, x, phi0);

	ROOT_FINDING::newton_method(poissonEq, phi, SPARSE_SOLVER::EIGEN::CholeskyDecompSolver(), 100, 1e-6);

	std::ofstream wfile("x-phi.dat");
	//wfile << x[0] << "\t" << phi0[0]*KbT << "\n";
	wfile << x[0] << "\t" << phi0[0] << "\n";
	for(int i=0; i<Ndim; ++i) {
		//wfile << x[i+1] << "\t" << phi[i]*KbT << "\n";
		wfile << x[i+1] << "\t" << phi[i] << "\n";
	}
	//wfile << x[Ndim+1] << "\t" << phi0[1]*KbT << "\n";
	wfile << x[Ndim+1] << "\t" << phi0[1] << "\n";
	wfile.close();

	constexpr double n_i = 1.5e10; // carrier density for the intrinsic silicon [cm^-3]

	auto holeDensity = [&n_i](const double phi) -> double {
					return n_i*std::exp(phi);
				};

	auto elecDensity = [&n_i](const double phi) -> double {
					return n_i*std::exp(-phi);
				};

	wfile.open("x-hole.dat");
	wfile << x[0] << "\t" << holeDensity(phi0[0]) << "\n";
	for(int i=0; i<Ndim; ++i) {
		wfile << x[i+1] << "\t" << holeDensity(phi[i]) << "\n";
	}
	wfile << x[Ndim+1] << "\t" << holeDensity(phi0[1]) << "\n";
	wfile.close();

	wfile.open("x-elec.dat");
	wfile << x[0] << "\t" << elecDensity(phi0[0]) << "\n";
	for(int i=0; i<Ndim; ++i) {
		wfile << x[i+1] << "\t" << elecDensity(phi[i]) << "\n";
	}
	wfile << x[Ndim+1] << "\t" << elecDensity(phi0[1]) << "\n";
	wfile.close();

	return 0;
}
