#include <iostream>
#include <Eigen/Sparse>
#include <vector>
#include <assert.h>
#include <functional>
#include <cmath>
#include <fstream>
#include <iomanip>


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
using dvector = std::vector<double>;

class chargeNeutrality : public ROOT_FINDING::ResidualBase<dvector, dvector>
{
	using ROOT_FINDING::ResidualBase<dvector, dvector>::_Ndim;
public:
	chargeNeutrality()
	: ROOT_FINDING::ResidualBase<dvector, dvector>(1),
	  _dopping(1, 0), _J(1, 0) {}

	virtual ~chargeNeutrality() {}

        virtual void jacobian(const dvector& psi)
        {
                constexpr double h = 5e-8;
                dvector psi_pdh(psi), psi_mdh(psi);

                psi_pdh[0] += h;
                psi_mdh[0] -= h;

                _J[0] = (_residual(psi_pdh, 0) - _residual(psi_mdh, 0))/(2.*h);

                psi_pdh[0] -= h;
                psi_mdh[0] += h;
		
        }

	virtual dvector& get_J() {
		return _J;
	}

	void insert_dopping(const std::vector<double>& dopping) {
		_dopping = dopping;
	}

private:

	virtual double _residual(const dvector& psi, const int& i) const {
		return (std::exp(psi[i]) - std::exp(-psi[i]))/_dopping[i] - 1.;
	}

	std::vector<double> _dopping; // ratio: N+/n_i where N+ is dopping density
	dvector _J;
};



template<class PermittivityObj>
class PoissonEquation : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
{
	using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
	using dvector = std::vector<double>;
public:
	PoissonEquation(const int Ndim, const dvector& dopping,
			const dvector& x, const dvector& boundaries, const double scale = 1.0)
	: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	 _J(Ndim, Ndim), _dopping(dopping), _x(x), _psi0(boundaries), _scale(scale)
	{
		assert(_Ndim == _dopping.size());
		assert(_Ndim + 2 == _x.size());
	}

	virtual ~PoissonEquation() {}

	virtual void jacobian(const denseVector& psi);

	virtual sparseMatrix& get_J() {
		return _J;
	}

private:
	virtual double _residual(const denseVector& psi, const int& i) const;

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
	// psi := q*phi/KbT
	std::vector<double> _psi0;    // boundary conditions at _x[0] and _x[_Ndim-1]
	const PermittivityObj _epsf;  // relative permittivity
	bool _isUpdated = false;
};


template<class PermittivityObj>
void PoissonEquation<PermittivityObj>::jacobian(const denseVector& psi)
{
	for(int i=0; i<_Ndim; ++i)
	{
		double &x_ip1 = _x[i+2], &x_i = _x[i+1], &x_im1 = _x[i];
		
		_J.coeffRef(i, i) = _epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i) +
				    _epsf((x_i+x_im1)/2.)/(x_i-x_im1) +
				    std::pow(_scale, 2)*_coeff*((x_ip1+x_i)/2. - (x_i+x_im1)/2.)*
				    (std::exp(psi[i]) + std::exp(-psi[i]));
		
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

	_J.makeCompressed();
}

template<class PermittivityObj>
double PoissonEquation<PermittivityObj>::_residual(const denseVector& psi, const int& i) const
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

	double result, psi_ip1, psi_i, psi_im1;
	const double &x_ip1 = _x[i+2], &x_i = _x[i+1], &x_im1 = _x[i];

	if(i == 0)
	{
		psi_ip1 = psi[1];
		psi_i   = psi[0];
		psi_im1 = _psi0[0]; // boundary condition
	}
	else if(i == _Ndim-1)
	{
		psi_ip1 = _psi0[1]; // boundary condition
		psi_i   = psi[_Ndim-1];
		psi_im1 = psi[_Ndim-2];
	}
	else
	{
		psi_ip1 = psi[i+1];
		psi_i   = psi[i];
		psi_im1 = psi[i-1];
	}

	result = -_epsf((x_ip1+x_i)/2.)*(psi_ip1 - psi_i)/(x_ip1-x_i) +
		  _epsf((x_i+x_im1)/2.)*(psi_i - psi_im1)/(x_i-x_im1) +
	 	  std::pow(_scale, 2)*_coeff*((std::exp(psi_i) - std::exp(-psi_i)) - _dopping[i])*(x_i - x_im1);

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
	constexpr int Ndim = 999;
	constexpr double Tsi = 1.; // [scale * micrometer]
	constexpr int Ndim = 100001;
	constexpr double Tsi = 5.; // [scale * micrometer]
	constexpr double KbT = 0.025851984732130292; //(ev)
	constexpr double n_i = 1.5e10; // [cm^-3]
	constexpr double KbT_J = 300*1.3806488e-23; //(J)
	constexpr double q0 = 1.6021766208e-19; // (C)

	if(argc == 1)
	{
		std::cout<<"  -- options \n"
			 <<"       argv[1]: q*phi_{s} [ev]\n"
			 <<"       argv[2]: scale ([x] = scale*[micrometer])\n"
			 <<"       argv[3]: file to read initial phi (optional)\n";
		return -1;
	}

	const double psis = std::atof(argv[1])/KbT;
	const double scale = std::atof(argv[2]);

	chargeNeutrality neutral;
	dvector dopping(Ndim, -1e5/1.5);
	neutral.insert_dopping({dopping[0]});

	dvector psic(1,-20);

	ROOT_FINDING::newton_method(neutral, psic, [](const dvector& A, dvector& x){x[0] /= A[0];});

	std::cout<<"q*phic = "<<psic[0]*KbT<<" [ev]"<<std::endl;

	dvector psi0 = {psis, psic[0]};

	dvector x(Ndim+2, 0);

	for(int i=0; i<Ndim+2; ++i) {
		x[i] = Tsi/(Ndim + 1)*i;
	}

	PoissonEquation<permittivityForSilicon> poissonEq(Ndim, dopping, x, psi0, scale);
	denseVector psi(Ndim);

	std::ifstream rfile(argv[3]);

	if(rfile.is_open())
	{
		std::cout<<"  --file: "<<std::string(argv[3])<<std::endl;
		double temp;
		rfile >> temp; rfile >> temp; // read x0 and phi0
		for(int i=0; i<Ndim; ++i) 
		{
			rfile >> temp;   // read x
			rfile >> psi[i]; // read phi [J/C]
			psi[i] *= q0/KbT_J;
		}
	}
	else
	{
		std::cout << "  --default initial psi: linear line for x(psi(x) = ax + b)"
			  << std::endl;
		for(int i=0; i<Ndim; ++i) {
			psi[i] = (psi0[1] - psi0[0])/(Ndim+1)*(i+1);
		}
	}

	rfile.close();

	ROOT_FINDING::newton_method(poissonEq, psi, SPARSE_SOLVER::EIGEN::CholeskyDecompSolver(), 1000, 1e-6);

	std::ofstream wfile(("x-qphi-" + std::string(argv[1]) + ".dat").c_str());
	wfile << x[0] << "\t" << std::setprecision(15) << psi0[0]*KbT << "\n";
	for(int i=0; i<Ndim; ++i) {
		wfile << x[i+1] << "\t" << psi[i]*KbT << "\n";
	}
	wfile << x[Ndim+1] << "\t" << psi0[1]*KbT << "\n";
	wfile.close();

	auto holeDensity = [&n_i](const double psi) -> double {
					return n_i*std::exp(-psi);
				};

	auto elecDensity = [&n_i](const double psi) -> double {
					return n_i*std::exp(psi);
				};


	wfile.open("x-hole-" + std::string(argv[1]) + ".dat");
	wfile << x[0] << "\t" << holeDensity(psi0[0]) << "\n";
	for(int i=0; i<Ndim; ++i) {
		wfile << x[i+1] << "\t" << holeDensity(psi[i]) << "\n";
	}
	wfile << x[Ndim+1] << "\t" << holeDensity(psi0[1]) << "\n";
	wfile.close();

	wfile.open("x-elec-" + std::string(argv[1]) + ".dat");
	wfile << x[0] << "\t" << elecDensity(psi0[0]) << "\n";
	for(int i=0; i<Ndim; ++i) {
		wfile << x[i+1] << "\t" << elecDensity(psi[i]) << "\n";
	}
	wfile << x[Ndim+1] << "\t" << elecDensity(psi0[1]) << "\n";
	wfile.close();

	return 0;
}
