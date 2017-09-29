#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <functional>
#include <fstream>
#include <Eigen/Core>
#include "calculus.h"
#include "sparse_solver.h"
#include "root_finding.h"


namespace SCHRODINGER_POISSON
{

	class Number_density
	{

	using dvector = std::vector<double>;

	public:
	
		Number_density(const int N, const double Width)
		: _N(N), _Width(Width), _integral(N)
		{}

		// Psi [micrometer^(-1/2)]
		dvector density(const Eigen::VectorXd& Esub0p91, const Eigen::VectorXd& Esub0p19,
			        const Eigen::MatrixXd& Psi0p91,  const Eigen::MatrixXd& Psi0p19) const
		{
			// nsub1, nsub2  [cm^-3]
			const dvector&& nsub1 = _n_sub_band(Esub0p91, Psi0p91, 0.19, 0.19);
			const dvector&& nsub2 = _n_sub_band(Esub0p19, Psi0p19, 0.91, 0.19);

			if(nsub1.size() != nsub2.size())
			{
				std::cout << "  !Error : SCHRODINGER_POISSON::Number_density::_n_sub_band\n"
					  << "  nsub1.size() != nsub2.size()\n";
				std::abort();
			}

			const int Npoints = nsub1.size();

			dvector nx(Npoints); // [cm^-3]

			for(int i=0; i<Npoints; ++i) {
				nx[i] = 2.*nsub1[i] + 4.*nsub2[i];
			}

			return std::move(nx);
		}

	private:

		class _FermiDiracDist
		{
		public:
			_FermiDiracDist(const double myyR, const double mzzR, const double Esub)
			: _myyR(myyR), _mzzR(mzzR), _Esub(Esub) {}

			// ky : [1/nm^2],  kz : [1/nm^2]
			double operator()(const double& ky, const double& kz) const {
				return 1./(std::exp((_Esub + _coeff*(ky*ky/_myyR + kz*kz/_mzzR))/_KbT) + 1.);
			}

		private:
			const double _myyR; // fraction for the electron mass (yy direction)
			const double _mzzR; // fraction for the electron mass (zz direction)
			const double _Esub; // [ev]
			static constexpr double _KbT   = 0.0258520;    // [ev]
			static constexpr double _coeff = 3.8099815e-2; // hbar^2/(2*m0) [ev*nm^2]
		};


		// Psi [micrometer^(-1/2)]
		dvector _n_sub_band(const Eigen::VectorXd& Esub, const Eigen::MatrixXd& Psi, const double myyR, const double mzzR) const
		{
			try
			{
				if(Esub.size() != Psi.cols()) {
					throw "  :Error!:  SCHRODINGER_POISSON::Number_density::_n_sub_band\n   Esub.size() != Psi.cols()";
				}
			}
			catch(const char errmsg[])
			{
				std::cout << errmsg << std::endl;
				std::abort();
			}

			dvector k(_N, 0);

			const int Nev = Esub.size();
			const int Npoints = Psi.rows();
			dvector nx(Npoints, 0);

			for(int i=0; i<_N; ++i) {
				k[i] = i*_Width/(_N - 1.) - _Width/2.;
			}

			for(int i=0; i<Nev; ++i)
			{
				_FermiDiracDist dist(myyR, mzzR, Esub(i));
				double density = 2.*_integral(dist, k, k)/std::pow(2*M_PI, 2)*std::pow(1e7, 2); // [cm^-2]
				for(int j=0; j<Npoints; ++j) {
					nx[j] += density*std::pow(Psi(j, i), 2)*1e4; // [cm^-3]
				}
			}

			return std::move(nx);
		}

		const int _N; // discritizing for k points
		const double _Width; // Width of an integration range : [-_Width/2 ~ _Width/2]
		NUMERIC_CALCULUS::simpson_2d_method _integral; // functor to integrate out for a given fermi-dirac dist.
	};



	class SparseEigenSolver
	{
	using dvector = std::vector<double>;
	using SparseDoubleInt = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

	public:
		SparseEigenSolver(const dvector& x, const double scale, const double meffRatio)
		: _x(x), _Nx(x.size()), _scale(scale), _coeff(3.8099815e-8),
		  _H(x.size()-2, x.size()-2)
		{
			const double dx = (_x[_Nx - 1] - _x[0])/(_Nx - 1.);

			for(int i=0; i<_Nx-2; ++i) {
				_H.coeffRef(i, i) = -_coeff*(-2.)/(meffRatio*std::pow(_scale*dx, 2));
			}

			for(int i=0; i<_Nx-3; ++i) {
				_H.coeffRef(i, i+1) = -_coeff*(1.)/(meffRatio*std::pow(_scale*dx, 2));
				_H.coeffRef(i+1, i) = -_coeff*(1.)/(meffRatio*std::pow(_scale*dx, 2));
			}
	
			_H.makeCompressed();
		}

		void insert_V(const dvector& V)
		{
			for(int i=0; i<_Nx-2; ++i) {
				_H.coeffRef(i, i) += V[i];
			}
		}

		void compute(Eigen::VectorXd& energy, Eigen::MatrixXd& psi,
			     const int nev, const int ncv) const
		{
			EIGEN_SOLVER::EIGEN::SPECTRA::SymEigsSolver eigenSolver;
			// Solve eigen problem
			eigenSolver(_H, energy, psi, nev, ncv, true);
			// normalization
			_set_norm(psi, nev);
			// check sign of wave function
			_set_sign(psi, nev);
		}

	private:

		void _set_norm(Eigen::MatrixXd& psi, const int& nev) const
		{
			const double dx = (_x[_Nx-1] - _x[0])/(_Nx - 1.);
	
			std::vector<double> psiSquare(_Nx, 0);

			for(int j=0; j<nev; ++j) 
			{
				for(int i=1; i<_Nx-1; ++i) {
					psiSquare[i] = std::pow(psi(i-1, j), 2);
				}

				double norm = NUMERIC_CALCULUS::simpson_1D_method(&psiSquare[0], _Nx, dx);

				for(int i=1; i<_Nx-1; ++i) {
					psi(i-1, j) /= std::sqrt(norm);
				}
			}	
		}	

		void _set_sign(Eigen::MatrixXd& psi, const int& nev, const double sign = 1.) const
		{
			for(int j=0; j<nev; ++j)
			{
				if(psi(0, j)*sign < 0) 
				{
					for(int i=0; i<_Nx-2; ++i) {
						psi(i, j) *= -1;
					}
				}
			}
		}

		const int _Nx;
		const double _scale;
		const double _coeff;
		const dvector _x;

		SparseDoubleInt _H;
	};


	using sparseMatrix = LINEAR_SOLVER::EIGEN::SparseDouble;
	using denseVector  = LINEAR_SOLVER::EIGEN::nvector;
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

	protected:
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
}

void read_file(const char filename[], std::vector<double>& E)
{
	std::vector<double>().swap(E);

	std::ifstream infile(filename);

	while(true)
	{
		double temp;
		infile >> temp;
		if(infile.eof()) {
			break;
		}
		E.push_back(temp);
	}

	infile.close();
}


int main(int argc, char* argv[])
{
	std::vector<double> Esub0p91, Esub0p19;
	//read_file("eig_0p91.dat", Esub0p91);
	//read_file("eig_0p19.dat", Esub0p19);

	SCHRODINGER_POISSON::Number_density number(401, 4.);

	//double result = number.avg_density(Esub0p91, Esub0p19);

	//std::cout << "tot:" << result << std::endl;

	return 0;
}
