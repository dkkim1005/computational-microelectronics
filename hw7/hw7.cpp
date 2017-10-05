#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <list>
#include <functional>
#include <fstream>
#include <Eigen/Core>
#include "calculus.h"
#include "sparse_solver.h"
#include "root_finding.h"


namespace SCHRODINGER_POISSON
{

	class Si_Number_density
	{

	using dvector = std::vector<double>;

	public:
	
		Si_Number_density(const int N, const double Width)
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
				std::cout << "  !Error : SCHRODINGER_POISSON::Si_Number_density::density\n"
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
		dvector _n_sub_band(const Eigen::VectorXd& Esub, const Eigen::MatrixXd& Psi,
				    const double myyR, const double mzzR) const
		{
			assert(Esub.size() == Psi.cols());

			const int Npoints = Psi.rows() + 2, nev = Esub.size();
			dvector k(_N, 0), nx(Npoints, 0);

			for(int i=0; i<_N; ++i) {
				k[i] = i*_Width/(_N - 1.) - _Width/2.;
			}

			for(int j=0; j<nev; ++j)
			{
				_FermiDiracDist dist(myyR, mzzR, Esub(j));
				double density = 2.*_integral(dist, k, k)/std::pow(2*M_PI, 2)*std::pow(1e7, 2); // [cm^-2]
				for(int i=1; i<Npoints-1; ++i) {
					nx[i] += density*std::pow(Psi(i-1, j), 2)*1e4; // [cm^-3]
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
		  _H(x.size()-2, x.size()-2), _V(x.size()-2, 0)
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
			assert(V.size() == _V.size());
			for(int i=0; i<_Nx-2; ++i) {
				_H.coeffRef(i, i) -= _V[i]; // remove a past one
				_H.coeffRef(i, i) += V[i]; // update a new one
			}

			_V = V; // update history
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
		const double _scale; // scale factor for the _x : _scale*_x [micrometer]
		const double _coeff; // hbar^2/2m0 : [ev*micrometer^2]
		const dvector _x; // [micrometer]
		dvector _V; // history of the applied potential[ev]

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
	class Si_Poisson_Equation : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
	{
		using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
		using dvector = std::vector<double>;
	public:
		Si_Poisson_Equation(const int Ndim, const dvector& dopping,
				const dvector& x, const dvector& boundaries, const double scale = 1.0)
		: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	 	_J(Ndim, Ndim), _dopping(dopping), _x(x), _psi0(boundaries), _scale(scale)
		{
			assert(_Ndim == _dopping.size());
			assert(_Ndim + 2 == _x.size());
		}

		virtual ~Si_Poisson_Equation() {}

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
	void Si_Poisson_Equation<PermittivityObj>::jacobian(const denseVector& psi)
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
	double Si_Poisson_Equation<PermittivityObj>::_residual(const denseVector& psi, const int& i) const
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


	// Poisson-Schrodinger combined solver
	template<class PermittivityObj>
	class Si_Schrodinger_Poisson_Equation : public ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>
	{
		using ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>::_Ndim;
		using dvector = std::vector<double>;
	public:
		Si_Schrodinger_Poisson_Equation(const int Ndim, const dvector& dopping,
					   const dvector& x, const dvector& boundaries,
					   const dvector& ndensity, const double scale = 1.0)
		: ROOT_FINDING::ResidualBase<sparseMatrix, denseVector>(Ndim),
	 	_J(Ndim, Ndim), _dopping(dopping), _x(x),
		_psi0(boundaries), _nRatio(ndensity), _scale(scale)
		{
			assert(_Ndim == _dopping.size());
			assert(_Ndim + 2 == _x.size());
			assert(_Ndim + 2 == _nRatio.size());

			for(auto& _n : _nRatio) {
				_n /= 1.5*1e10; // n_i : 1.5*1e10 [cm^-3]
			}
		}

		virtual ~Si_Schrodinger_Poisson_Equation() {}

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
		std::vector<double> _nRatio; // ratio: n/n_i where n is electron density obtained from the schrodinger eq [dimensionless]
		// psi := q*phi/KbT
		std::vector<double> _psi0;    // boundary conditions at _x[0] and _x[_Ndim-1]
		const PermittivityObj _epsf;  // relative permittivity
		bool _isUpdated = false;
	};


	template<class PermittivityObj>
	void Si_Schrodinger_Poisson_Equation<PermittivityObj>::jacobian(const denseVector& psi)
	{
		for(int i=0; i<_Ndim; ++i)
		{
			double &x_ip1 = _x[i+2], &x_i = _x[i+1], &x_im1 = _x[i];
		
			/*
			   _nRatio[n+1] : the ratio for the electron density with
					  intrinsic carrier density at the 'n' site
			*/
			_J.coeffRef(i, i) = _epsf((x_ip1+x_i)/2.)/(x_ip1 - x_i) +
					    _epsf((x_i+x_im1)/2.)/(x_i-x_im1) +
					    std::pow(_scale, 2)*_coeff*((x_ip1+x_i)/2. - (x_i+x_im1)/2.)*
				    	    (_nRatio[i+1] + std::exp(-psi[i]));
		
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
	double Si_Schrodinger_Poisson_Equation<PermittivityObj>::_residual(const denseVector& psi, const int& i) const
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

		/*
		   _nRatio[n+1] : the ratio for the electron density with
				  intrinsic carrier density at the 'n' site
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
		 	  std::pow(_scale, 2)*_coeff*((_nRatio[i+1] - std::exp(-psi_i)) - _dopping[i])*(x_i - x_im1);

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

	
	inline std::vector<double> convert_psi_to_V(const Eigen::VectorXd& psi)
	{
		const int N = psi.size();
		std::vector<double> V(N);
		constexpr double KbT = 0.025851984732130292; //[ev]
		constexpr double EcmEi = 0.56; // E_{c} - E_{i} [ev]
		for(int i=0; i<N; ++i) {
			V[i] = -psi(i)*KbT + EcmEi;
		}
		return std::move(V);
	}
	
}


namespace TEMPORARY
{
	inline bool read_2d_file(const char filename[],
			         std::vector<double>& x, std::vector<double>& y)
	{
		/*
			x_{0}	y_{0}
			x_{1}	y_{1}
			  .       .
			  .       .
			  .       .
			x_{n-1}	y_{n-1}
		*/

		std::list<double> tempListx;
		std::list<double> tempListy;

		std::ifstream infile(filename);

		if(!infile.is_open()) {
			return false;
		}

		while(true)
		{
			double tempx, tempy;
			infile >> tempx;
			infile >> tempy;

			if(infile.eof()) {
				break;
			}
			tempListx.push_back(tempx);
			tempListy.push_back(tempy);
		}

		infile.close();

		assert(tempListx.size() == tempListy.size());

		const int N = tempListx.size();

		if(N != x.size()) {
			std::vector<double>(N).swap(x);
		}
		if(N != y.size()) {
			std::vector<double>(N).swap(y);
		}

		int i = 0;
		for(auto const& temp: tempListx)
		{
			x[i] = temp;
			i += 1;
		}

		int j = 0;
		for(auto const& temp: tempListy)
		{
			y[j] = temp;
			j += 1;
		}

		return true;
	}


	void write_2d_file(const char filename[], const std::vector<double>& x, const std::vector<double>& y)
	{
		std::ofstream wfile(filename);
		assert(x.size() == y.size());

		const int N = x.size();
	
		for(int i=0; i<N; ++i) {
			wfile << x[i] << "\t" << y[i] << "\n";
		}
	
		wfile.close();
	}


	void psi_read(const char filename[], Eigen::VectorXd& psi) 
	{
		std::vector<double> readX, readY;
		bool info = read_2d_file(filename, readX, readY);

		assert(info & psi.size() == readY.size()-2);

		for(int i=0; i<psi.size(); ++i) {
			psi(i) = readY[i+1];
		}
	}

	template<class T1, class T2>
	double delta_norm(const T1& v1, const T2& v2)
	{
		assert(v1.size() == v2.size());
		const int N = v1.size();
		double accum = 0;
		for(int i=0; i<N; ++i) {
			accum += std::pow(v1[i] - v2[i], 2);
		}
		return std::sqrt(accum);
	}
}


int main(int argc, char* argv[])
{
	using dvector = std::vector<double>;

	if(argc == 1)
	{
                std::cout<<"  -- options \n"
                         <<"       argv[1]: q*phi_{s} [ev]\n"
                         <<"       argv[2]: scale ([x] = scale*[micrometer])\n"
			 <<"       argv[3]: file to read initial phi (optional)\n";
                return -1; 
	}

	constexpr int Npoints = 1001;
	constexpr double Tsi = 1.; // [micrometer]
	constexpr double KbT = 0.025851984732130292; //[ev]
	const double qphis = std::atof(argv[1]); // [ev]
	const double scale = std::atof(argv[2]), m0p91R = 0.91, m0p19R = 0.19;
	const int nev = 100, ncv = Npoints/2; // ARPACK parameter
	const int niter = 100; // the # of the iteration for a loop
	const double tols = 1e-7;
	
	dvector x(Npoints), dopping(Npoints-2, -1e5/1.5);

	for(int i=0; i<Npoints; ++i) {
		x[i] = i*Tsi/(Npoints - 1.);
	}

	dvector psin(1, -10); // q*phi/KbT [dimensionless]

	std::cout << "\n   !    Set initial starting points\n" << std::endl;
	std::cout << "   -- Compute charge neutrality equation" << std::endl << std::flush;
	SCHRODINGER_POISSON::chargeNeutrality neutral;
	neutral.insert_dopping(dopping);

	ROOT_FINDING::newton_method(neutral, psin, [](const dvector& J, dvector& x)->void{x[0] /= J[0];});

	// boundary conditions
	dvector psiBound = {qphis/KbT, psin[0]};
	std::cout << "   -- Boundary conditions :\n"
		  << "      q0*phi(x_{0}): " << psiBound[0]*KbT << " [ev]"
		  << ",   q0*phi(x_{N-1}): " << psiBound[1]*KbT << " [ev]"
		  << std::endl << std::flush;


	Eigen::VectorXd psi(Npoints - 2);

	if(argc >= 4)
	{
		std::ifstream rFile(argv[3]);
		if(rFile.is_open()) {
			TEMPORARY::psi_read(argv[3], psi);
		}
		else
		{
			std::cout << "   !Error: we do not find the file to read a psi: " << argv[3] << std::endl;
			std::abort();
		}
		std::cout << "   -- file: " << argv[3] << std::endl;
	}
	else
	{
		SCHRODINGER_POISSON::Si_Poisson_Equation<SCHRODINGER_POISSON::permittivityForSilicon>
			classical_poisson (x.size() - 2, dopping, x, psiBound, scale);

		double dpsi = (psiBound[1] - psiBound[0])/(Npoints - 1.);
		for(int i=0; i<Npoints-2; ++i) {
			psi(i) = psiBound[0] + (i+1)*dpsi;
		}
		std::cout << "   -- Default: we set initial psi on a linear line" << std::endl;

		ROOT_FINDING::newton_method(classical_poisson, psi, LINEAR_SOLVER::EIGEN::CholeskyDecompSolver());
	}

	SCHRODINGER_POISSON::SparseEigenSolver eigen_solver_m0p91R(x, scale, m0p91R);
	SCHRODINGER_POISSON::SparseEigenSolver eigen_solver_m0p19R(x, scale, m0p19R);

	dvector V = SCHRODINGER_POISSON::convert_psi_to_V(psi);

	eigen_solver_m0p91R.insert_V(V);
	eigen_solver_m0p19R.insert_V(V);

	Eigen::MatrixXd waveFunc_m0p91R;
	Eigen::MatrixXd waveFunc_m0p19R;
	Eigen::VectorXd energy_m0p91R;
	Eigen::VectorXd energy_m0p19R;

	eigen_solver_m0p91R.compute(energy_m0p91R, waveFunc_m0p91R, nev, ncv);
	eigen_solver_m0p19R.compute(energy_m0p19R, waveFunc_m0p19R, nev, ncv);

	SCHRODINGER_POISSON::Si_Number_density densityIntegrator(1001, 5.);

	dvector density = densityIntegrator.density(energy_m0p91R, energy_m0p91R,
			          waveFunc_m0p19R, waveFunc_m0p19R); // [cm^-3]


	using SEMICLASSICAL_EQUATION =
		SCHRODINGER_POISSON::Si_Schrodinger_Poisson_Equation<SCHRODINGER_POISSON::permittivityForSilicon>;

	std::cout << "\n\n   ####### Start self-consistent loop! #######\n\n" << std::flush;

	for(int i=1; i<=niter; ++i)
	{
		std::cout << "\n   -- The # of iterations: " << i << std::endl << std::flush;

		SEMICLASSICAL_EQUATION schPoieq(Npoints-2, dopping, x, psiBound, density, scale);

		std::cout << "   -- SEMICLASSICAL-EQUATION:" << std::endl << std::flush;

		ROOT_FINDING::newton_method(schPoieq, psi, LINEAR_SOLVER::EIGEN::CholeskyDecompSolver(), 100000);

	
		V = SCHRODINGER_POISSON::convert_psi_to_V(psi);

		eigen_solver_m0p91R.insert_V(V);
		eigen_solver_m0p19R.insert_V(V);

		std::cout << "   -- EigenSolver with the given potential 'V(x)':" << std::endl << std::flush;
		eigen_solver_m0p91R.compute(energy_m0p91R, waveFunc_m0p91R, nev, ncv);
		eigen_solver_m0p19R.compute(energy_m0p19R, waveFunc_m0p19R, nev, ncv);

		auto density_hist = density;

		std::cout << "   -- K-points integration : " << std::endl << std::flush;
		density = densityIntegrator.density(energy_m0p91R, energy_m0p91R,
				          waveFunc_m0p19R, waveFunc_m0p19R); // [cm^-3]

		double delta_density = TEMPORARY::delta_norm(density_hist, density)/
				       std::sqrt(LINALG::inner_product(density, density, Npoints));

		std::cout << "      |n_{i} - n_{i+1}|/|n_{i+1}|: " << delta_density << std::endl;

		if(delta_density < tols)
		{
			std::cout << "\n\n   *-- converge! --* \n\n" << std::endl;
			break;
		}
	}

	TEMPORARY::write_2d_file(("density_" + std::string(argv[1]) + ".dat").c_str(), x, density);

	std::vector<double> psiw(Npoints, 0);
	psiw[0] = psiBound[0];
	psiw[Npoints-1] = psiBound[1];
	for(int i=0; i<Npoints-2; ++i) {
		psiw[i+1] = psi(i);
	}

	TEMPORARY::write_2d_file(("psi_" + std::string(argv[1]) + ".dat").c_str(), x, psiw);

	return 0;
}
