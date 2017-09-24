#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#include <fstream>
#include <cstdlib>
#include "sparse_solver.h"

// compile : g++ -o filename hw1.cpp -std=c++11 -I/usr/local/include/eigen3 -I/home/alice/Work/spectra/spectra-0.5.0/include

namespace NUMERIC_CALCULUS 
{
	template<class T1, class T2>
	T1 simpson_integration(const T1* psi, const int N, const T2 dx)
	{
		assert(N%2 == 1);

		T1 accum = 0;

		for(int i=1;i<N-1; i+=2) {
			accum += 4.*psi[i];
		}
		for(int i=2;i<N-2; i+=2) {
			accum += 2.*psi[i];
		}

		accum *= dx/3.;

		return accum;
	}
} 


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

			double norm = NUMERIC_CALCULUS::simpson_integration(&psiSquare[0], _Nx, dx);

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



int main(int argc, char* argv[])
{
	if(argc == 1)
	{
		std::cout<<"  -- options \n"
			 <<"       argv[1]: scale ([x] = scale*[micrometer])\n"
			 <<"       argv[2]: filename to read data for x and phi(x)\n";
		return -1;
	}

	// The number of width for discritization.
	constexpr double EcmEi = 0.56; // E_{c} - E_{i} = (E_{c} - E_{v})/2.
	constexpr int Nx = 1001, nev = 60;
	const double scale = std::atof(argv[1]);
	assert(Nx%2 == 1);
	std::vector<double> x(Nx, 0); // [micrometer]
	std::vector<double> V(Nx-2, 0);

	std::ifstream rfile(argv[2]);
	if(rfile.is_open())
	{
		std::cout<<"  --file: "<<std::string(argv[2])<<std::endl;
		double temp;
		rfile >> x[0]; rfile >> temp;
		for(int i=0; i<Nx-2; ++i)
		{
			rfile >> x[i+1];
			rfile >> V[i]; // q*phi(x)
			V[i] = -V[i] + EcmEi; // V(x) = -q*phi(x) + E_{c} - E_{i}
		}
		rfile >> x[Nx-1];

		rfile.close();
	}
	else {
		std::cout<<"  --there is no file to read: "<<std::string(argv[2])<<std::endl;
		std::abort();
	}

	// open a file object to record numeric solutions.
	std::ofstream outFile(("wave-" + std::string(argv[2])).c_str());
	assert(outFile.is_open());

	SparseEigenSolver infWallSolver(x, scale, 1.);

	Eigen::MatrixXd psi;
	Eigen::VectorXd energy;

	infWallSolver.insert_V(V);

	infWallSolver.compute(energy, psi, nev, Nx/10);

	// record numeric solution from the ground state to the N'th excited state.
	outFile << x[0] << " ";
	for(int i=0; i<nev; ++i) {
		outFile << 0. << " ";
	}
	outFile<<std::endl;

	for(int i=1; i<Nx-1; ++i)
	{
		outFile << x[i] << " ";
		for(int j=0; j<nev; ++j) {
			outFile << std::pow(psi(i-1, j), 2) << " ";
		}
		outFile<<std::endl;
	}

	outFile << x[Nx - 1] << " ";
	for(int i=0; i<nev; ++i) {
		outFile << 0. << " ";
	}

	outFile.close();

	std::cout<<"  --energy [ev]\n"<<energy<<std::endl;;
		
	return 0;
}
