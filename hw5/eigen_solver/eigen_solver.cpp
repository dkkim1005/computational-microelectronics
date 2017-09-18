#include <iostream>
#include <vector>
#include <cmath>
#include <assert.h>
#include <fstream>
#include <cstdlib>

// compile : g++ -o filename hw1.cpp -std=c++14 -lopenblas -llapack

#define MAX(a, b) (a>b ? a : b)
#define MIN(a, b) (a>b ? b : a)

namespace LAPACK_SOLVER
{

	using dvector = std::vector<double>;

	extern "C" 
	{
		void dsyev_(const char* JOBZ, const char* UPLO, const int* N, double* A,
		const int* LDA, double* W, double* WORK, const int* LWORK, int* INFO);
	}


	template<class T>
	void transpose(std::vector<T>& mat) 
	{
		const int N = std::sqrt(mat.size());
		for(int i=0; i<N; ++i)
		{
			for(int j=i+1; j<N; ++j) 
			{
				T temp = mat[i*N + j];
				mat[i*N + j] = mat[j*N + i];
				mat[j*N + i] = temp;
			}
		}
	}


	void diag(dvector& A, dvector& lambda)
	{
		assert(A.size() == std::pow(lambda.size(), 2));
		const char JOBZ = 'V';
		const char UPLO = 'U';
		const int N = lambda.size();
		const int LDA = MAX(1, N);
		dvector WORK(1);
		int LWORK = -1;
		int INFO = 0;

		dsyev_(&JOBZ, &UPLO, &N, &A[0], &N, &lambda[0], &WORK[0], &LWORK, &INFO);

		LWORK = MAX(1, WORK[0]);
		dvector(LWORK, 0).swap(WORK);

		dsyev_(&JOBZ, &UPLO, &N, &A[0], &N, &lambda[0], &WORK[0], &LWORK, &INFO);

		transpose(A);
	}
}


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


void set_sign_eigen(std::vector<double>& A, const double sign = 1.)
{
	const int N = std::sqrt(A.size());
	for(int j=0; j<N; ++j)
	{
		if(A[j]*sign < 0) 
		{
			for(int i=0; i<N; ++i) {
				A[N*i + j] *= -1;
			}
		}
	}
}


void set_norm(std::vector<double>& A, const double a)
{
	const int N = std::sqrt(A.size()) + 2;
	const double dx = a/(N-1.);
	std::vector<double> x(N);

	for(int i=0; i<N; ++i) {
		x[i] = i*dx;
	}

	std::vector<double> psiSquare(N, 0);

	for(int j=0; j<N-2; ++j) 
	{
		for(int i=1; i<N-1; ++i) {
			psiSquare[i] = std::pow(A[(N-2)*(i-1)+ j], 2);
		}

		double norm = NUMERIC_CALCULUS::simpson_integration(&psiSquare[0], N, dx);

		for(int i=1; i<N-1; ++i) {
			A[(N-2)*(i-1)+ j] /= std::sqrt(norm);
		}
	}
}


int main(int argc, char* argv[])
{
	if(argc == 1)
	{
		std::cout<<"  -- options \n"
			 <<"       argv[1]: N\n"
			 <<"       argv[2]: filename to read data for x and phi(x)\n"
			 <<"       argv[3]: filename to write a wave function\n";
		return -1;
	}

	// The number of width for discritization.
	const int N = std::atoi(argv[1]), Ncut = 100;
	assert(N%2 == 1);
	std::vector<double> x(N, 0); // [micrometer]
	std::vector<double> V(N-2, 0);

	std::ifstream rfile(argv[2]);
	if(rfile.is_open())
	{
		std::cout<<"  --file: "<<std::string(argv[2])<<std::endl;
		double temp;
		rfile >> x[0]; rfile >> temp;
		for(int i=0; i<N-2; ++i)
		{
			rfile >> x[i+1];
			rfile >> V[i];
		}
		rfile >> x[N-1];
	}
	else {
		std::cout<<"  --there is no file to read: "<<std::string(argv[2])<<std::endl;
		std::abort();
	}

	rfile.close();

	// open a file object to record numeric solutions.
	std::ofstream outFile(argv[3]);
	assert(outFile.is_open());
	constexpr double ev = 1.6021766208e-19;
	const double HBAR = 6.62607004e-34/(2*M_PI),
			M = 9.10938356e-31,
			a = x[N-1] - x[0];
	const double da = a/(N - 1.);
	const double coeff = -std::pow(HBAR, 2)/
			    (2.*M*std::pow(da, 2)*ev*1e-12); // [ev * m^-2]

	std::cout<<"coeff: "<<coeff<<std::endl;

	std::vector<double> A(std::pow(N-2, 2), 0), lambda(N-2);

	for(int i=0; i<N-2; ++i) {
		A[i*(N-2) + i] = -2.*coeff - V[i]; // V[i]: q*phi --> -q*phi (electron has negative charge!)
	}

	// LAPACK_SOLVER::diag solver only needs an upper matrix component.
	for(int i=0; i<N-3; ++i) {
		A[(i+1)*(N-2) + i] = 1.*coeff;
	}

	LAPACK_SOLVER::diag(A, lambda);

	set_sign_eigen(A);

	// nomalizaing the wave functions
	set_norm(A, a);


	// record numeric solution from the ground state to the N'th excited state.
	outFile << x[0] << " ";
	for(int i=0; i<MIN(N, Ncut)-2; ++i) {
		outFile << 0. << " ";
	}
	outFile<<std::endl;


	for(int i=1; i<N-1; ++i)
	{
		outFile << x[i] << " ";
		for(int j=0; j<MIN(N, Ncut)-2; ++j) {
			outFile << A[(N-2)*(i-1) + j] << " ";
		}
		outFile<<std::endl;
	}

	outFile << x[N - 1] << " ";
	for(int i=0; i<MIN(N, Ncut)-2; ++i) {
		outFile << 0. << " ";
	}

	outFile.close();
	
	return 0;
}
