#include <iostream>
#include <cmath>
#include <vector>
#include <cstring>
#include <assert.h>
#include <fstream>
#include <functional>


namespace LINALG
{
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
}


namespace LAPACK_SOLVER
{
	using ivector = std::vector<int>;
	using dvector = std::vector<double>;

	extern "C"
	{
	        void dgesv_ (const int* N, const int* NRHS, double* A, const int* LDA,
                	     int* IPIV, double* B, const int* LDB, int* INFO);
	}


	// solve a linear equation Ax = y
        void linear_solver(const dvector& A, dvector& y)
        {
		assert(A.size() == std::pow(y.size(), 2));
		const int N = y.size();

                dvector _A = A; 

                LINALG::transpose(_A); 

                ivector IPIV(N, 0); 

                const int NRHS = 1, LDA = N, LDB = N;
                int INFO = 0;

                dgesv_(&N, &NRHS, &_A[0], &LDA, &IPIV[0], &y[0], &LDB, &INFO);

		assert(INFO == 0);
        }
}


namespace
{
	template<class EPSFUNCTOR>
	void matrix_construction(const EPSFUNCTOR& eps,
				 std::vector<double>& A, std::vector<double>& x)
	{
		assert(A.size() == std::pow(x.size(), 2));

		const int N = std::sqrt(A.size());
		const double dx = (x[N-1] - x[0])/(N - 1.);

		A[0] = 1.;
		A[N*N - 1] = 1.;
		/*
			- eps(x_{i-0.5})pi(x_{i-1})/dx
			+ (eps(x_{i+0.5}) + eps(x_{i-0.5}))pi(x_{i})/dx
			- eps(x_{i+0.5})pi_{i+1}/dx = 0
		*/
		for(int i=1; i<N-1; ++i)
		{
			A[N*i + i-1] = -eps((x[i] + x[i-1])/2.)/dx;
			A[N*i + i]   = (eps((x[i+1] + x[i])/2.) +
					eps((x[i] + x[i-1])/2.))/dx;
			A[N*i + i+1] = -eps((x[i+1] + x[i])/2.)/dx;
		}
	}
}


int main(int argc, char* argv[])
{
	const int N = std::atoi(argv[1]);
	std::ofstream ofile(("pi-" + std::string(argv[2])).c_str());
	assert(ofile.is_open());

	const double eps_0 = 8.8541878*1e-12; // [C^2 / (m^2*N)]

	struct MaterialInfo
	{
		// tickness of the silicon and silicon dioxide
		const double T_Si = 3e-3, T_Di = 7e-3; //(m)
		// Relative permittivity(silicon, silicon dioxide)
		const double Er_Si = 11.68, Er_Di = 3.9; 
	};

	MaterialInfo info;

	const double T_TOT = info.T_Si + info.T_Di;

	std::vector<double> A(N*N, 0), pi(N, 0);

	// set boundary conditions
	pi[0] = 1; pi[N-1] = 5;

	// define a lambda function for the position dependent permittivity(xi: 0 to T_TOT)
	auto eps = [&info](const double xi) -> double
		{
			/*   silicon  | silicon dioxide
		 	x 0|---------T_Si---------------*/
			double result = 0;

			if(info.T_Si > xi) {
				result = info.Er_Si;
			}
			else {
				result = info.Er_Di;
			}
			return result;
		};


	// x : from 0 to T_TOT
	std::vector<double> x(N, 0);
	for(int i=0; i<N; ++i) {
		x[i] = i*T_TOT/(N - 1.);
	}

	::matrix_construction(eps, A, x);

	LAPACK_SOLVER::linear_solver(A, pi);

	// print result
	for(int i=0; i<N; ++i) {
		ofile<<x[i]<<" "<<pi[i]<<"\n";
	}

	ofile.close();

	ofile.open(("res-" + std::string(argv[2])).c_str());
	const double E_Si = (pi[1] - pi[0])/(x[1] - x[0]), E_Di = (pi[N-1] - pi[N-2])/(x[N-1] - x[N-2]);
	const double V_Si = E_Si*info.T_Si, V_Di = E_Di*info.T_Di;
	const double V = V_Si + V_Di;

	ofile << "  E(Si) : " << E_Si << " [N/(C*m^2)]" << std::endl
	      << "  E(Di) : " << E_Di << " [C/(C*m^2)]" << std::endl;

	const double Q = (E_Si*info.Er_Si + E_Di*info.Er_Di)*eps_0/2.;

	ofile << "  V(Si) : " << V_Si << " [N/(C*m)]" << std::endl
	      << "  V(Di) : " << V_Di << " [N/(C*m)]" << std::endl
	      << "  total V : " << V_Si + V_Di << " [N/(C*m)]" << std::endl
	      << "  total capacitance : " << Q/V << " [C/V*m^2]" << std::endl
	      << "-----------------------------" << std::endl
	      << "  In theory ::\n  total capacitance :"
	      << 1./(info.T_Si/info.Er_Si  + info.T_Di/info.Er_Di)*eps_0
	      << " [C/V*m^2]" << std::endl;

	return 0;
}
