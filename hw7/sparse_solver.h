#ifndef EIGEN_SPARSESOLVER_HEADER
#define EIGEN_SPARSESOLVER_HEADER

#include <Eigen/Sparse>
#include <SymEigsSolver.h>
#include <MatOp/SparseSymMatProd.h>
#include <iostream>
#include <cstdlib>

namespace EIGEN_SOLVER
{
  namespace EIGEN
  {
    using SparseDoubleInt = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

    namespace SPECTRA
    {
      class SymEigsSolver
      {
      public:
        SymEigsSolver() {}
	void operator()(const SparseDoubleInt& A, Eigen::VectorXd& evalues, Eigen::MatrixXd& evectors,
			const int nev, const int ncv, const bool info = false) const
	{
		/* ncv must satisfy nev < ncv <= n, n is the size of matrix
		   This parameter must satisfy nev<ncv<=n, and is advised to take ncv>=2nev.*/

		// Construct matrix operation object using the wrapper class SparseGenMatProd
		Spectra::SparseSymMatProd<double> op(A);

		// Construct eigen solver object, requesting the smallest eigenvalues
		Spectra::SymEigsSolver<double, Spectra::SMALLEST_ALGE, Spectra::SparseSymMatProd<double> > eigs(&op, nev, ncv);

		eigs.init();

		int nconv = eigs.compute();

		if(info) {
			std::cout << "  --converged set: " << nconv << " (nev: " << nev << ")\n";
		}

		if(eigs.info() != Spectra::SUCCESSFUL)
		{
			std::cout << "   !Error: EIGEN_SOLVER::EIGEN::SPECTRA::SymEigsSolver\n"
				  << "           Solver fails to gain results...\n";
			std::abort();
		}

		evalues = eigs.eigenvalues();
		evectors = eigs.eigenvectors(nev);
	}
      };
    }
  }
}


namespace LINEAR_SOLVER
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
					std::cout<< "!Error : LINEAR_SOLVER::EIGEN::CholeskyDecompSolver"
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
					std::cout<< "!Error : LINEAR_SOLVER::EIGEN::LUdecompSolver"
						 << std::endl;
					assert(false);
				}

				b = solver.solve(x);
			}
		};

	}
}


#endif
