#include <iostream>
#include <Eigen/Sparse>
#include <vector>
#include <assert.h>
#include <functional>
#include <cmath>
#include <memory>


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



namespace DENSE_SOLVER
{
	namespace LAPACK
	{
		using ivector = std::vector<int>;
		using dvector = std::vector<double>;

		extern "C"
		{
		        void dgesv_ (const int* N, const int* NRHS, double* A, const int* LDA,
       	         	     	     int* IPIV, double* B, const int* LDB, int* INFO);
		}

		// solve a linear equation Ax = y
        	static auto const linear_solver = [](const dvector& A, dvector& y) -> void
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
        	};
	}
}



namespace SPARSE_SOLVER
{
	namespace EIGEN
	{
		using SparseDouble = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;
		using COLAMDO = Eigen::COLAMDOrdering<long long>;
		using nvector = Eigen::VectorXd;

		// For a symmetrc sparse matrix
		static auto const cholesky_solver = [](const SparseDouble& A, nvector& b) -> void
		{
			const int Ndim = b.size();
			const nvector x(b);

			Eigen::SimplicialCholesky<SparseDouble> solver(A);

			if(solver.info() != Eigen::Success) {
				std::cout<< "!Error : SPARSE_SOLVER::EIGEN::cholesky_solver"
					 << std::endl;
				assert(false);
			};

			b = solver.solve(x);
		};

		// For a general sparse matrix
		static auto const LU_solver = [](const SparseDouble& A, nvector& b) -> void
		{
			const int Ndim = b.size();
			const nvector x(b);

			Eigen::SparseLU<SparseDouble, COLAMDO> solver;

			solver.analyzePattern(A);

			solver.factorize(A);

			if(solver.info() != Eigen::Success) {
				std::cout<< "!Error : SPARSE_SOLVER::EIGEN::LU_solver"
					 << std::endl;
				assert(false);
			};

			b = solver.solve(x);
		};
	}
}



namespace ROOT_FINDING
{
	template<class VectorObj>
	class objectBase
	{
	public:
		explicit objectBase(const int Ndim) : _Ndim(Ndim) {}
		virtual ~objectBase() {}

		void residue(const VectorObj& root, VectorObj& f) const;

		virtual void jacobian(const VectorObj& root) = 0;

		int size() const {
			return _Ndim;
		}

	protected:
		const int _Ndim;

		virtual double _residual(const VectorObj& root, const int& i) const = 0;
	};

	template<class VectorObj>
	void objectBase<VectorObj>::residue(const VectorObj& root, VectorObj& f) const
	{
		for(int i=0; i<_Ndim; ++i) {
			f[i] = _residual(root, i);
		}
	}
	


	template<class EquationObj, class VectorObj, class SolverObj>
	bool newton_method(EquationObj& object, VectorObj& root, const SolverObj& solver,
			   const size_t niter = 100, const double tol = 1e-7)
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
			object.solve(up, solver);

			for(int i=0; i<Ndim; ++i) {
				root[i] -= up[i];
			}
		}

		return isConverge;
	}
}



using sparseMatrix = SPARSE_SOLVER::EIGEN::SparseDouble;
using denseVector  = SPARSE_SOLVER::EIGEN::nvector;
using EigenBase   = ROOT_FINDING::objectBase<denseVector>;


class sp_multi_charge_neutrality : public EigenBase
{
	using EigenBase::_Ndim;
public:
	explicit sp_multi_charge_neutrality(const int Ndim)
	: EigenBase(Ndim), _nplusArray(Ndim), _J(Ndim, Ndim) {}

	virtual ~sp_multi_charge_neutrality() {}

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

	void insert_Nplus(const std::vector<double>& nplusArray) {
		_nplusArray = nplusArray;
	}

	template<class SolverObj>
	void solve(denseVector& v, const SolverObj& solver) const {
		solver(_J, v);
	}

private:

	virtual double _residual(const denseVector& root, const int& i) const {
		return _ni*(std::exp(root[i]) - std::exp(-root[i]))/_nplusArray[i] - 1.;
	}

	const double _ni = 1.5e10; // silicon carrier density at T = 300k (1e10)
	std::vector<double> _nplusArray;
	sparseMatrix _J;
};


int main(int argc, char* argv[])
{
	const int Ndim = 1000;
	const double KbT = 300*8.617343e-5;
	std::vector<double> nplusArray(Ndim, 1*1e15);
	for(int i=0; i<nplusArray.size()/2; ++i) {
		nplusArray[i] *= -1;
	}

	std::shared_ptr<sp_multi_charge_neutrality> ptrObject(new sp_multi_charge_neutrality(Ndim));
	denseVector roots(Ndim);

	ptrObject->insert_Nplus(nplusArray);

	for(int i=0; i<Ndim; ++i) {
		roots[i] = 50.;
	}

	ROOT_FINDING::newton_method(*ptrObject, roots, SPARSE_SOLVER::EIGEN::LU_solver, 1000);

	std::cout<<KbT*roots[0]<<"(ev)"<<std::endl;
	std::cout<<KbT*roots[Ndim-1]<<"(ev)"<<std::endl;

	return 0;
}
