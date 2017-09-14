#include <iostream>
#include <Eigen/Sparse>
#include <vector>
#include <assert.h>
#include <functional>
#include <cmath>


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
	template<class JacobianObj, class VectorObj>
	class objectBase
	{
	public:
		explicit objectBase(const int Ndim) : _Ndim(Ndim) {}
		virtual ~objectBase() {}

		void residue(const VectorObj& root, VectorObj& f) const;

		virtual void jacobian(const VectorObj& root, JacobianObj& J) const = 0;

		int size() const {
			return _Ndim;
		}

	protected:
		const int _Ndim;

		virtual double _residual(const VectorObj& root, const int& i) const = 0;
	};

	template<class JacobianObj, class VectorObj>
	void objectBase<JacobianObj, VectorObj>::residue(const VectorObj& root, VectorObj& f) const
	{
		for(int i=0; i<_Ndim; ++i) {
			f[i] = _residual(root, i);
		}
	}
	


	template<class EquationObj, class JacobianObj, class VectorObj, class SolverObj>
	bool newton_method(const EquationObj& object, JacobianObj& J, VectorObj& root, const SolverObj& solver,
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

			//std::cout<<"  iter: "<<n<<"\t\tcost: "<<cost<<std::endl;

			if(cost < tol) 
			{
				isConverge = true;
				break;
			}

			object.jacobian(root, J);

			up = f;

			solver(J, up);

			for(int i=0; i<Ndim; ++i) {
				root[i] -= up[i];
			}
		}

		return isConverge;
	}
}


class xsquare_m_1 : public ROOT_FINDING::objectBase<std::vector<double>, std::vector<double> >  
{
	using ROOT_FINDING::objectBase<std::vector<double>, std::vector<double> >::_Ndim;
public:
	xsquare_m_1() :ROOT_FINDING::objectBase<std::vector<double>, std::vector<double> >(1) {}

        virtual void jacobian(const std::vector<double>& root, std::vector<double>& J) const
        {
                const double h = 5e-8;
                std::vector<double> root_pdh(root), root_mdh(root);

                for(int i=0;i<_Ndim;++i)
                {
                        for(int j=0;j<_Ndim;++j)
                        {
                                root_pdh[j] += h;
                                root_mdh[j] -= h;

                                J[i*_Ndim + j] = (_residual(root_pdh, i) - _residual(root_mdh, i))/(2.*h);

                                root_pdh[j] -= h;
                                root_mdh[j] += h;
                        }
                }
        }

private:
	virtual double _residual(const std::vector<double>& root, const int& i) const {
		return std::pow(root[0], 2) - 3.;
	}
};



class sp_xsquare_m_1 : public ROOT_FINDING::objectBase<SPARSE_SOLVER::EIGEN::SparseDouble, SPARSE_SOLVER::EIGEN::nvector>  
{
	using ROOT_FINDING::objectBase<SPARSE_SOLVER::EIGEN::SparseDouble, SPARSE_SOLVER::EIGEN::nvector>::_Ndim;
public:
	sp_xsquare_m_1() :ROOT_FINDING::objectBase<SPARSE_SOLVER::EIGEN::SparseDouble, SPARSE_SOLVER::EIGEN::nvector>(1) {}

        virtual void jacobian(const SPARSE_SOLVER::EIGEN::nvector& root, SPARSE_SOLVER::EIGEN::SparseDouble& J) const
        {
                const double h = 5e-8;
		const int i = 0;
                SPARSE_SOLVER::EIGEN::nvector root_pdh(root), root_mdh(root);

                root_pdh[i] += h;
                root_mdh[i] -= h;

                J.coeffRef(0, 0) = (_residual(root_pdh, i) - _residual(root_mdh, i))/(2.*h);
        }

private:
	virtual double _residual(const SPARSE_SOLVER::EIGEN::nvector& root, const int& i) const {
		return std::pow(root[0], 2) - 3.;
	}
};



int main(int argc, char* argv[])
{
	std::vector<double> x = {2}, J = {0};
	xsquare_m_1 object;

	ROOT_FINDING::newton_method(object, J, x, DENSE_SOLVER::LAPACK::linear_solver);

	std::cout<<x[0]<<std::endl;

	SPARSE_SOLVER::EIGEN::SparseDouble A(1, 1);
	A.coeffRef(0, 0) = 3;
	SPARSE_SOLVER::EIGEN::nvector root(1);

	root[0] = 1;

	sp_xsquare_m_1 sp_object;

	ROOT_FINDING::newton_method(sp_object, A, root, SPARSE_SOLVER::EIGEN::LU_solver);

	std::cout<<root<<std::endl;

	return 0;
}
