#include <iostream>
#include <vector>
#include <assert.h>
#include <numeric>
#include <cmath>
#include <cstring>
#include <fstream>


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


namespace ROOT_FINDING
{
	using dvector = std::vector<double>;

	class objectBase
	{
	public:
		explicit objectBase(const int Ndim) : _Ndim(Ndim) {}
		virtual ~objectBase() {}


		void residue(const dvector& root, dvector& f) const
		{
			for(int i=0; i<_Ndim; ++i) {
				f[i] = _residual(&root[0], i);
			}
		}

		virtual void jacobian(const dvector& root, dvector& J) const
       		{
	                const double h = 5e-8;
			dvector root_pdh(&root[0], &root[0] + _Ndim), root_mdh(&root[0], &root[0] + _Ndim);

       	 	        for(int i=0;i<_Ndim;++i)
			{
                        	for(int j=0;j<_Ndim;++j)
                        	{
                                	root_pdh[j] += h;
                                	root_mdh[j] -= h;

                                	J[i*_Ndim + j] = (_residual(&root_pdh[0], i) - _residual(&root_mdh[0], i))/(2.*h);

                                	root_pdh[j] -= h;
                                	root_mdh[j] += h;
                        	}
                	}
        	}

		int size() const {
			return _Ndim;
		}

	protected:
		const int _Ndim;

		virtual double _residual(const double* root, const int i) const = 0 ;
	};


	bool newton_method(const objectBase& object, dvector& root, const double tol = 1e-7, const size_t niter = 100)
	{
		assert(object.size() == root.size());

		const int Ndim = object.size();
		dvector J(Ndim*Ndim, 0), up(Ndim, 0), f(Ndim, 0);
		bool isConverge = false;

		for(size_t n=0; n<niter; ++n)
		{
			object.residue(root, f);

			double cost = std::sqrt(std::inner_product(f.begin(), f.end(), f.begin(), 0.));

			std::cout<<"iter: "<<n<<"\t\tcost: "<<cost<<std::endl;

			if(cost < tol) 
			{
				isConverge = true;
				break;
			}

			object.jacobian(root, J);

			std::memcpy(&up[0], &f[0], sizeof(double)*Ndim);

			LAPACK_SOLVER::linear_solver(J, up);

			for(int i=0; i<Ndim; ++i) {
				root[i] -= up[i];
			}
		}

		return isConverge;
	}
}


// (x+y)^2 - 3 = 0
// x - 2y - 4
class derivedObject : public ROOT_FINDING::objectBase 
{
public:
	explicit derivedObject() : objectBase(2) {}

private:
	virtual double _residual(const double* root, const int i) const
	{
		double result = 0;
		if(i == 0) {
			result = std::pow(root[0] + root[1], 2) - 3;
		}
		else if(i == 1) {
			result = root[0] - 2*root[1] - 4;
		}
		else {
			assert(false);
		}

		return result;
	}
};



int main(int argc, char* argv[])
{
	derivedObject object;
	std::vector<double> root = {1, 1};

	ROOT_FINDING::newton_method(object, root);

	for(auto const& r_i : root) {
		std::cout<<r_i<<" ";
	}

	std::cout<<std::endl;

	return 0;
}
