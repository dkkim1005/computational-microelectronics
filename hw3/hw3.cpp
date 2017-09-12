#include <iostream>
#include <vector>
#include <assert.h>
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

	template<class T>
	T inner_product(const std::vector<T>& v1, const std::vector<T>& v2, const int& N)
	{
		T result = 0;
		for(int i=0; i<N; ++i) {
			result += v1[i]*v2[i];
		}
		return result;
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


		void residue(const dvector& root, dvector& f) const;

		virtual void jacobian(const dvector& root, dvector& J) const;

		int size() const {
			return _Ndim;
		}

	protected:
		const int _Ndim;

		virtual double _residual(const dvector& root, const int& i) const = 0 ;
	};

	void objectBase::residue(const dvector& root, dvector& f) const
	{
		for(int i=0; i<_Ndim; ++i) {
			f[i] = _residual(root, i);
		}
	}
	
	void objectBase::jacobian(const dvector& root, dvector& J) const
       	{
                const double h = 5e-8;
		dvector root_pdh(&root[0], &root[0] + _Ndim), root_mdh(&root[0], &root[0] + _Ndim);

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



	bool newton_method(const objectBase& object, dvector& root, const size_t niter = 100, const double tol = 1e-7)
	{
		assert(object.size() == root.size());

		const int Ndim = object.size();
		dvector J(Ndim*Ndim, 0), up(Ndim, 0), f(Ndim, 0);
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

			LAPACK_SOLVER::linear_solver(J, up);

			for(int i=0; i<Ndim; ++i) {
				root[i] -= up[i];
			}
		}

		return isConverge;
	}
}



class xsquare_m_1 : public ROOT_FINDING::objectBase  
{
public:
	xsquare_m_1() :ROOT_FINDING::objectBase(1) {}
private:
	virtual double _residual(const std::vector<double>& root, const int& i) const {
		return std::pow(root[0], 2) - 1.;
	}
};



class charge_neutrality : public ROOT_FINDING::objectBase
{
public:
	charge_neutrality()
	: ROOT_FINDING::objectBase(1) {}

	void insert_Nplus(const double& Nplus) {
		_Nplus = Nplus;
	}

private:
	virtual double _residual(const std::vector<double>& root, const int& i) const {
		return (_ni*(std::exp(root[0]) - std::exp(-root[0]))/_Nplus) - 1.;
	}

	const double _ni = 1.5e10; // silicon carrier density at T = 300k
	double _Nplus = 0;
};



class multi_charge_neutrality : public ROOT_FINDING::objectBase
{
public:
	explicit multi_charge_neutrality(const int Ndim)
	: ROOT_FINDING::objectBase(Ndim), _nplusArray(Ndim, 0) {}

	void insert_Nplus(const std::vector<double>& nplusArray) {
		_nplusArray = nplusArray;
	}

private:
	virtual double _residual(const std::vector<double>& root, const int& i) const {
		return (_ni*(std::exp(root[i]) - std::exp(-root[i]))/_nplusArray[i]) - 1.;
	}

	const double _ni = 1.5e10; // silicon carrier density at T = 300k (1e10)
	std::vector<double> _nplusArray;
};



int main(int argc, char* argv[])
{
	xsquare_m_1 x_eq_object;
	charge_neutrality charge_eq;
	multi_charge_neutrality multi_eq(1000);

	const double initialPoint = 20.;

	std::vector<double> rootXsq = {2.}, rootCharge = {initialPoint}, rootMulti(multi_eq.size(), initialPoint);

	ROOT_FINDING::newton_method(x_eq_object, rootXsq);
	std::cout<<"  --root(starting: +2): "<<rootXsq[0]<<std::endl;

	rootXsq[0] = -2;

	ROOT_FINDING::newton_method(x_eq_object, rootXsq);
	std::cout<<"  --root(starting: -2): "<<rootXsq[0]<<std::endl;

	const double _KT = 4.1419464e-21; // [J] 

	for(auto const& nplus : {1e15, 1e16, 1e17, 1e18, 1e19, 1e20})
	{
		charge_eq.insert_Nplus(nplus);
		ROOT_FINDING::newton_method(charge_eq, rootCharge);
		std::cout<<"  --root(N+ :"<<nplus<<"): "<<rootCharge[0]*_KT<<"(J)"<<std::endl;
		rootCharge[0] = initialPoint;
	}

	std::vector<double> nplusArr(multi_eq.size(), 0);
	for(int i=0; i<nplusArr.size(); ++i) {
		nplusArr[i] = (1e20 - 1e15)/(nplusArr.size() - 1.)*i + 1e15;
	}

	multi_eq.insert_Nplus(nplusArr);
	
	ROOT_FINDING::newton_method(multi_eq, rootMulti);

	std::ofstream outFile("pi-Nplus.dat");

	for(int i=0; i<nplusArr.size(); ++i) {
		outFile << nplusArr[i] << "\t" << rootMulti[i]*_KT << std::endl;
	}

	outFile.close();

	return 0;
}
