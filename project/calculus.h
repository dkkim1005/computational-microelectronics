#ifndef __NUMERIC_CALCULUS__
#define __NUMERIC_CALCULUS__

#include <iostream>
#include <vector>
#include <cstdlib>
#include <assert.h>

namespace NUMERIC_CALCULUS
{
	double simpson_1D_method(const double* psi, const int N, const double dx)
	{
		assert(N%2 == 1);

		double accum = 0;

		for(int i=1;i<N-1; i+=2) {
			accum += 4.*psi[i];
		}
		for(int i=2;i<N-2; i+=2) {
			accum += 2.*psi[i];
		}

		accum *= dx/3.;

		return accum;
	}

	double Trapezoidal_1D(const double* psi, const double* x, const int N)
	{
		assert(N%2 == 1);

		double accum = 0;

		for(int i=0; i<N-1; ++i) {
			accum += (psi[i+1] + psi[i])/2.*(x[i+1] - x[i]);
		}

		return accum;
	}

	// Ref : http://www.physics.usyd.edu.au/teach_res/mp/doc/math_integration_2D.pdf
	using dvector = std::vector<double>;

	class simpson_2d_method
	{

	public:
		explicit simpson_2d_method(const int N);

		template<class FUNCTOR>
		double operator()(const FUNCTOR& func, const dvector& x, const dvector& y) const;
	private:
		dvector _gen_S_matrix(const int N) const;

		const int _N;
		const dvector _S;
	};

	simpson_2d_method::simpson_2d_method(const int N)
	: _N(N), _S(_gen_S_matrix(N)) {}

	dvector simpson_2d_method::_gen_S_matrix(const int N) const
	{
		dvector S(N*N, 0);

		try
		{
			if(N%2 != 1 || N < 5) {
				throw "  !Error!:  \n  NUMERIC_CALCULUS::_gen_S_matrix: N%2 != 1 or N < 5...\n";
			}
		}
		catch(const char errmsg[])
		{
			std::cout<<errmsg;
			std::abort();
		}

		const int na = (N-1)/2;

		for(int i=0; i<na; ++i)
		{
			for(int j=0; j<na; ++j) {
				S[N+1 + 2*N*i + 2*j] = 16;
			}
		}

		for(int i=0; i<na; ++i)
		{
			for(int j=0; j<na-1; ++j)
			{
				S[N + 2 + 2*N*i + 2*j] = 8;
				S[2*N + 1 + 2*N*j + 2*i] = 8;
			}
		}

		for(int i=0; i<na-1; ++i)
		{
			for(int j=0; j<na-1; ++j) {
				S[2*N + 2 + 2*N*i + 2*j] = 4;
			}
		}

		for(int j=1; j<N-1; ++j)
		{
			if(j & 1 == 1)
			{
				S[j] = 4;
				S[j + N*(N-1)] = 4;
				S[N*j] = 4;
				S[N*j + N - 1] = 4;
			}
			else
			{
				S[j] = 2;
				S[j + N*(N-1)] = 2;
				S[N*j] = 2;
				S[N*j + N - 1] = 2;
			}
		}

		S[0] = 1;
		S[N-1] = 1;
		S[(N-1)*N] = 1;
		S[N*N - 1] = 1;

		return std::move(S);
	}

	template<class FUNCTOR>
	double simpson_2d_method::operator()(const FUNCTOR& func, const dvector& x, const dvector& y) const
	{
		try
		{
			if(x.size() != y.size() || x.size()%2 == 0) {
				throw "  !Error:  \n  two_dimensional_Simpson: x.size() != y.size() || x.size()%2 == 0...\n";
			}
			else if(x.size()*y.size() != _N*_N) {
				throw "  !Error:  \n  two_dimensional_Simpson: x.size()*y.size() != _N*_N...\n";
			}
		}
		catch(const char errmsg[])
		{
			std::cout << errmsg;
			std::abort();
		}

		dvector F(_N*_N, 0);

		const double hx = (x[_N-1] - x[0])/(_N - 1.), hy = (y[_N-1] - y[0])/(_N - 1.);

		for(int i=0; i<_N; ++i)
		{
			for(int j=0; j<_N; ++j) {
				F[_N*i + j] = func(x[i], y[j]);
			}
		}

		double traceSum = 0;

		for(int i=0; i<_N*_N; ++i) {
			traceSum += _S[i]*F[i];
		}

		traceSum *= hx*hy/9.;

		return traceSum;
	}
}


#endif
