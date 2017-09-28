#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <functional>
#include <fstream>


namespace NUMERIC_CALCULUS
{

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



namespace SCHRODINGER_POISSON
{
class Number_density
{
public:
	
	explicit Number_density(const int N, const double Width)
	: _N(N), _Width(Width), _integral(N)
	{}


	double avg_density(const std::vector<double>& Esub0p91, const std::vector<double>& Esub0p19) const
	{
		const double nsub1 = _n_sub_band(Esub0p91, 0.19, 0.19);
		const double nsub2 = _n_sub_band(Esub0p19, 0.91, 0.19);
		const double result = (2*nsub1 + 4*nsub2)*1e4; // [cm^-3]

		return result;
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


	double _n_sub_band(const std::vector<double>& Esub, const double myyR, const double mzzR) const
	{
		std::vector<double> k(_N, 0);

		double density = 0;

		for(int i=0; i<_N; ++i) {
			k[i] = i*_Width/(_N - 1.) - _Width/2.;
		}

		for(auto const& E : Esub)
		{
			_FermiDiracDist dist(myyR, mzzR, E);
			density += 2.*_integral(dist, k, k)/std::pow(2*M_PI, 2)*std::pow(1e7, 2); // [cm^-2]
		}

		return density;
	}

	const int _N; // discritizing for k points
	const double _Width; // Width of an integration range : [-_Width/2 ~ _Width/2]
	NUMERIC_CALCULUS::simpson_2d_method _integral; // functor to integrate out for a given fermi-dirac dist.
};


}

void read_file(const char filename[], std::vector<double>& E)
{
	std::vector<double>().swap(E);

	std::ifstream infile(filename);

	while(true)
	{
		double temp;
		infile >> temp;
		if(infile.eof()) {
			break;
		}
		E.push_back(temp);
	}

	infile.close();
}


int main(int argc, char* argv[])
{
	std::vector<double> Esub0p91, Esub0p19;
	read_file("eig_0p91.dat", Esub0p91);
	read_file("eig_0p19.dat", Esub0p19);

	SCHRODINGER_POISSON::Number_density number(401, 4.);

	double result = number.avg_density(Esub0p91, Esub0p19);

	std::cout << "tot:" << result << std::endl;

	return 0;
}
