#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <fstream>

namespace
{
	using dcomplex = std::complex<double>;
	static const dcomplex IMGPOLE = dcomplex(0, 1e-30);

	double bernoulli(const double x) {
		return ((x+IMGPOLE)/(std::exp(x+IMGPOLE) - 1.)).real();
	}

	double Dbernoulli(const double x)
	{
		return ((std::exp(x+IMGPOLE) - 1. - (x+IMGPOLE)*std::exp(x+IMGPOLE))/
			std::pow((std::exp(x+IMGPOLE)-1.), 2)).real();
	}

	std::vector<double> linspace(const double init, const double fin, const size_t size)
	{
		std::vector<double> arr(size, 0);

		const double dx = (fin - init)/(size - 1.);

		for(int i=0; i<size; ++i) {
			arr[i] = init + dx*i;
		}

		return std::move(arr);
	}
}



int main(int argc, char* argv[])
{
	auto x = ::linspace(-10, 10, 1001);

	std::ofstream wfile1("bernoulli.dat");
	std::ofstream wfile2("Dbernoulli.dat");

	for(int i=0; i<x.size(); ++i)
	{
		wfile1 << x[i] << "\t" << ::bernoulli(x[i]) << std::endl;
		wfile2 << x[i] << "\t" << ::Dbernoulli(x[i]) << std::endl;
	}

	wfile1.close();
	wfile2.close();

	return 0;
}
