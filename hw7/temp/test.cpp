#include <vector>
#include <list>
#include <fstream>
#include "calculus.h"

using dvector = std::vector<double>;

inline bool read_file(const char filename[], dvector& x, dvector& y)
{
	/*
		x_{0}	E_{0}
		x_{1}	E_{1}
		  .       .
		  .       .
		  .       .
		x_{n}	E_{n-1}
	*/

	std::list<double> tempListx;
	std::list<double> tempListy;
	std::ifstream infile(filename);

	if(!infile.is_open()) {
		return false;
	}

	while(true)
	{
		double tempx, tempy;
		infile >> tempx;
		infile >> tempy;
		if(infile.eof()) {
			break;
		}
		tempListx.push_back(tempx);
		tempListy.push_back(tempy);
	}

	infile.close();

	dvector(tempListx.size()).swap(x);
	dvector(tempListy.size()).swap(y);

	int i = 0;
	for(auto const& temp: tempListx)
	{
		x[i] = temp;
		i += 1;
	}

	int j = 0;
	for(auto const& temp: tempListy)
	{
		y[j] = temp;
		j += 1;
	}

	return true;
}


int main(int argc, char* argv[])
{
	std::vector<double> x, E;
	assert(read_file("density.dat", x, E));

	const int N = x.size();
	const double dx = (x[N-1] - x[0])/(N - 1.);

	double result = NUMERIC_CALCULUS::simpson_1D_method(&x[0], x.size(), dx)*1e-4;

	std::cout << result << std::endl;

	return 0;
}
