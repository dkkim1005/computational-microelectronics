#include <iostream>
#include <vector>
#include <cstring>
#include <cmath>

template<class T>
class MatAdaptorBase
{
public:
        MatAdaptorBase(T& Matrix)
        : _Matrix(Matrix) {}

        virtual double& operator()(const int i, const int j) = 0;
protected:
        T& _Matrix;
};


class stlVector : public MatAdaptorBase<std::vector<double> > 
{
public:
	stlVector(std::vector<double>& A)
	: MatAdaptorBase(A), _L(std::sqrt(A.size())) {}

        virtual double& operator()(const int i, const int j) {
		return _Matrix[_L*i + j];
	}

private:
	const int _L;
};


template<class T>
class MatBuilderBase
{
public:
        virtual T build(const int L) = 0;
};


class stlVectorBuild : public MatBuilderBase<std::vector<double> >
{
public:
        virtual std::vector<double> build(const int L) {
                return std::move(std::vector<double>(L*L, 0));
        }
};





int main(int argc, char* argv[])
{
	std::vector<double> A(4, 0);

	stlVector B(A);

	B(0, 0) = 1;
	B(0, 1) = 2;
	B(1, 0) = 3;
	B(1, 1) = 4;

	for(auto b : A) {
		std::cout << b << " ";
	}

	std::cout << std::endl;

	return 0;
}
