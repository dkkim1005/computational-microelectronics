#include <iostream>
#include <vector>
#include <list>
#include <functional>
#include <cmath>
#include <fstream>
#include "calculus.h"

namespace KuboGreenwood
{
	const double _coeff1 = 5.184182*1e-7, // q*hbar^2/(m0^2*KbT) [C*m^2/(J*s)]
		     _coeff2 = 3.8099815e-2,  // hbar^2/(2*m0) [ev*nm^2]
		     _KbT    = 0.0258520;     // [ev]

	// data storage
	struct infoStorage {
		// myyR, mzzR : relative ratio with the original mass of the electron
		// Esub : [ev]
		double myyR, mzzR, Esub;
	};

	template<class TensorKernel, class TauKernel>
	class numerator
	{
	public:
		numerator(const infoStorage& Info,
			  const TensorKernel& KKperRR, const TauKernel& Tau)
		: _myyR(Info.myyR), _mzzR(Info.mzzR), _Esub(Info.Esub),
		  _KKperRR(KKperRR), _Tau(Tau) {}

		double operator()(const double& ky, const double& kz) const {
			/*
			   _coeff1  : [C*m^2/(J*s)]
			   _KKperRR : [nanometer^-2] = 1e18*[m^-2]
				      (domain K of the _dFD has dimension for the [nm] scale)
	  		   _Tau     : [sec]
			   _dFD     : [dimensionless]
			*/
			return -_coeff1*1e18*_KKperRR(ky, kz)*_Tau(ky, kz)*_dFD(ky, kz);
		}

	private:
		const double _myyR, _mzzR, _Esub;
		const TensorKernel& _KKperRR;
		const TauKernel& _Tau;

		//derivative with the Fermi-Dirac distribution
		double _dFD(const double& ky, const double& kz) const
		{
			const double x = (_Esub + _coeff2*(ky*ky/_myyR + kz*kz/_mzzR))/_KbT;
			return -std::exp(x)/std::pow(std::exp(x) + 1., 2);
		}
	};


	class denominator
	{
	public:
		explicit denominator(const infoStorage& Info)
		: _myyR(Info.myyR), _mzzR(Info.mzzR), _Esub(Info.Esub) {}

		double operator()(const double& ky, const double& kz) const {
			// _FD : [dimensionless]
			return _FD(ky, kz);
		}
	private:
		const double _myyR, _mzzR, _Esub;

		
		// Fermi-Dirac distribution
		double _FD(const double& ky, const double& kz) const
		{
			const double x = (_Esub + _coeff2*(ky*ky/_myyR + kz*kz/_mzzR))/_KbT;
			return 1./(std::exp(x) + 1.);
		}
	};


	class kernelYY // [nanometer^-2] = 1e18*[m^-2]
	{
	public:
		explicit kernelYY(const infoStorage& Info)
		: _myyR(Info.myyR) {}

		double operator()(const double& ky, const double& kz) const {
			return std::pow(ky/_myyR, 2);
		}
	private:
		const double _myyR;
	};


	class kernelZZ // [nanometer^-2] = 1e18*[m^-2]
	{
	public:
		explicit kernelZZ(const infoStorage& Info)
		: _mzzR(Info.mzzR) {}

		double operator()(const double& ky, const double& kz) const {
			return std::pow(kz/_mzzR, 2);
		}
	private:
		const double _mzzR;
	};


	class kernelYZ // [nanometer^-2] = 1e18*[m^-2]
	{
	public:
		explicit kernelYZ(const infoStorage& Info)
		: _myyR(Info.myyR), _mzzR(Info.mzzR) {}

		double operator()(const double& ky, const double& kz) const {
			return ky*kz/(_myyR*_mzzR);
		}
	private:
		const double _myyR, _mzzR;
	};


	class kernelZY // [nanometer^-2] = 1e18*[m^-2]
	{
	public:
		explicit kernelZY(const infoStorage& Info)
		: _myyR(Info.myyR), _mzzR(Info.mzzR) {}

		double operator()(const double& ky, const double& kz) const {
			return kz*ky/(_mzzR*_myyR);
		}
	private:
		const double _myyR, _mzzR;
	};


	class kernelTau
	{
	public:
		double operator()(const double& ky, const double& kz) const {
			return 1e-12; // [sec]
		}
	};


	struct muMatrix
	{
		muMatrix(const double muyy_, const double muzz_,
			 const double muyz_, const double muzy_)
			: muyy(muyy_), muzz(muzz_), muyz(muyz_), muzy(muzy_) {}

		muMatrix(const muMatrix& rhs)
			: muyy(rhs.muyy), muzz(rhs.muzz), muyz(rhs.muyz), muzy(rhs.muzy) {}

		muMatrix(const muMatrix&& rhs)
			: muyy(rhs.muyy), muzz(rhs.muzz), muyz(rhs.muyz), muzy(rhs.muzy) {}

		double muyy, muzz, muyz, muzy;
	};

	template<class VECTOR1, class VECTOR2>
	muMatrix mobility(const VECTOR1& Esub0p91, const VECTOR2& Esub0p19,
		      const int Npoints = 1001, const double Width = 6.)
	{
		const int N_Esub0p91 = Esub0p91.size(),
			  N_Esub0p19 = Esub0p19.size();

		NUMERIC_CALCULUS::simpson_2d_method Integrator(Npoints);

		std::vector<double> k(Npoints, 0);
	
		for(int i=0; i<Npoints; ++i) {
			k[i] = i*Width/(Npoints - 1.) - Width/2.;
		}

		double numeYY0p91accum = 0,
		       numeZZ0p91accum = 0,
		       numeYZ0p91accum = 0,
		       numeZY0p91accum = 0,
		       denom0p91accum = 0;

		for(int i=0; i<N_Esub0p91; ++i)
		{
			infoStorage Info;
			Info.myyR = 0.19;
			Info.mzzR = 0.19;
			Info.Esub = Esub0p91[i];

			kernelYY kYY(Info);
			kernelZZ kZZ(Info);
			kernelYZ kYZ(Info);
			kernelZY kZY(Info);
			kernelTau kTau;

			numerator<kernelYY, kernelTau> numyy(Info, kYY, kTau);
			numerator<kernelZZ, kernelTau> numzz(Info, kZZ, kTau);
			numerator<kernelYZ, kernelTau> numyz(Info, kYZ, kTau);
			numerator<kernelZY, kernelTau> numzy(Info, kZY, kTau);
			denominator denomFunctor(Info);

			const double numeYY = Integrator(numyy, k, k),
				     numeZZ = Integrator(numzz, k, k),
				     numeYZ = Integrator(numyz, k, k),
				     numeZY = Integrator(numzy, k, k),
				     denom  = Integrator(denomFunctor, k, k);

			numeYY0p91accum += numeYY;
			numeZZ0p91accum += numeZZ;
			numeYZ0p91accum += numeYZ;
			numeZY0p91accum += numeZY;
			denom0p91accum  += denom;
		}


		double numeYY0p19accum = 0,
		       numeZZ0p19accum = 0,
		       numeYZ0p19accum = 0,
		       numeZY0p19accum = 0,
		       denom0p19accum = 0;

		for(int i=0; i<N_Esub0p19; ++i)
		{
			infoStorage Info;
			Info.myyR = 0.91;
			Info.myyR = 0.19;
			Info.Esub = Esub0p19[i];

			kernelYY kYY(Info);
			kernelZZ kZZ(Info);
			kernelYZ kYZ(Info);
			kernelZY kZY(Info);
			kernelTau kTau;

			numerator<kernelYY, kernelTau> numyy(Info, kYY, kTau);
			numerator<kernelZZ, kernelTau> numzz(Info, kZZ, kTau);
			numerator<kernelYZ, kernelTau> numyz(Info, kYZ, kTau);
			numerator<kernelZY, kernelTau> numzy(Info, kZY, kTau);
			denominator denomFunctor(Info);

			const double numeYY = Integrator(numyy, k, k),
				     numeZZ = Integrator(numzz, k, k),
				     numeYZ = Integrator(numyz, k, k),
				     numeZY = Integrator(numzy, k, k),
				     denom  = Integrator(denomFunctor, k, k);

			numeYY0p19accum += numeYY;
			numeZZ0p19accum += numeZZ;
			numeYZ0p19accum += numeYZ;
			numeZY0p19accum += numeZY;
			denom0p19accum  += denom;
		}

		double muyy = (2.*numeYY0p91accum + 4.*numeYY0p19accum)/
			      (2.*denom0p91accum + 4.*denom0p19accum),

		       muzz = (2.*numeZZ0p91accum + 4.*numeZZ0p19accum)/
			      (2.*denom0p91accum + 4.*denom0p19accum),

		       muyz = (2.*numeYZ0p91accum + 4.*numeYZ0p19accum)/
			      (2.*denom0p91accum + 4.*denom0p19accum),

		       muzy = (2.*numeZY0p91accum + 4.*numeZY0p19accum)/
			      (2.*denom0p91accum + 4.*denom0p19accum);

		return std::move(muMatrix(muyy, muzz, muyz, muzy));
	}
}

namespace TEMPORARY
{
	inline bool read_2d_file(const char filename[],
			         std::vector<double>& x, std::vector<double>& y)
	{
		/*
			x_{0}	y_{0}
			x_{1}	y_{1}
			  .       .
			  .       .
			  .       .
			x_{n-1}	y_{n-1}
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

		assert(tempListx.size() == tempListy.size());

		const int N = tempListx.size();

		if(N != x.size()) {
			std::vector<double>(N).swap(x);
		}
		if(N != y.size()) {
			std::vector<double>(N).swap(y);
		}

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
}

int main(int argc, char* argv[])
{
	std::vector<double> Esub0p91, Esub0p19;

	TEMPORARY::read_2d_file("e-0p91-0p19.dat", Esub0p91, Esub0p19);

	auto M = KuboGreenwood::mobility(Esub0p91, Esub0p19);

	std::cout << "muyy : " << M.muyy << std::endl
		  << "muyz : " << M.muyz << std::endl
		  << "muzz : " << M.muzz << std::endl
		  << "muzy : " << M.muzy << std::endl;

	return 0;
}
