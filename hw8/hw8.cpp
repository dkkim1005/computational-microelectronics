#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include "calculus.h"

namespace KuboGreenwood
{
	const double _coeff1 = 5.184182*1e-3, // q*hbar^2/(m0^2*KbT) [C*(cm)^2/(J*s)]
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
		  _KKperRR(KKperRR), _Tau(Tau), _dFD(Info) {}

		double operator()(const double& ky, const double& kz) const {
			/*
			   _coeff1  : [C*(cm)^2/(J*s)]
			   _KKperRR : [nanometer^-2] = 1e14*[cm^-2]
				      (domain K of the _dFD has dimension for the [nm] scale)
	  		   _Tau     : [sec]
			   _dFD     : [dimensionless]
			*/
			return _coeff1*1e14*_KKperRR(ky, kz)*_Tau(ky, kz)*_dFD(ky, kz);
		}

	private:
		const double _myyR, _mzzR, _Esub;
		const TensorKernel& _KKperRR;
		const TauKernel& _Tau;

		class derivFermiDirac
		{
		public:
			explicit derivFermiDirac(const infoStorage& Info)
			: _myyR(Info.myyR), _mzzR(Info.mzzR), _Esub(Info.Esub) {}

			double operator()(const double& ky, const double& kz) const
			{
				return -1./std::pow(std::exp((_Esub + _coeff2*(ky*ky/_myyR + kz*kz/_mzzR))/_KbT) + 1., 2)*
					   std::exp((_Esub + _coeff2*(ky*ky/_myyR + kz*kz/_mzzR))/_KbT);
			}
		private:
			const double _myyR, _mzzR, _Esub;
		};

		const derivFermiDirac _dFD;
	};


	class denominator
	{
	public:
		explicit denominator(const infoStorage& Info)
		: _myyR(Info.myyR), _mzzR(Info.mzzR), _Esub(Info.Esub), _FD(Info) {}

		double operator()(const double& ky, const double& kz) const {
			// _FD : [dimensionless]
			return _FD(ky, kz);
		}
	private:
		const double _myyR, _mzzR, _Esub;

		class FermiDirac
		{
		public:
			explicit FermiDirac(const infoStorage& Info)
			: _myyR(Info.myyR), _mzzR(Info.mzzR), _Esub(Info.Esub) {}

			double operator()(const double& ky, const double& kz) const {
				return 1./(std::exp((_Esub + _coeff2*(ky*ky/_myyR + kz*kz/_mzzR))/_KbT) + 1.);
			}

		private:
			const double _myyR, _mzzR, _Esub;
		};

		const FermiDirac _FD;
	};


	class kernelYY // [nanometer^-2] = 1e14*[cm^-2]
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


	class kernelZZ // [nanometer^-2] = 1e14*[cm^-2]
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


	class kernelYZ // [nanometer^-2] = 1e14*[cm^-2]
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


	class kernelZY // [nanometer^-2] = 1e14*[cm^-2]
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


	class kernelTau // [nanometer^-2] = 1e14*[cm^-2]
	{
	public:
		double operator()(const double& ky, const double& kz) const {
			return 1e-12; // [sec]
		}
	};


	void test_routine()
	{
		const int Npoints = 1001;
		const double Width = 6.;
		infoStorage Info;
		Info.myyR = 0.91;
		Info.mzzR = 0.19;
		Info.Esub = 0.5;

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

		std::vector<double> k(Npoints, 0);
	
		for(int i=0; i<Npoints; ++i) {
			k[i] = i*Width/(Npoints - 1.) - Width/2.;
		}

		NUMERIC_CALCULUS::simpson_2d_method Integrator(Npoints);

		const double numeYY = Integrator(numyy, k, k),
			     numeZZ = Integrator(numzz, k, k),
			     numeYZ = Integrator(numyz, k, k),
			     numeZY = Integrator(numzy, k, k),
			     denom  = Integrator(denomFunctor, k, k);

		std::cout << "mu_{yy} : " << numeYY/denom << " [cm^2/(V*sec)]" << std::endl;
		std::cout << "mu_{zz} : " << numeZZ/denom << " [cm^2/(V*sec)]" << std::endl;
		std::cout << "mu_{yz} : " << numeYZ/denom << " [cm^2/(V*sec)]" << std::endl;
		std::cout << "mu_{zy} : " << numeZY/denom << " [cm^2/(V*sec)]" << std::endl;
	}
}


int main(int argc, char* argv[])
{
	KuboGreenwood::test_routine();

	return 0;
}
