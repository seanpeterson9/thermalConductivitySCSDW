from pycuda import *
import pycuda.driver as cuda
import pycuda.autoinit as init
from pycuda.compiler import SourceModule
import numpy as np
import math
from scipy.integrate import quad
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
import csv
import gc

gpu_code = SourceModule("""

struct matrix4 {
	double m11;
	double m12;
	double m13;
	double m14;
	
	double m21;
	double m22;
	double m23;
	double m24;

	double m31;
	double m32;
	double m33;
	double m34;

	double m41;
	double m42;
	double m43;
	double m44;
}matrix4;

struct vector4 {
	double v1;
	double v2;
	double v3;
	double v4;
}vector4;

__device__ double vector4Dot(struct vector4 a, struct vector4 b) {
    double ret = a.v1*b.v1 + a.v2*b.v2 + a.v3*b.v3 + a.v4*b.v4;
    return ret;
}

__device__ struct vector4 matrixVecMult4(struct matrix4 m, struct vector4 v) {
    struct vector4 ret;
    
    ret.v1 = m.m11*v.v1 + m.m12*v.v2 + m.m13*v.v3 + m.m14*v.v4;
    ret.v2 = m.m21*v.v1 + m.m22*v.v2 + m.m23*v.v3 + m.m24*v.v4;
    ret.v3 = m.m31*v.v1 + m.m32*v.v2 + m.m33*v.v3 + m.m34*v.v4;
    ret.v4 = m.m41*v.v1 + m.m42*v.v2 + m.m43*v.v3 + m.m44*v.v4;

    return ret;
}

__device__ struct matrix4 matrixMult4(struct matrix4 a, struct matrix4 b) {
	struct matrix4 ret;

	ret.m11 = a.m11*b.m11 + a.m12*b.m21 + a.m13*b.m31 + a.m14*b.m41;
	ret.m12 = a.m11*b.m12 + a.m12*b.m22 + a.m13*b.m32 + a.m14*b.m42;
	ret.m13 = a.m11*b.m13 + a.m12*b.m23 + a.m13*b.m33 + a.m14*b.m43;
	ret.m14 = a.m11*b.m14 + a.m12*b.m24 + a.m13*b.m34 + a.m14*b.m44;

	ret.m21 = a.m21*b.m11 + a.m22*b.m21 + a.m23*b.m31 + a.m24*b.m41;
	ret.m22 = a.m21*b.m12 + a.m22*b.m22 + a.m23*b.m32 + a.m24*b.m42;
	ret.m23 = a.m21*b.m13 + a.m22*b.m23 + a.m23*b.m33 + a.m24*b.m43;
	ret.m24 = a.m21*b.m14 + a.m22*b.m24 + a.m23*b.m34 + a.m24*b.m44;

	ret.m31 = a.m31*b.m11 + a.m32*b.m21 + a.m33*b.m31 + a.m34*b.m41;
	ret.m32 = a.m31*b.m12 + a.m32*b.m22 + a.m33*b.m32 + a.m34*b.m42;
	ret.m33 = a.m31*b.m13 + a.m32*b.m23 + a.m33*b.m33 + a.m34*b.m43;
	ret.m34 = a.m31*b.m14 + a.m32*b.m24 + a.m33*b.m34 + a.m34*b.m44;

	ret.m41 = a.m41*b.m11 + a.m42*b.m21 + a.m43*b.m31 + a.m44*b.m41;
	ret.m42 = a.m41*b.m12 + a.m42*b.m22 + a.m43*b.m32 + a.m44*b.m42;
	ret.m43 = a.m41*b.m13 + a.m42*b.m23 + a.m43*b.m33 + a.m44*b.m43;
	ret.m44 = a.m41*b.m14 + a.m42*b.m24 + a.m43*b.m34 + a.m44*b.m44;

	return ret;
}

__device__ struct matrix4 matrixTrans4(struct matrix4 a) {
	struct matrix4 ret;
	
	ret.m11 = a.m11;
	ret.m12 = a.m21;
	ret.m13 = a.m31;
	ret.m14 = a.m41;

	ret.m21 = a.m12;
	ret.m22 = a.m22;
	ret.m23 = a.m32;
	ret.m24 = a.m42;
	
	ret.m31 = a.m13;
	ret.m32 = a.m23;
	ret.m33 = a.m33;
	ret.m34 = a.m43;

	ret.m41 = a.m14;
	ret.m42 = a.m24;
	ret.m43 = a.m34;
	ret.m44 = a.m44;

	return ret;
}

__device__ double matrixDet4(struct matrix4 a) {
	double ret;
	
	ret = a.m11*a.m22*a.m33*a.m44 + a.m11*a.m23*a.m34*a.m42 + a.m11*a.m24*a.m32*a.m43 + a.m12*a.m21*a.m34*a.m43 + a.m12*a.m23*a.m31*a.m44 + a.m12*a.m24*a.m33*a.m41 + a.m13*a.m21*a.m32*a.m44 + a.m13*a.m22*a.m34*a.m41 + a.m13*a.m24*a.m31*a.m42 + a.m14*a.m21*a.m33*a.m42 + a.m14*a.m22*a.m31*a.m43 + a.m14*a.m23*a.m32*a.m41 - a.m11*a.m22*a.m34*a.m43 - a.m11*a.m23*a.m32*a.m44 - a.m11*a.m24*a.m33*a.m42 - a.m12*a.m21*a.m33*a.m44 - a.m12*a.m23*a.m34*a.m41 - a.m12*a.m24*a.m31*a.m43 - a.m13*a.m21*a.m34*a.m42 - a.m13*a.m22*a.m31*a.m44 - a.m13*a.m24*a.m32*a.m41 - a.m14*a.m21*a.m32*a.m43 - a.m14*a.m22*a.m33*a.m41 - a.m14*a.m23*a.m31*a.m42;

	return ret;
}

__device__ struct matrix4 matrixInv4(struct matrix4 a) {
	struct matrix4 b;
	
	double detA = matrixDet4(a);
	
	b.m11 = (a.m22*a.m33*a.m44 + a.m23*a.m34*a.m42 + a.m24*a.m32*a.m43 - a.m22*a.m34*a.m43 - a.m23*a.m32*a.m44 - a.m24*a.m33*a.m42) / detA;
	b.m12 = (a.m12*a.m34*a.m43 + a.m13*a.m32*a.m44 + a.m14*a.m33*a.m42 - a.m12*a.m33*a.m44 - a.m13*a.m34*a.m42 - a.m14*a.m32*a.m43) / detA;
	b.m13 = (a.m12*a.m23*a.m44 + a.m13*a.m24*a.m42 + a.m14*a.m22*a.m43 - a.m12*a.m24*a.m43 - a.m13*a.m22*a.m44 - a.m14*a.m23*a.m42) / detA;
	b.m14 = (a.m12*a.m24*a.m33 + a.m13*a.m22*a.m34 + a.m14*a.m23*a.m32 - a.m12*a.m23*a.m34 - a.m13*a.m24*a.m32 - a.m14*a.m22*a.m33) / detA;
	
	b.m21 = (a.m21*a.m34*a.m43 + a.m23*a.m31*a.m44 + a.m24*a.m33*a.m41 - a.m21*a.m33*a.m44 - a.m23*a.m34*a.m41 - a.m24*a.m31*a.m43) / detA;
	b.m22 = (a.m11*a.m33*a.m44 + a.m13*a.m34*a.m41 + a.m14*a.m31*a.m43 - a.m11*a.m34*a.m43 - a.m13*a.m31*a.m44 - a.m14*a.m33*a.m41) / detA;
	b.m23 = (a.m11*a.m24*a.m43 + a.m13*a.m21*a.m44 + a.m14*a.m23*a.m41 - a.m11*a.m23*a.m44 - a.m13*a.m24*a.m41 - a.m14*a.m21*a.m43) / detA;
	b.m24 = (a.m11*a.m23*a.m34 + a.m13*a.m24*a.m31 + a.m14*a.m21*a.m33 - a.m11*a.m24*a.m33 - a.m13*a.m21*a.m34 - a.m14*a.m23*a.m31) / detA;
	
	b.m31 = (a.m21*a.m32*a.m44 + a.m22*a.m34*a.m41 + a.m24*a.m31*a.m42 - a.m21*a.m34*a.m42 - a.m22*a.m31*a.m44 - a.m24*a.m32*a.m41) / detA;
	b.m32 = (a.m11*a.m34*a.m42 + a.m12*a.m31*a.m44 + a.m14*a.m32*a.m41 - a.m11*a.m32*a.m44 - a.m12*a.m34*a.m41 - a.m14*a.m31*a.m42) / detA;
	b.m33 = (a.m11*a.m22*a.m44 + a.m12*a.m24*a.m41 + a.m14*a.m21*a.m42 - a.m11*a.m24*a.m42 - a.m12*a.m21*a.m44 - a.m14*a.m22*a.m41) / detA;
	b.m34 = (a.m11*a.m24*a.m32 + a.m12*a.m21*a.m34 + a.m14*a.m22*a.m31 - a.m11*a.m22*a.m34 - a.m12*a.m24*a.m31 - a.m14*a.m21*a.m32) / detA;
	
	b.m41 = (a.m21*a.m33*a.m42 + a.m22*a.m31*a.m43 + a.m23*a.m32*a.m41 - a.m21*a.m32*a.m43 - a.m22*a.m33*a.m41 - a.m23*a.m31*a.m42) / detA;
	b.m42 = (a.m11*a.m32*a.m43 + a.m12*a.m33*a.m41 + a.m13*a.m31*a.m42 - a.m11*a.m33*a.m42 - a.m12*a.m31*a.m43 - a.m13*a.m32*a.m41) / detA;
	b.m43 = (a.m11*a.m23*a.m42 + a.m12*a.m21*a.m43 + a.m13*a.m22*a.m41 - a.m11*a.m22*a.m43 - a.m12*a.m23*a.m41 - a.m13*a.m21*a.m42) / detA;
	b.m44 = (a.m11*a.m22*a.m33 + a.m12*a.m23*a.m31 + a.m13*a.m21*a.m32 - a.m11*a.m23*a.m32 - a.m12*a.m21*a.m33 - a.m13*a.m22*a.m31) / detA;
	
	return b;
}

__device__ struct vector4 eigenvalues4(double xi_k, double xi_kQ, double delta_k, double delta_kQ, double M) {
	struct vector4 ret;
	
	if((fabs(M) == 0.) && (fabs(delta_k) == 0.) && (fabs(delta_kQ) == 0.)) {
		ret.v1 = xi_k;
		ret.v2 = -xi_k;
		ret.v3 = xi_kQ;
		ret.v4 = -xi_kQ;
	}
	else if((fabs(M) == 0.) && ((delta_k != 0.) || (delta_kQ != 0.))) {	
		double Ek = sqrt(pow(xi_k,2) + pow(delta_k,2));
		double EkQ = sqrt(pow(xi_kQ,2) + pow(delta_kQ,2));
		
		ret.v1 = Ek;
		ret.v2 = -Ek;
		ret.v3 = EkQ;
		ret.v4 = -EkQ;
	}
	else if((M != 0.) && (delta_k == 0.) && (delta_kQ == 0.)) {
		double xi_k_plus = (xi_k + xi_kQ)/2.;
		double xi_k_minus = (xi_k - xi_kQ)/2.;
		
		ret.v1 = xi_k_plus + sqrt(pow(xi_k_minus,2.) + pow(M,2.));
		ret.v2 = -(xi_k_plus + sqrt(pow(xi_k_minus,2.) + pow(M,2.)));
		ret.v3 = xi_k_plus - sqrt(pow(xi_k_minus,2.) + pow(M,2.));
		ret.v4 = -(xi_k_plus - sqrt(pow(xi_k_minus,2.) + pow(M,2.)));
	}
	else {
		double xi_k_plus = (xi_k + xi_kQ)/2.;
		double xi_k_minus = (xi_k - xi_kQ)/2.;
		double delta_k_plus = (delta_k + delta_kQ)/2.;
		double delta_k_minus = (delta_k - delta_kQ)/2.;
	
		double Gamma_k = pow(xi_k_plus,2) + pow(xi_k_minus,2) + pow(delta_k_plus,2) + pow(delta_k_minus,2) + pow(M,2);
		double Lambda_k = sqrt(pow(xi_k_plus*xi_k_minus + delta_k_plus*delta_k_minus,2) + pow(M,2)*(pow(xi_k_plus,2) + pow(delta_k_plus,2)));
	
		double E1_plus = sqrt(Gamma_k + 2.*Lambda_k);
		double E1_minus = -E1_plus;
		double E2_plus = 0.;
		E2_plus = sqrt(fabs(Gamma_k - 2.*Lambda_k));
		double E2_minus = -E2_plus;
		
		ret.v1 = E1_plus;
		ret.v2 = E1_minus;
		ret.v3 = E2_plus;
		ret.v4 = E2_minus;
	}

	return ret;
}

__device__ struct matrix4 identityMatrix4() {
        struct matrix4 ret;
        
        ret.m11 = 1.;
        ret.m12 = 0;
        ret.m13 = 0;
        ret.m14 = 0;

        ret.m21 = 0;
        ret.m22 = 1.;
        ret.m23 = 0;
        ret.m24 = 0;

        ret.m31 = 0;
        ret.m32 = 0;
        ret.m33 = 1.;
        ret.m34 = 0;

        ret.m41 = 0;
        ret.m42 = 0;
        ret.m43 = 0;
        ret.m44 = 1.;

        return ret;
}

__device__ struct matrix4 eigenvectors4_pureSDW(struct matrix4 matrix, struct vector4 eigenvalues) {
	struct matrix4 eigVecs;
	
	double xi_k = matrix.m11;
	double delta_k = matrix.m21;
	double xi_kQ = matrix.m33;
	double delta_kQ = matrix.m43;
	double M = matrix.m31;
	double xi_k_minus = .5*(xi_k-xi_kQ);
	
	double denom = sqrt(2*(pow(xi_k_minus,2) + pow(M,2) + xi_k_minus*sqrt(pow(xi_k_minus,2) + pow(M,2))));
	double gk = (xi_k_minus + sqrt(pow(xi_k_minus,2) + pow(M,2))) / denom;
	double hk = M / denom;
	
	eigVecs.m11 = gk;
	eigVecs.m21 = 0;
	eigVecs.m31 = hk;
	eigVecs.m41 = 0;
	
	eigVecs.m12 = 0;
	eigVecs.m22 = gk;
	eigVecs.m32 = 0;
	eigVecs.m42 = -hk;
	
	eigVecs.m13 = -hk;
	eigVecs.m23 = 0;
	eigVecs.m33 = gk;
	eigVecs.m43 = 0;
	
	eigVecs.m14 = 0;
	eigVecs.m24 = hk;
	eigVecs.m34 = 0;
	eigVecs.m44 = gk;
	
	return eigVecs;
}

__device__ struct matrix4 eigenvectors4(struct matrix4 matrix, struct vector4 eigenvalues,double f) {
	struct matrix4 eigVecs;
	
	double xi_k = matrix.m11;
	double delta_k = matrix.m21;
	double xi_kQ = matrix.m33;
	double delta_kQ = matrix.m43;
	double M = matrix.m31;

	if(M == 0) {
		if(delta_k == 0. && delta_kQ == 0.) {
			eigVecs = identityMatrix4();
		}
		else {
			double err = 1e-8;

			double Ek = sqrt(pow(xi_k,2) + pow(delta_k,2));
			double EkQ = sqrt(pow(xi_kQ,2) + pow(delta_kQ,2));
			
			double uk,vk;
			
			if(Ek > err) {
				if((fabs(delta_k) > 0.) && (sqrt(2.*Ek*(Ek + xi_k)) > 0.)) { 
					uk = (Ek + xi_k)/sqrt(2.*Ek*(Ek + xi_k));
					vk = (delta_k)/sqrt(2.*Ek*(Ek + xi_k));
				}
				else {
					uk = sqrt(.5*(1 + xi_k/Ek));
					vk = sqrt(.5*(1 - xi_k/Ek));
				}
			}
			else {
				uk = sqrt(.5);
				vk = sqrt(.5);
			}
			eigVecs.m11 = uk;
			eigVecs.m21 = vk;
			eigVecs.m31 = 0.;
			eigVecs.m41 = 0.;
				
			eigVecs.m12 = -vk;
			eigVecs.m22 = uk;
			eigVecs.m32 = 0.;
			eigVecs.m42 = 0.;
	
			double ukQ,vkQ;
	
			if(EkQ > err) {
				if((fabs(delta_kQ) > 0.) && (sqrt(2.*EkQ*(EkQ + xi_kQ)) > 0.)) {
					ukQ = (EkQ + xi_kQ)/sqrt(2.*EkQ*(EkQ + xi_kQ));
					vkQ = (delta_kQ)/sqrt(2.*EkQ*(EkQ + xi_kQ));
				}
				else {
					ukQ = sqrt(.5*(1 + xi_kQ/EkQ));
					vkQ = sqrt(.5*(1 - xi_kQ/EkQ));
				}
			}
			else {
				ukQ = sqrt(.5);
				vkQ = sqrt(.5);
			}
		
			eigVecs.m13 = 0.;
			eigVecs.m23 = 0.;
			eigVecs.m33 = ukQ;
			eigVecs.m43 = vkQ;
			
			eigVecs.m14 = 0.;
			eigVecs.m24 = 0.;
			eigVecs.m34 = -vkQ;
			eigVecs.m44 = ukQ;
		}
	}
	else {
		
		if((fabs(delta_k) == 0.) && (fabs(delta_kQ) == 0.)) {
			double xi_k_minus = .5*(xi_k-xi_kQ);
			double denom = sqrt(2*(pow(xi_k_minus,2) + pow(M,2) + xi_k_minus*sqrt(pow(xi_k_minus,2) + pow(M,2))));
			double gk = (xi_k_minus + sqrt(pow(xi_k_minus,2) + pow(M,2))) / denom;
			double hk = M / denom;
			
			eigVecs.m11 = gk;
			eigVecs.m21 = 0;
			eigVecs.m31 = hk;
			eigVecs.m41 = 0;
			
			eigVecs.m12 = 0;
			eigVecs.m22 = gk;
			eigVecs.m32 = 0;
			eigVecs.m42 = -hk;
			
			eigVecs.m13 = -hk;
			eigVecs.m23 = 0;
			eigVecs.m33 = gk;
			eigVecs.m43 = 0;
			
			eigVecs.m14 = 0;
			eigVecs.m24 = hk;
			eigVecs.m34 = 0;
			eigVecs.m44 = gk;
			
		}
		
		else {
			struct vector4 eigsSDW = eigenvalues4(xi_k, xi_kQ, 0., 0., M);
			struct vector4 eigsCoexist = eigenvalues4(xi_k,xi_kQ,delta_k,delta_kQ,M);
			struct vector4 eigsPert;
			struct matrix4 eigVecsSDW = eigenvectors4_pureSDW(matrix, eigsSDW);
			
			double xi_k_minus = .5*(xi_k-xi_kQ);
			double delta_k_plus = .5*(delta_k+delta_kQ);
			double denom = sqrt(2*(pow(xi_k_minus,2) + pow(M,2) + xi_k_minus*sqrt(pow(xi_k_minus,2) + pow(M,2))));
			double gk = (xi_k_minus + sqrt(pow(xi_k_minus,2) + pow(M,2))) / denom;
			double hk = M / denom;
			double gk2 = pow(gk,2);
			double hk2 = pow(hk,2);
			
			double E_alpha = eigsSDW.v1;
			double E_beta = eigsSDW.v3;
			double E1 = eigsCoexist.v1;
			double E2 = eigsCoexist.v3;
			
			if(fabs(delta_k_plus) > 0) {
				
				eigVecs.m11 = eigVecsSDW.m11 + (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m12 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m14;
				eigVecs.m21 = eigVecsSDW.m21 + (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m22 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m24;
				eigVecs.m31 = eigVecsSDW.m31 + (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m32 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m34;
				eigVecs.m41 = eigVecsSDW.m41 + (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m42 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m44;
				
				eigVecs.m12 = eigVecsSDW.m12 - (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m11 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m13;
				eigVecs.m22 = eigVecsSDW.m22 - (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m21 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m23;
				eigVecs.m32 = eigVecsSDW.m32 - (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m31 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m33;
				eigVecs.m42 = eigVecsSDW.m42 - (delta_k*(gk2-hk2)/(E_alpha+E1)) * eigVecsSDW.m41 - ((E_alpha-E1 + (pow(delta_k*(gk2-hk2),2)/(E_alpha+E1)))/(2*delta_k*gk*hk)) * eigVecsSDW.m43;
				if(E_beta > 0.) {
					eigVecs.m13 = eigVecsSDW.m13 + (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m14 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m12;
					eigVecs.m23 = eigVecsSDW.m23 + (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m24 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m22;
					eigVecs.m33 = eigVecsSDW.m33 + (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m34 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m32;
					eigVecs.m43 = eigVecsSDW.m43 + (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m44 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m42;
					
					eigVecs.m14 = eigVecsSDW.m14 - (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m13 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m11;
					eigVecs.m24 = eigVecsSDW.m24 - (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m23 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m21;
					eigVecs.m34 = eigVecsSDW.m34 - (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m33 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m31;
					eigVecs.m44 = eigVecsSDW.m44 - (delta_k*(gk2-hk2)/(E_beta+E2)) * eigVecsSDW.m43 + ((E_beta-E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta+E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m41;
				}
				else {
					eigVecs.m13 = eigVecsSDW.m14 - (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m13 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m11;
					eigVecs.m23 = eigVecsSDW.m24 - (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m23 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m21;
					eigVecs.m33 = eigVecsSDW.m34 - (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m33 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m31;
					eigVecs.m43 = eigVecsSDW.m44 - (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m43 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m41;
					
					eigVecs.m14 = eigVecsSDW.m13 + (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m14 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m12;
					eigVecs.m24 = eigVecsSDW.m23 + (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m24 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m22;
					eigVecs.m34 = eigVecsSDW.m33 + (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m34 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m32;
					eigVecs.m44 = eigVecsSDW.m43 + (delta_k*(gk2-hk2)/(E_beta-E2)) * eigVecsSDW.m44 + ((E_beta+E2 + (pow(delta_k*(gk2-hk2),2)/(E_beta-E2)))/(2*delta_k*gk*hk)) * eigVecsSDW.m42;
				}
			}
			else {
				
				double Ea = sqrt(pow(E_alpha,2) + pow(delta_k,2));
				double Eb = sqrt(pow(E_beta,2) + pow(delta_k,2));
				double uk_alpha = (Ea + E_alpha) / sqrt(2*Ea*(Ea + E_alpha));
				double vk_alpha = delta_k / sqrt(2*Ea*(Ea + E_alpha));
				double uk_beta = (Eb + E_beta) / sqrt(2*Eb*(Eb + E_beta));
				double vk_beta = delta_k / sqrt(2*Eb*(Eb + E_beta));
				
				eigVecs.m11 = uk_alpha*eigVecsSDW.m11 + vk_alpha * eigVecsSDW.m12;
				eigVecs.m21 = uk_alpha*eigVecsSDW.m21 + vk_alpha * eigVecsSDW.m22;
				eigVecs.m31 = uk_alpha*eigVecsSDW.m31 + vk_alpha * eigVecsSDW.m32;
				eigVecs.m41 = uk_alpha*eigVecsSDW.m41 + vk_alpha * eigVecsSDW.m42;
				
				eigVecs.m12 = uk_alpha*eigVecsSDW.m12 - vk_alpha * eigVecsSDW.m11;
				eigVecs.m22 = uk_alpha*eigVecsSDW.m22 - vk_alpha * eigVecsSDW.m21;
				eigVecs.m32 = uk_alpha*eigVecsSDW.m32 - vk_alpha * eigVecsSDW.m31;
				eigVecs.m42 = uk_alpha*eigVecsSDW.m42 - vk_alpha * eigVecsSDW.m41;
				
				eigVecs.m13 = uk_beta*eigVecsSDW.m13 - vk_beta * eigVecsSDW.m14;
				eigVecs.m23 = uk_beta*eigVecsSDW.m23 - vk_beta * eigVecsSDW.m24;
				eigVecs.m33 = uk_beta*eigVecsSDW.m33 - vk_beta * eigVecsSDW.m34;
				eigVecs.m43 = uk_beta*eigVecsSDW.m43 - vk_beta * eigVecsSDW.m44;
				
				eigVecs.m14 = uk_beta*eigVecsSDW.m14 + vk_beta * eigVecsSDW.m13;
				eigVecs.m24 = uk_beta*eigVecsSDW.m24 + vk_beta * eigVecsSDW.m23;
				eigVecs.m34 = uk_beta*eigVecsSDW.m34 + vk_beta * eigVecsSDW.m33;
				eigVecs.m44 = uk_beta*eigVecsSDW.m44 + vk_beta * eigVecsSDW.m43;
			}
			
			double Norm1 = sqrt(pow(eigVecs.m11,2) + pow(eigVecs.m21,2) + pow(eigVecs.m31,2) + pow(eigVecs.m41,2));
			double Norm2 = sqrt(pow(eigVecs.m12,2) + pow(eigVecs.m22,2) + pow(eigVecs.m32,2) + pow(eigVecs.m42,2));
			double Norm3 = sqrt(pow(eigVecs.m13,2) + pow(eigVecs.m23,2) + pow(eigVecs.m33,2) + pow(eigVecs.m43,2));
			double Norm4 = sqrt(pow(eigVecs.m14,2) + pow(eigVecs.m24,2) + pow(eigVecs.m34,2) + pow(eigVecs.m44,2));
			
			eigVecs.m11 = eigVecs.m11/Norm1;
			eigVecs.m21 = eigVecs.m21/Norm1;
			eigVecs.m31 = eigVecs.m31/Norm1;
			eigVecs.m41 = eigVecs.m41/Norm1;
			
			eigVecs.m12 = eigVecs.m12/Norm2;
			eigVecs.m22 = eigVecs.m22/Norm2;
			eigVecs.m32 = eigVecs.m32/Norm2;
			eigVecs.m42 = eigVecs.m42/Norm2;
			
			eigVecs.m13 = eigVecs.m13/Norm3;
			eigVecs.m23 = eigVecs.m23/Norm3;
			eigVecs.m33 = eigVecs.m33/Norm3;
			eigVecs.m43 = eigVecs.m43/Norm3;
			
			eigVecs.m14 = eigVecs.m14/Norm4;
			eigVecs.m24 = eigVecs.m24/Norm4;
			eigVecs.m34 = eigVecs.m34/Norm4;
			eigVecs.m44 = eigVecs.m44/Norm4;
		}
	}	
	return eigVecs;
}


__device__ double deltaGaussHerm(double Ek, double Ek_prime, double f) {
	double hermiteZero = 1;
	double hermiteTwo = 4*pow(Ek - Ek_prime,2)/(2*f*f) - 2;
	double coeffZero = 1/sqrt(M_PI);
	double coeffTwo = -1/(4*sqrt(M_PI));

	double gauss = exp(-pow(Ek - Ek_prime,2)/(2*f*f));
	return (1/(f*sqrt(2.)))*gauss*(coeffZero*hermiteZero + coeffTwo*hermiteTwo);
}

__device__ struct matrix4 coherence(struct matrix4 eigVecs, struct matrix4 eigVecsPrime) {
	struct matrix4 m;
	struct matrix4 coherenceMatrix;

	m.m11 = 1;
	m.m22 = -1;
	m.m33 = 1;
	m.m44 = -1;

	m.m12 = 0.;
	m.m13 = 0.;
	m.m14 = 0.;
	
	m.m21 = 0.;
	m.m23 = 0.;
	m.m24 = 0.;

	m.m31 = 0.;
	m.m32 = 0.;
	m.m34 = 0.;

	m.m41 = 0.;
	m.m42 = 0.;
	m.m43 = 0.;
	
	coherenceMatrix = matrixMult4(matrixTrans4(eigVecsPrime),matrixMult4(m,eigVecs));
	
	return coherenceMatrix;
}

__device__ double2 coherenceAnalytic(double xi_k,double delta_k,double E_k,double xi_k_prime,double delta_k_prime,double E_k_prime) {
	double2 ret;
	
	if (E_k == 0. || E_k_prime == 0.) {
		ret.x = 1.;
		ret.y = 0.;
	}
	else {
		ret.x = 0.5*(1 + (xi_k*xi_k_prime - delta_k*delta_k_prime)/(E_k*E_k_prime));
		ret.y = 0.;
	}
	return ret;
}

__device__ double2 coherenceSDW(double xi_k, double xi_kQ, double xi_k_prime, double xi_kQ_prime, double M_k, double M_k_prime) {
	double2 ret;

	double xi_k_minus = .5*(xi_k - xi_kQ);
	double xi_k_prime_minus = .5*(xi_k_prime - xi_kQ_prime);
	double Pi_k = sqrt(pow(xi_k_minus,2.) + pow(M_k,2.));
	double Pi_k_prime = sqrt(pow(xi_k_prime_minus,2.) + pow(M_k_prime,2.));

	ret.x = sqrt(.5*(1 + (xi_k_minus * xi_k_prime_minus + M_k * M_k_prime)/(Pi_k * Pi_k_prime)));
	ret.y = sqrt(.5*(1 - (xi_k_minus * xi_k_prime_minus + M_k * M_k_prime)/(Pi_k * Pi_k_prime)));
	return ret;
}

__global__ void n_scattering_gpu(int n, int m, double f, double* xi, double* xi_Q, double* dl, double* v, double* Tau_inv) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	double xi_k, xi_kQ, xi_k_prime, xi_kQ_prime;
	double dkl, vel;
	double deltaFactor;

	for(int i = 0; i < m; i++) {
		
		if(idx < n && m*idy + i < n) {
			xi_k = xi[idx];
			xi_kQ = xi_Q[idx];
			xi_k_prime = xi[m*idy + i];
			xi_kQ_prime = xi_Q[m*idy + i];

			dkl = dl[m*idy + i];
			vel = v[m*idy + i];
            
			deltaFactor = deltaGaussHerm(xi_k,xi_k_prime,f);
			
			Tau_inv[((n/m)+1)*idx + idy] += deltaFactor*dkl/(vel);
            
		}
	}
}

__global__ void SCSDW_scattering_gpu(int n1, int n2, int m, double f, double* M1, double* M2, double* delta1, double* delta_Q1, double* delta2, double* delta_Q2, double* xi1, double* xi_Q1, double* xi2, double* xi_Q2, double* dl1, double* dl2, double* dxi1, double* dxi2, double* v1, double* v2, double* Tau_inv, int whichGrid) {
	int idx = blockIdx.x*blockDim.x + threadIdx.x;
	int idy = blockIdx.y*blockDim.y + threadIdx.y;

	double xi_k1, xi_kQ1, xi_k_prime1, xi_kQ_prime1;
	double xi_k2, xi_kQ2, xi_k_prime2, xi_kQ_prime2;
	double delta_k1, delta_kQ1, delta_k_prime1, delta_kQ_prime1;
	double delta_k2, delta_kQ2, delta_k_prime2, delta_kQ_prime2;
	double dkl1, dkl2, dxi_k1, dxi_k2, vel1, vel2;
	double M_k1, M_k2, M_k_prime1, M_k_prime2;
	struct matrix4 H1, Hprime1;
	struct matrix4 H2, Hprime2;
	struct vector4 Ek1, Ek_prime1;
	struct vector4 Ek2, Ek_prime2;
	struct matrix4 eigVecs1,eigVecsPrime1;
	struct matrix4 eigVecs2,eigVecsPrime2;
	struct matrix4 coherenceMatrix1, coherenceMatrix2;
	double deltaFactor;

	if(whichGrid == 11) {
		for(int i = 0; i < m; i++) {
			if(idx < n1 && m*idy + i < n1) {
				xi_k1 = xi1[idx];
				xi_kQ1 = xi_Q1[idx];
				xi_k_prime1 = xi1[m*idy + i];
				xi_kQ_prime1 = xi_Q1[m*idy + i];
				
				delta_k1 = delta1[idx];
				delta_kQ1 = delta_Q1[idx];
				delta_k_prime1 = delta1[m*idy + i];
				delta_kQ_prime1 = delta_Q1[m*idy + i];
				
				M_k1 = M1[idx];
				M_k_prime1 = M1[m*idy + i];
				
				dkl1 = dl1[m*idy + i];
				dxi_k1 = dxi1[m*idy + i];
				vel1 = v1[m*idy + i];
				
				H1.m11 = xi_k1;
				H1.m12 = delta_k1;
				H1.m13 = M_k1;
				H1.m14 = 0.;
				
				H1.m21 = delta_k1;
				H1.m22 = -xi_k1;
				H1.m23 = 0.;
				H1.m24 = M_k1;
				
				H1.m31 = M_k1;
				H1.m32 = 0.;
				H1.m33 = xi_kQ1;
				H1.m34 = delta_kQ1;
				
				H1.m41 = 0.;
				H1.m42 = M_k1;
				H1.m43 = delta_kQ1;
				H1.m44 = -xi_kQ1;
				
				Hprime1.m11 = xi_k_prime1;
				Hprime1.m12 = delta_k_prime1;
				Hprime1.m13 = M_k_prime1;
				Hprime1.m14 = 0.;
				
				Hprime1.m21 = delta_k_prime1;
				Hprime1.m22 = -xi_k_prime1;
				Hprime1.m23 = 0.;
				Hprime1.m24 = M_k_prime1;
				
				Hprime1.m31 = M_k_prime1;
				Hprime1.m32 = 0.;
				Hprime1.m33 = xi_kQ_prime1;
				Hprime1.m34 = delta_kQ_prime1;
				
				Hprime1.m41 = 0.;
				Hprime1.m42 = M_k_prime1;
				Hprime1.m43 = delta_kQ_prime1;
				Hprime1.m44 = -xi_kQ_prime1;
				
				Ek1 = eigenvalues4(xi_k1,xi_kQ1,delta_k1,delta_kQ1,M_k1);
				Ek_prime1 = eigenvalues4(xi_k_prime1,xi_kQ_prime1,delta_k_prime1,delta_kQ_prime1,M_k_prime1);
				
				eigVecs1 = eigenvectors4(H1,Ek1,f);
				eigVecsPrime1 = eigenvectors4(Hprime1,Ek_prime1,f);
				coherenceMatrix1 = coherence(eigVecs1,eigVecsPrime1);

				deltaFactor = deltaGaussHerm(Ek1.v1,Ek_prime1.v1,f);

				if(coherenceMatrix1.m11 == coherenceMatrix1.m11) {
					Tau_inv[((n1/m)+1)*idx + idy] += pow(coherenceMatrix1.m11,2.)*deltaFactor*dkl1*dxi_k1/(vel1);
				}
			}
		}
	}
	else if(whichGrid == 12) {
		for(int i = 0; i < m; i++) {
			if(idx < n1 && m*idy + i < n2) {
				xi_k1 = xi1[idx];
				xi_kQ1 = xi_Q1[idx];
				xi_k_prime2 = xi2[m*idy + i];
				xi_kQ_prime2 = xi_Q2[m*idy + i];
				
				delta_k1 = delta1[idx];
				delta_kQ1 = delta_Q1[idx];
				delta_k_prime2 = delta2[m*idy + i];
				delta_kQ_prime2 = delta_Q2[m*idy + i];
				
				M_k1 = M1[idx];
				M_k_prime2 = M2[m*idy + i];
				
				dkl2 = dl2[m*idy + i];
				dxi_k2 = dxi2[m*idy + i];
				vel2 = v2[m*idy + i];
				
				H1.m11 = xi_k1;
				H1.m12 = delta_k1;
				H1.m13 = M_k1;
				H1.m14 = 0.;
				
				H1.m21 = delta_k1;
				H1.m22 = -xi_k1;
				H1.m23 = 0.;
				H1.m24 = M_k1;
				
				H1.m31 = M_k1;
				H1.m32 = 0.;
				H1.m33 = xi_kQ1;
				H1.m34 = delta_kQ1;
				
				H1.m41 = 0.;
				H1.m42 = M_k1;
				H1.m43 = delta_kQ1;
				H1.m44 = -xi_kQ1;
				
				Hprime2.m11 = xi_k_prime2;
				Hprime2.m12 = delta_k_prime2;
				Hprime2.m13 = M_k_prime2;
				Hprime2.m14 = 0.;
				
				Hprime2.m21 = delta_k_prime2;
				Hprime2.m22 = -xi_k_prime2;
				Hprime2.m23 = 0.;
				Hprime2.m24 = M_k_prime2;
				
				Hprime2.m31 = M_k_prime2;
				Hprime2.m32 = 0.;
				Hprime2.m33 = xi_kQ_prime2;
				Hprime2.m34 = delta_kQ_prime2;
				
				Hprime2.m41 = 0.;
				Hprime2.m42 = M_k_prime2;
				Hprime2.m43 = delta_kQ_prime2;
				Hprime2.m44 = -xi_kQ_prime2;
				
				Ek1 = eigenvalues4(xi_k1,xi_kQ1,delta_k1,delta_kQ1,M_k1);
				Ek_prime2 = eigenvalues4(xi_k_prime2,xi_kQ_prime2,delta_k_prime2,delta_kQ_prime2,M_k_prime2);
				
				eigVecs1 = eigenvectors4(H1,Ek1,f);
				eigVecsPrime2 = eigenvectors4(Hprime2,Ek_prime2,f);
				coherenceMatrix1 = coherence(eigVecs1,eigVecsPrime2);

				deltaFactor = deltaGaussHerm(Ek1.v1,Ek_prime2.v3,f);
				
				if(coherenceMatrix1.m13 == coherenceMatrix1.m13) {
					Tau_inv[((n2/m)+1)*idx + idy] += pow(coherenceMatrix1.m13,2.)*deltaFactor*dkl2*dxi_k2/(vel2);
				}
			}
		}
	}
	else if(whichGrid == 22) {
		for(int i = 0; i < m; i++) {
			if(idx < n2 && m*idy + i < n2) {
				xi_k2 = xi2[idx];
				xi_kQ2 = xi_Q2[idx];
				xi_k_prime2 = xi2[m*idy + i];
				xi_kQ_prime2 = xi_Q2[m*idy + i];
				
				delta_k2 = delta2[idx];
				delta_kQ2 = delta_Q2[idx];
				delta_k_prime2 = delta2[m*idy + i];
				delta_kQ_prime2 = delta_Q2[m*idy + i];
				
				M_k2 = M2[idx];
				M_k_prime2 = M2[m*idy + i];
				
				dkl2 = dl2[m*idy + i];
				dxi_k2 = dxi2[m*idy + i];
				vel2 = v2[m*idy + i];
				
				H2.m11 = xi_k2;
				H2.m12 = delta_k2;
				H2.m13 = M_k2;
				H1.m14 = 0.;
				
				H2.m21 = delta_k2;
				H2.m22 = -xi_k2;
				H2.m23 = 0.;
				H2.m24 = M_k2;
				
				H2.m31 = M_k2;
				H2.m32 = 0.;
				H2.m33 = xi_kQ2;
				H2.m34 = delta_kQ2;
				
				H2.m41 = 0.;
				H2.m42 = M_k2;
				H2.m43 = delta_kQ2;
				H2.m44 = -xi_kQ2;
				
				Hprime2.m11 = xi_k_prime2;
				Hprime2.m12 = delta_k_prime2;
				Hprime2.m13 = M_k_prime2;
				Hprime2.m14 = 0.;
				
				Hprime2.m21 = delta_k_prime2;
				Hprime2.m22 = -xi_k_prime2;
				Hprime2.m23 = 0.;
				Hprime2.m24 = M_k_prime2;
				
				Hprime2.m31 = M_k_prime2;
				Hprime2.m32 = 0.;
				Hprime2.m33 = xi_kQ_prime2;
				Hprime2.m34 = delta_kQ_prime2;
				
				Hprime2.m41 = 0.;
				Hprime2.m42 = M_k_prime2;
				Hprime2.m43 = delta_kQ_prime2;
				Hprime2.m44 = -xi_kQ_prime2;
				
				Ek2 = eigenvalues4(xi_k2,xi_kQ2,delta_k2,delta_kQ2,M_k2);
				Ek_prime2 = eigenvalues4(xi_k_prime2,xi_kQ_prime2,delta_k_prime2,delta_kQ_prime2,M_k_prime2);
				
				eigVecs2 = eigenvectors4(H2,Ek2,f);
				eigVecsPrime2 = eigenvectors4(Hprime2,Ek_prime2,f);
				coherenceMatrix2 = coherence(eigVecs2,eigVecsPrime2);

				deltaFactor = deltaGaussHerm(Ek2.v3,Ek_prime2.v3,f);
				
				if(coherenceMatrix2.m33 == coherenceMatrix2.m33) {
					Tau_inv[((n2/m)+1)*idx + idy] += pow(coherenceMatrix2.m33,2.)*deltaFactor*dkl2*dxi_k2/(vel2);
				}
			}
		}
	}
	else if(whichGrid == 21) {
		for(int i = 0; i < m; i++) {
			if(idx < n2 && m*idy + i < n1) {
				xi_k2 = xi2[idx];
				xi_kQ2 = xi_Q2[idx];
				xi_k_prime1 = xi1[m*idy + i];
				xi_kQ_prime1 = xi_Q1[m*idy + i];
				
				delta_k2 = delta2[idx];
				delta_kQ2 = delta_Q2[idx];
				delta_k_prime1 = delta1[m*idy + i];
				delta_kQ_prime1 = delta_Q1[m*idy + i];
				
				M_k2 = M2[idx];
				M_k_prime1 = M1[m*idy + i];
				
				dkl1 = dl1[m*idy + i];
				dxi_k1 = dxi1[m*idy + i];
				vel1 = v1[m*idy + i];
				
				H2.m11 = xi_k2;
				H2.m12 = delta_k2;
				H2.m13 = M_k2;
				H2.m14 = 0.;
				
				H2.m21 = delta_k2;
				H2.m22 = -xi_k2;
				H2.m23 = 0.;
				H2.m24 = M_k2;
				
				H2.m31 = M_k2;
				H2.m32 = 0.;
				H2.m33 = xi_kQ2;
				H2.m34 = delta_kQ2;
				
				H2.m41 = 0.;
				H2.m42 = M_k2;
				H2.m43 = delta_kQ2;
				H2.m44 = -xi_kQ2;
				
				Hprime1.m11 = xi_k_prime1;
				Hprime1.m12 = delta_k_prime1;
				Hprime1.m13 = M_k_prime1;
				Hprime1.m14 = 0.;
				
				Hprime1.m21 = delta_k_prime1;
				Hprime1.m22 = -xi_k_prime1;
				Hprime1.m23 = 0.;
				Hprime1.m24 = M_k_prime1;
				
				Hprime1.m31 = M_k_prime1;
				Hprime1.m32 = 0.;
				Hprime1.m33 = xi_kQ_prime1;
				Hprime1.m34 = delta_kQ_prime1;
				
				Hprime1.m41 = 0.;
				Hprime1.m42 = M_k_prime1;
				Hprime1.m43 = delta_kQ_prime1;
				Hprime1.m44 = -xi_kQ_prime1;
				
				Ek2 = eigenvalues4(xi_k2,xi_kQ2,delta_k2,delta_kQ2,M_k2);
				Ek_prime1 = eigenvalues4(xi_k_prime1,xi_kQ_prime1,delta_k_prime1,delta_kQ_prime1,M_k_prime1);
				
				eigVecs2 = eigenvectors4(H2,Ek2,f);
				eigVecsPrime1 = eigenvectors4(Hprime1,Ek_prime1,f);
				coherenceMatrix2 = coherence(eigVecs2,eigVecsPrime1);

				deltaFactor = deltaGaussHerm(Ek2.v3,Ek_prime1.v1,f);
				
				if(coherenceMatrix2.m31 == coherenceMatrix2.m31) {
					Tau_inv[((n1/m)+1)*idx + idy] += pow(coherenceMatrix2.m31,2.)*deltaFactor*dkl1*dxi_k1/(vel1);
				}
			}
		}
	}
}

""")#,options = ['-I/home/speterson/Research/souravStuff/eigen-3.4.0/'])

def dispersionTB(mu,k_x,k_y,a,t1,t2):
	return mu - t1*(np.cos(2*k_x*a)+np.cos(2*k_y*a)) - t2*np.cos(2*k_x*a)*np.cos(2*k_y*a)
	#return mu - t1*(np.cos(k_x*a)+np.cos(k_y*a)) - t2*np.cos(k_x*a)*np.cos(k_y*a)

def dispersionTB_aniso(mu,k_x,k_y,a,t1,t2):
	k_1 = np.sin(np.pi/4)*k_x + np.cos(np.pi/4)*k_y
	k_2 = np.cos(np.pi/4.)*k_x - np.sin(np.pi/4)*k_y

	howFar = np.pi/np.sqrt(8)
	howFar2 = np.pi/np.sqrt(2)

	howMany1 = np.zeros(k_1.shape)
	howMany2 = np.zeros(k_2.shape)

	while len(k_1[k_1 > howFar]) > 0:
		howMany1[k_1 > howFar] = howMany1[k_1 > howFar] + 1
		k_1[k_1 > howFar] = k_1[k_1 > howFar] - np.pi/np.sqrt(2)
    	
	while len(k_1[k_1 < -howFar]) > 0:
		howMany1[k_1 < -howFar] = howMany1[k_1 < -howFar] + 1
		k_1[k_1 < -howFar] = k_1[k_1 < -howFar] + np.pi/np.sqrt(2)

	while len(k_2[k_2 > howFar2]) > 0:
		howMany2[k_2 > howFar2] = howMany2[k_2 > howFar2] + 1
		k_2[k_2 > howFar2] = k_2[k_2 > howFar2] - np.pi*np.sqrt(2)

	while len(k_2[k_2 < -howFar2]) > 0:
		howMany2[k_2 < -howFar2] = howMany2[k_2 < -howFar2] + 1
		k_2[k_2 < -howFar2] = k_2[k_2 < -howFar2] + np.pi*np.sqrt(2)

	mask = (howMany1 % 2 != 0)

	k_x = np.sin(np.pi/4)*k_1 - np.cos(np.pi/4)*k_2
	k_y = np.cos(np.pi/4)*k_1 + np.sin(np.pi/4)*k_2

	ret=  -mu -t1*(np.cos(2*k_x*a)+np.cos(2*k_y*a))-t2*(np.cos(2*k_x*a)*np.cos(2*k_y*a))
	ret[mask] = np.abs(ret[mask])
	return ret

def deltaFunc_swave(k_x,k_y,delta):
	try:
		return delta*np.ones(k_x.shape)
	except:
		return delta

def deltaFuncQ_dxy(kx,ky,Q):
	try:
		delta_kQ = np.zeros(kx.shape)
		mask1 = (kx > 0) & (ky > 0)
		mask2 = (kx < 0) & (ky > 0)
		mask3 = (kx < 0) & (ky < 0)
		mask4 = (kx > 0) & (ky < 0)
		delta_kQ[mask1] = deltaFunc_dxy(kx[mask1]+2*Q[0],ky[mask1]+2*Q[1],1)
		delta_kQ[mask2] = deltaFunc_dxy(kx[mask2]-2*Q[0],ky[mask2]+2*Q[1],1)
		delta_kQ[mask3] = deltaFunc_dxy(kx[mask3]-2*Q[0],ky[mask3]-2*Q[1],1)
		delta_kQ[mask4] = deltaFunc_dxy(kx[mask4]+2*Q[0],ky[mask4]-2*Q[1],1)
	except:
		if (kx > 0) and (ky > 0):
			delta_kQ = deltaFunc_dxy(kx+2*Q[0],ky+2*Q[1],1)
		elif (kx < 0) and (ky > 0):
			delta_kQ = deltaFunc_dxy(kx-2*Q[0],ky+2*Q[1],1)
		elif (kx < 0) and (ky < 0):
			delta_kQ = deltaFunc_dxy(kx-2*Q[0],ky-2*Q[1],1)
		elif (kx > 0) and (ky < 0):
			delta_kQ = deltaFunc_dxy(kx+2*Q[0],ky-2*Q[1],1)
	return delta_kQ
	
def deltaFunc_dxy(kx,ky,delta):
	#return delta*2.*np.sqrt(2.)*np.sin(k_x)*np.sin(k_y)
	kx_tmp = np.abs(kx)
	ky_tmp = np.abs(ky)
	kparr = 2*(kx_tmp+ky_tmp)/np.pi
	kperp = 2*(-kx_tmp+ky_tmp)/np.pi
	kparr_new = kparr % 1
	try:
		kx_new = np.zeros(kx.shape)
		ky_new = np.zeros(ky.shape)
		mask = (kparr - kparr_new) % 2. == 1.
		kx_new[mask] = -np.pi/4. + np.pi*(kparr_new[mask]-kperp[mask])/4.
		ky_new[mask] = -np.pi/4. + np.pi*(kparr_new[mask]+kperp[mask])/4.
		kx_new[~mask] = np.pi*(kparr_new[~mask]-kperp[~mask])/4.
		ky_new[~mask] = np.pi*(kparr_new[~mask]+kperp[~mask])/4.
	except:
		if (kparr - kparr_new) % 2. == 1.:
			kx_new = -np.pi/4. + np.pi*(kparr_new-kperp)/4.
			ky_new = -np.pi/4. + np.pi*(kparr_new+kperp)/4.
		else:
			kx_new = np.pi*(kparr_new-kperp)/4.
			ky_new = np.pi*(kparr_new+kperp)/4.
	return delta*2.*np.sqrt(2.)*np.abs(np.sin(kx_new)*np.sin(ky_new))*np.sign(np.cos(2*kx-np.pi/2)*np.cos(2*ky-np.pi/2))
		
def deltaFunc_dx2y2(kx,ky,delta):
	#return 0.5*delta*(np.cos(kx)-np.cos(ky))
	kx_tmp = np.abs(kx)
	ky_tmp = np.abs(ky)
	kparr = 2*(kx_tmp+ky_tmp)/np.pi
	kperp = 2*(-kx_tmp+ky_tmp)/np.pi
	kparr_new = kparr % 1
	try:
		kx_new = np.zeros(kx.shape)
		ky_new = np.zeros(ky.shape)
		mask = (kparr - kparr_new) % 2. == 1.
		kx_new[mask] = -np.pi/4. + np.pi*(kparr_new[mask]-kperp[mask])/4.
		ky_new[mask] = -np.pi/4. + np.pi*(kparr_new[mask]+kperp[mask])/4.
		kx_new[~mask] = np.pi*(kparr_new[~mask]-kperp[~mask])/4.
		ky_new[~mask] = np.pi*(kparr_new[~mask]+kperp[~mask])/4.
	except:
		if (kparr - kparr_new) % 2. == 1.:
			kx_new = -np.pi/4. + np.pi*(kparr_new-kperp)/4.
			ky_new = -np.pi/4. + np.pi*(kparr_new+kperp)/4.
		else:
			kx_new = np.pi*(kparr_new-kperp)/4.
			ky_new = np.pi*(kparr_new+kperp)/4.
	return 0.5*delta*np.abs(np.cos(kx_new)-np.cos(ky_new))*np.sign(np.cos(kx)-np.cos(ky))

def deltaFunc_dxyCircular(phi):
	return np.sqrt(2)*np.sin(2.*phi)

def deltaFunc_swaveCircular(phi):
	try:
		return np.ones(phi.shape)
	except:
		return 1.

def M_iso(kx,ky,M):
	try:
		return M*np.ones(k_x.shape)
	except:
		return M

def M_aniso(kx,ky,M):
	ret = np.zeros(kx.shape)
	mask = ((kx > 0.) & (ky > 0.)) | ((kx < 0.) & (ky < 0.))
	ret[mask] = M
	return ret

def fermi_deriv(E,T):
	return 1/(T*(2.*np.cosh(E/(2.*T)))**2.)

def eigenvalues4(xi_k, xi_kQ, delta_k, delta_kQ, M):
	if(M == 0.) and (delta_k == 0.) and (delta_kQ == 0.):
		ret1 = xi_k
		ret2 = -xi_k
		ret3 = xi_kQ
		ret4 = -xi_kQ
			
	elif(M == 0.) and (delta_k != 0. or delta_kQ != 0.):
		Ek = np.sqrt((xi_k**2) + (delta_k**2))
		EkQ = np.sqrt((xi_kQ**2) + (delta_kQ**2))
		
		ret1 = Ek
		ret2 = -Ek
		ret3 = EkQ
		ret4 = -EkQ
	elif(M != 0.) and (delta_k == 0.) and (delta_kQ == 0.):
		xi_k_plus = (xi_k + xi_kQ)/2.
		xi_k_minus = (xi_k - xi_kQ)/2.
		
		ret1 = xi_k_plus + np.sqrt((xi_k_minus)**2. + M**2.)
		ret2 = -(xi_k_plus + np.sqrt((xi_k_minus)**2. + M**2.))
		ret3 = xi_k_plus - np.sqrt((xi_k_minus)**2. + M**2.)
		ret4 = -(xi_k_plus - np.sqrt((xi_k_minus)**2. + M**2.))
	else:
		xi_k_plus = (xi_k + xi_kQ)/2.
		xi_k_minus = (xi_k - xi_kQ)/2.
		delta_k_plus = (delta_k + delta_kQ)/2.
		delta_k_minus = (delta_k - delta_kQ)/2.
	
		Gamma_k = (xi_k_plus**2) + (xi_k_minus**2) + (delta_k_plus**2) + (delta_k_minus**2) + (M**2)
		Lambda_k = np.sqrt((xi_k_plus*xi_k_minus + delta_k_plus*delta_k_minus)**2 + (M**2)*((xi_k_plus**2) + (delta_k_plus**2)))
	
		E1_plus = np.sqrt(Gamma_k + 2.*Lambda_k)
		E1_minus = -E1_plus
		if Gamma_k > 2.*Lambda_k:
			E2_plus = np.sqrt(Gamma_k - 2.*Lambda_k)
		else:
			E2_plus = 0.
		E2_minus = -E2_plus
		
		ret1 = E1_plus
		ret2 = E1_minus
		ret3 = E2_plus
		ret4 = E2_minus

	return ret1,ret2,ret3,ret4

def velocityTB(kx,ky,a,t1,t2):
	
	vel_x = 2.*a*(t1*np.sin(2.*kx*a)+t2*np.sin(2.*kx*a)*np.cos(2.*ky*a))
	vel_y = 2.*a*(t1*np.sin(2.*ky*a)+t2*np.cos(2.*kx*a)*np.sin(2.*ky*a))
	
	#vel_x = a*(t1*np.sin(kx*a)+t2*np.sin(kx*a)*np.cos(ky*a))
	#vel_y = a*(t1*np.sin(ky*a)+t2*np.cos(kx*a)*np.sin(ky*a))
	return vel_x, vel_y

def energySelfConsistency(kx,ky,dispersion,t1,t2,M,Q,M_order,delta,deltaFunc):
	xi_k = dispersion(0.,kx,ky,1.,t1,t2)
	xi_kQ = dispersion(0.,kx+Q[0],ky+Q[1],1.,t1,t2)
	delta_k = deltaFunc(kx,ky,delta)
	delta_kQ = deltaFunc(kx+Q[0],ky+Q[1],delta)
	M_k = M_order(kx,ky,M)

	E1plus,E1minus,E2plus,E2minus = eigenvalues4(xi_k,xi_kQ,delta_k,delta_kQ,M_k)

	return E1plus, E2plus
	
def deltaGaussHerm(Ek,Ek_prime,f):
	hermiteZero = 1
	hermiteTwo = 4.*(((Ek - Ek_prime)**2)/(2*f*f)) - 2.
	coeffZero = 1./np.sqrt(np.pi)
	coeffTwo = -1./(4.*np.sqrt(np.pi))

	gauss = np.exp(-((Ek - Ek_prime)**2)/(2*f*f))
	return (1./(f*np.sqrt(2.)))*gauss*(coeffZero*hermiteZero + coeffTwo*hermiteTwo)
	
def energyPureSDW(mu,kx,ky,a,t1,t2,M,Q):
	xi_k = dispersionTB(mu,kx,ky,a,t1,t2)
	xi_kQ = dispersionTB(mu,kx+Q[0],ky+Q[1],a,t1,t2)
	
	xi_plus = .5*(xi_k + xi_kQ)
	xi_minus = .5*(xi_k - xi_kQ)
	
	Gamma_k = (xi_plus)**2 + (xi_minus)**2 + M**2
	Lambda_k = np.sqrt((xi_plus*xi_minus)**2 + (M*xi_plus)**2)
	
	E1 = np.sqrt(Gamma_k + 2*Lambda_k)
	E2 = np.sqrt(np.abs(Gamma_k - 2*Lambda_k))
	
	return E1,E2

def mixingNodeSelfConsistency(kx,ky,disp,deltaFunc,t1,t2,M,Q,delta):
	xi_k = disp(0.,kx,ky,1.,t1,t2)
	xi_kQ = disp(0.,kx+Q[0],ky+Q[1],1.,t1,t2)
	
	xi_plus = .5*(xi_k+xi_kQ)
	xi_minus = .5*(xi_k-xi_kQ)
	
	delta_k = deltaFunc(kx,ky,delta)
	
	return [M - np.sqrt((xi_plus)**2 + (delta_k)**2),xi_minus]
	
def mixingNodeLocations(disp,deltaFunc,t1,t2,M,Q,delta):
	k1 = fsolve(lambda x: mixingNodeSelfConsistency(x[0],x[1],disp,deltaFunc,t1,t2,M,Q,delta),[.4,1.1])
	kx1 = k1[0]
	ky1 = k1[1]
	
	k2 = fsolve(lambda x: mixingNodeSelfConsistency(x[0],x[1],disp,deltaFunc,t1,t2,M,Q,delta),[.9,.7])
	kx2 = k2[0]
	ky2 = k2[1]
	
	return kx1,ky1,kx2,ky2
	
def energyMixed(sc_order,disp,kx,ky,deltaFunc,t1,t2,M,Q,delta):
	xi_k = disp(0.,kx,ky,1.,t1,t2)
	xi_kQ = disp(0.,kx+Q[0],ky+Q[1],1.,t1,t2)
	
	delta_k = deltaFunc(kx,ky,delta)
	if sc_order == 'dxy':
		delta_kQ = delta*deltaFuncQ_dxy(kx,ky,Q)
	elif sc_order == 'swave':
		delta_kQ = deltaFunc(kx+Q[0],ky+Q[1],delta)
	
	xi_k_plus = (xi_k + xi_kQ)/2.
	xi_k_minus = (xi_k - xi_kQ)/2.
	delta_k_plus = (delta_k + delta_kQ)/2.
	delta_k_minus = (delta_k - delta_kQ)/2.
	
	Gamma_k = (xi_k_plus**2) + (xi_k_minus**2) + (delta_k_plus**2) + (delta_k_minus**2) + (M**2)
	Lambda_k = np.sqrt((xi_k_plus*xi_k_minus + delta_k_plus*delta_k_minus)**2 + (M**2)*((xi_k_plus**2) + (delta_k_plus**2)))
	
	if Gamma_k > 2.*Lambda_k:
		E2_plus = np.sqrt(Gamma_k - 2.*Lambda_k)
	else:
		E2_plus = 0.
	return E2_plus
	
def mixingNodeGrid(sc_order,disp,deltaFunc,t1,t2,M,Q,delta,E_c,N_phi,N_E):
	E_array = np.linspace(1e-4,E_c,N_E)
	phi_array = np.linspace(1e-8,2*np.pi,N_phi)
	phi_grid,E_grid = np.meshgrid(phi_array,E_array)
	
	kxNode1,kyNode1,kxNode2,kyNode2 = mixingNodeLocations(disp,deltaFunc,t1,t2,M,Q,delta)
	
	kr1 = np.zeros(E_grid.shape)
	kx1 = np.zeros(E_grid.shape)
	ky1 = np.zeros(E_grid.shape)
	
	kr2 = np.zeros(E_grid.shape)
	kx2 = np.zeros(E_grid.shape)
	ky2 = np.zeros(E_grid.shape)
	
	for i in range(0,N_E):
		for j in range(0,N_phi):
			kr1[i][j] = fsolve(lambda k: (energyMixed(sc_order,disp,kxNode1 + k*np.cos(phi_grid[i][j]),kyNode1 + k*np.sin(phi_grid[i][j]),deltaFunc,t1,t2,M,Q,delta) - E_grid[i][j])**2,1e-3)
			kr2[i][j] = fsolve(lambda k: (energyMixed(sc_order,disp,kxNode2 + k*np.cos(phi_grid[i][j]),kyNode2 + k*np.sin(phi_grid[i][j]),deltaFunc,t1,t2,M,Q,delta) - E_grid[i][j])**2,1e-3)
			
			kx1[i][j] = kxNode1 + np.abs(kr1[i][j])*np.cos(phi_grid[i][j])
			ky1[i][j] = kyNode1 + np.abs(kr1[i][j])*np.sin(phi_grid[i][j])
			
			kx2[i][j] = kxNode2 + np.abs(kr2[i][j])*np.cos(phi_grid[i][j])
			ky2[i][j] = kyNode2 + np.abs(kr2[i][j])*np.sin(phi_grid[i][j])
			
		if i == N_E - 1:
			f1 = interp1d(phi_grid[i],kr1[i],kind = 'cubic')
			f2 = interp1d(phi_grid[i],kr2[i],kind = 'cubic')
	dkl1 = []
	dkl2 = []
	for i in range(0,N_E):
		dkx1 = np.gradient(kx1[i])
		dky1 = np.gradient(ky1[i])
		dkl1.append(np.sqrt(dkx1**2 + dky1**2))
		
		dkx2 = np.gradient(kx2[i])
		dky2 = np.gradient(ky2[i])
		dkl2.append(np.sqrt(dkx2**2 + dky2**2))
	
	dkl1 = np.array(dkl1)
	dkl2 = np.array(dkl2)

	xi_k1 = disp(0.,kx1,ky1,1.,t1,t2)
	xi_kQ1 = disp(0.,kx1+Q[0],ky1+Q[1],1.,t1,t2)
	
	xi_k2 = disp(0.,kx2,ky2,1.,t1,t2)
	xi_kQ2 = disp(0.,kx2+Q[0],ky2+Q[1],1.,t1,t2)
	
	dxi1 = np.zeros(kx1.shape)
	dxiQ1 = np.zeros(kx1.shape)
	dxi2 = np.zeros(kx2.shape)
	dxiQ2 = np.zeros(kx2.shape)
	
	for i in range(0,N_phi):
		dxi1[:,i] = np.abs(np.gradient(xi_k1[:,i]))
		dxiQ1[:,i] = np.abs(np.gradient(xi_kQ1[:,i]))
		
		dxi2[:,i] = np.abs(np.gradient(xi_k2[:,i]))
		dxiQ2[:,i] = np.abs(np.gradient(xi_kQ2[:,i]))
	
	dxiNode1 = np.zeros(dxi1.shape)
	dxiNode2 = np.zeros(dxi2.shape)
	
	mask1 = ky1 < np.pi/2 - kx1
	mask2 = ky2 < np.pi/2 - kx2
	
	dxiNode1[mask1] = dxi1[mask1]
	dxiNode1[~mask1] = dxiQ1[~mask1]
	
	dxiNode2[mask2] = dxi2[mask2]
	dxiNode2[~mask2] = dxiQ2[~mask2]
	
	kx = np.concatenate((kx1,kx2))
	ky = np.concatenate((ky1,ky2))
	dkl = np.concatenate((dkl1,dkl2))
	dxi = np.concatenate((dxiNode1,dxiNode2))
	
	vx,vy = velocityTB(kx,ky,1,t1,t2)
	vxQ,vyQ = velocityTB(kx+Q[0],ky+Q[1],1,t1,t2)
	
	return kx.flatten(),ky.flatten(),dkl.flatten(),dxi.flatten(),vx.flatten(),vy.flatten(),vxQ.flatten(),vyQ.flatten(),f1,kxNode1,kyNode1,f2,kxNode2,kyNode2

def xi_grid(xi,t1,t2,theta):
	kF = np.zeros(theta.shape)
	for i in range(0,len(theta)):
		kF[i] = fsolve(lambda k: (xi - dispersionTB(0.,k*np.cos(theta[i]),k*np.sin(theta[i]),1.,t1,t2))**2,1.)
	kx = kF*np.cos(theta)
	ky = kF*np.sin(theta)
	
	return kx,ky

def xiQ_grid(xiQ,t1,t2,Q,theta):
	kF = np.zeros(theta.shape)
	for i in range(0,len(theta)):
		kF[i] = fsolve(lambda k: (xiQ - dispersionTB(0.,np.pi/2 - k*np.cos(theta[i])+Q[0],np.pi/2 - k*np.sin(theta[i])+Q[1],1.,t1,t2))**2,1.)
	kx = np.pi/2 - kF*np.cos(theta)
	ky = np.pi/2 - kF*np.sin(theta)
	
	return kx,ky

def xi_grid_polar(xi,t1,t2,theta):
	kF = np.zeros(theta.shape)
	for i in range(0,len(theta)):
		kF[i] = fsolve(lambda k: (xi - dispersionTB(0.,k*np.cos(theta[i]),k*np.sin(theta[i]),1.,t1,t2))**2,1.)
	
	return kF

def xiQ_grid_polar(xiQ,t1,t2,Q,theta):
	kF = np.zeros(theta.shape)
	for i in range(0,len(theta)):
		kF[i] = fsolve(lambda k: (xiQ - dispersionTB(0.,np.pi/2 - k*np.cos(theta[i])+Q[0],np.pi/2 - k*np.sin(theta[i])+Q[1],1.,t1,t2))**2,1.)
	
	return kF

def ky1SDW_E2(kx,E,t1,t2,M):
	ky = (1/2)*np.arccos((-4*t1**2*np.cos(2*kx) - 4*E*t2*np.cos(2*kx) + np.sqrt((-4*t1**2*np.cos(2*kx) - 4*E*t2*np.cos(2*kx))**2 - 4*(2*E**2 - 2*M**2 - t1**2 - t1**2*np.cos(4*kx))*(-2*t1**2 + t2**2 + t2**2*np.cos(4*kx))))/(2*(2*t1**2 - t2**2 - t2**2*np.cos(4*kx))))
	
	return ky

def ky2SDW_E2(kx,E,t1,t2,M):
	ky = (1/2)*np.arccos((4*t1**2*np.cos(2*kx) + 4*E*t2*np.cos(2*kx) + np.sqrt((-4*t1**2*np.cos(2*kx) - 4*E*t2*np.cos(2*kx))**2 - 4*(2*E**2 - 2*M**2 - t1**2 - t1**2*np.cos(4*kx))*(-2*t1**2 + t2**2 + t2**2*np.cos(4*kx))))/(2*(-2*t1**2 + t2**2 + t2**2*np.cos(4*kx))))
	
	return ky

def ky3SDW_E2(kx,E,t1,t2,M):
	ky = (1/2)*np.arccos((-4*t1**2*np.cos(2*kx) + 4*E*t2*np.cos(2*kx) + np.sqrt((-4*t1**2*np.cos(2*kx) + 4*E*t2*np.cos(2*kx))**2 - 4*(2*E**2 - 2*M**2 - t1**2 - t1**2*np.cos(4*kx))*(-2*t1**2 + t2**2 + t2**2*np.cos(4*kx))))/(2*(2*t1**2 - t2**2 - t2**2*np.cos(4*kx))))
	
	return ky

def ky4SDW_E2(kx,E,t1,t2,M):
	ky = (1/2)*np.arccos((4*t1**2*np.cos(2*kx) - 4*E*t2*np.cos(2*kx) + np.sqrt((-4*t1**2*np.cos(2*kx) + 4*E*t2*np.cos(2*kx))**2 - 4*(2*E**2 - 2*M**2 - t1**2 - t1**2*np.cos(4*kx))*(-2*t1**2 + t2**2 + t2**2*np.cos(4*kx))))/(2*(-2*t1**2 + t2**2 + t2**2*np.cos(4*kx))))
	
	return ky
	
def thetaCrit1_E2(E,t1,t2,M):
	err = 1e-6
	
	kx_crit_left = (1/2)*np.arccos(np.sqrt(-(M**2/t1**2) - (2*E)/t2 + (M*np.sqrt(4*t1**4 + 4*E*t1**2*t2 + M**2*t2**2))/(t1**2*t2))/np.sqrt(2)) - err
	kx_crit_right = (1/2)*np.arccos(-(np.sqrt(-(M**2/t1**2) - (2*E)/t2 + (M*np.sqrt(4*t1**4 + 4*E*t1**2*t2 + M**2*t2**2))/(t1**2*t2))/np.sqrt(2))) + err
	
	ky_crit_left = ky1SDW_E2(kx_crit_left,E,t1,t2,M)
	ky_crit_right = ky1SDW_E2(kx_crit_right,E,t1,t2,M)
	
	theta_crit_left = np.arctan(ky_crit_left/kx_crit_left)
	theta_crit_right = np.arctan(ky_crit_right/kx_crit_right)
	
	return theta_crit_left,theta_crit_right

def thetaCrit3_E2(E,t1,t2,M):
	err = 1e-6
	
	kx_crit_left = (1/2)*np.arccos(np.sqrt(-(M**2/t1**2) + (2*E)/t2 + np.sqrt(M**2*t2**2*(4*t1**4 - 4*E*t1**2*t2 + M**2*t2**2))/(t1**2*t2**2))/np.sqrt(2)) - err
	kx_crit_right = (1/2)*np.arccos(-(np.sqrt(-(M**2/t1**2) + (2*E)/t2 + np.sqrt(M**2*t2**2*(4*t1**4 - 4*E*t1**2*t2 + M**2*t2**2))/(t1**2*t2**2))/np.sqrt(2))) + err
	
	ky_crit_left = ky3SDW_E2(kx_crit_left,E,t1,t2,M)
	ky_crit_right = ky3SDW_E2(kx_crit_right,E,t1,t2,M)
	
	theta_crit_left = np.arctan(ky_crit_left/kx_crit_left)
	theta_crit_right = np.arctan(ky_crit_right/kx_crit_right)
	
	return theta_crit_left,theta_crit_right
	
def thetaCrit3_E1(E,t1,t2,M):
	err = 1e-6
	
	kx_crit_left = (1/2)*np.arccos(np.sqrt(-(M**2/t1**2) + (2*E)/t2 - np.sqrt(M**2*t2**2*(4*t1**4 - 4*E*t1**2*t2 + M**2*t2**2))/(t1**2*t2**2))/np.sqrt(2)) + err
	kx_crit_right = (1/2)*np.arccos(-(np.sqrt(-(M**2/t1**2) + (2*E)/t2 - np.sqrt(M**2*t2**2*(4*t1**4 - 4*E*t1**2*t2 + M**2*t2**2))/(t1**2*t2**2))/np.sqrt(2))) - err
		
	ky_crit_left = ky3SDW_E2(kx_crit_left,E,t1,t2,M)
	ky_crit_right = ky3SDW_E2(kx_crit_right,E,t1,t2,M)
	
	theta_crit_left = np.arctan(ky_crit_left/kx_crit_left)
	theta_crit_right = np.arctan(ky_crit_right/kx_crit_right)
	
	if np.isnan(kx_crit_left):
		theta_crit_left = np.pi/2.
	if np.isnan(kx_crit_right):
		theta_crit_left = 0.
	
	return theta_crit_left,theta_crit_right
	
def SDW_equipotential_E1(t1,t2,M,Q,E,theta):
	dtheta = np.abs(np.gradient(theta))
	fac = 1e-5
	
	if np.abs(E) >= energyPureSDW(0,np.pi/4,np.pi/4,1,t1,t2,M,Q)[0]:
		
		xi = dispersionTB(0,0,ky3SDW_E2(0,np.abs(E),t1,t2,M),1,t1,t2)
		
		theta_crit_left,theta_crit_right = thetaCrit3_E1(np.abs(E),t1,t2,M)
		
		kF_normal = xi_grid_polar(xi,t1,t2,theta)
		kFQ_normal = xiQ_grid_polar(xi,t1,t2,Q,theta)
		kF_E1 = np.zeros(kF_normal.shape)
		kx_E1 = np.zeros(kF_normal.shape)
		ky_E1 = np.zeros(kF_normal.shape)
		kFQ_E1 = np.zeros(kFQ_normal.shape)
		kxQ_E1 = np.zeros(kFQ_normal.shape)
		kyQ_E1 = np.zeros(kFQ_normal.shape)
		for i in range(0,len(theta)):
			if theta[i] < theta_crit_left and theta[i] > theta_crit_right:
				#kF_E1[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,k*np.cos(theta[i]),k*np.sin(theta[i]),1.,t1,t2,M,Q)[0])**2.,.95*kF_normal[i],factor = fac)[0]
				#kFQ_E1[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,np.pi/2 - k*np.cos(theta[i]) + Q[0],np.pi/2 - k*np.sin(theta[i]) + Q[1],1.,t1,t2,M,Q)[0])**2.,.95*kFQ_normal[i],factor = fac)[0]
				kF_E1[i] = fsolve(lambda k: (np.abs(E) + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) - np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,.95*kF_normal[i],factor = fac)[0]
				kFQ_E1[i] = fsolve(lambda k: (np.abs(E) + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) - np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,.95*kFQ_normal[i],factor = fac)[0]
				kx_E1[i] = kF_E1[i]*np.cos(theta[i])
				ky_E1[i] = kF_E1[i]*np.sin(theta[i])
				kxQ_E1[i] = np.pi/2 - kFQ_E1[i]*np.cos(theta[i])
				kyQ_E1[i] = np.pi/2 - kFQ_E1[i]*np.sin(theta[i])
			else:
				ky_E1[i] = -100
				kyQ_E1[i] = -100
	
	dl_E1 = kF_E1*dtheta/np.cos(dtheta)
	dlQ_E1 = kFQ_E1*dtheta/np.cos(dtheta)
	
	return list(kx_E1),list(ky_E1),list(dl_E1),list(kxQ_E1),list(kyQ_E1),list(dlQ_E1)
	
def SDW_equipotential_E2(t1,t2,M,Q,E,theta):
	dtheta = np.abs(np.gradient(theta))
	fac = 1e-5
	if E <= 0:
		xi = dispersionTB(0,0,ky1SDW_E2(0,np.abs(E),t1,t2,M),1,t1,t2)
		if np.abs(E) > energyPureSDW(0,np.pi/4,np.pi/4,1,t1,t2,M,Q)[0]:
			kF_normal = xi_grid_polar(xi,t1,t2,theta)
			kFQ_normal = xiQ_grid_polar(xi,t1,t2,Q,theta)
			kF_E2 = np.zeros(kF_normal.shape)
			kx_E2 = np.zeros(kF_normal.shape)
			ky_E2 = np.zeros(kF_normal.shape)
			kFQ_E2 = np.zeros(kFQ_normal.shape)
			kxQ_E2 = np.zeros(kFQ_normal.shape)
			kyQ_E2 = np.zeros(kFQ_normal.shape)
			for i in range(0,len(theta)):
				#kF_E2[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,k*np.cos(theta[i]),k*np.sin(theta[i]),1.,t1,t2,M,Q)[1])**2.,kF_normal[i],factor = fac)[0]
				#kFQ_E2[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,np.pi/2 - k*np.cos(theta[i]) + Q[0],np.pi/2 - k*np.sin(theta[i]) + Q[1],1.,t1,t2,M,Q)[1])**2.,kFQ_normal[i],factor = fac)[0]
				kF_E2[i] = np.abs(fsolve(lambda k: (E + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) + np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,kF_normal[i],factor = fac)[0])
				kFQ_E2[i] = np.abs(fsolve(lambda k: (E + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) + np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,kFQ_normal[i],factor = fac)[0])
			kx_E2 = kF_E2*np.cos(theta)
			ky_E2 = kF_E2*np.sin(theta)
			kxQ_E2 = np.pi/2 - kFQ_E2*np.cos(theta)
			kyQ_E2 = np.pi/2 - kFQ_E2*np.sin(theta)
		else:
			
			theta_crit_left,theta_crit_right = thetaCrit1_E2(np.abs(E),t1,t2,M)
			
			kF_normal = xi_grid_polar(xi,t1,t2,theta)
			kFQ_normal = xiQ_grid_polar(xi,t1,t2,Q,theta)
			kF_E2 = np.zeros(kF_normal.shape)
			kx_E2 = np.zeros(kF_normal.shape)
			ky_E2 = np.zeros(kF_normal.shape)
			kFQ_E2 = np.zeros(kFQ_normal.shape)
			kxQ_E2 = np.zeros(kFQ_normal.shape)
			kyQ_E2 = np.zeros(kFQ_normal.shape)
			for i in range(0,len(theta)):
				if theta[i] > theta_crit_left or theta[i] < theta_crit_right:
					#kF_E2[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,k*np.cos(theta[i]),k*np.sin(theta[i]),1.,t1,t2,M,Q)[1])**2.,kF_normal[i],factor = fac)[0]
					#kFQ_E2[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,np.pi/2 - k*np.cos(theta[i]) + Q[0],np.pi/2 - k*np.sin(theta[i]) + Q[1],1.,t1,t2,M,Q)[1])**2.,kFQ_normal[i],factor = fac)[0]
					kF_E2[i] = np.abs(fsolve(lambda k: (E + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) + np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,kF_normal[i],factor = fac)[0])
					kFQ_E2[i] = np.abs(fsolve(lambda k: (E + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) + np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,kFQ_normal[i],factor = fac)[0])
					kx_E2[i] = kF_E2[i]*np.cos(theta[i])
					ky_E2[i] = kF_E2[i]*np.sin(theta[i])
					kxQ_E2[i] = np.pi/2 - kFQ_E2[i]*np.cos(theta[i])
					kyQ_E2[i] = np.pi/2 - kFQ_E2[i]*np.sin(theta[i])
				else:
					ky_E2[i] = -100
					kyQ_E2[i] = -100
	else:
		if np.abs(E) <= energyPureSDW(0,np.pi/2,0,1,t1,t2,M,Q)[1]:
			
			xi = dispersionTB(0,0,ky3SDW_E2(0,np.abs(E),t1,t2,M),1,t1,t2)
			
			theta_crit_left,theta_crit_right = thetaCrit3_E2(np.abs(E),t1,t2,M)
			
			kF_normal = xi_grid_polar(xi,t1,t2,theta)
			kFQ_normal = xiQ_grid_polar(xi,t1,t2,Q,theta)
			kF_E2 = np.zeros(kF_normal.shape)
			kx_E2 = np.zeros(kF_normal.shape)
			ky_E2 = np.zeros(kF_normal.shape)
			kFQ_E2 = np.zeros(kFQ_normal.shape)
			kxQ_E2 = np.zeros(kFQ_normal.shape)
			kyQ_E2 = np.zeros(kFQ_normal.shape)
			for i in range(0,len(theta)):
				if theta[i] > theta_crit_left or theta[i] < theta_crit_right:
					#kF_E2[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,k*np.cos(theta[i]),k*np.sin(theta[i]),1.,t1,t2,M,Q)[1])**2.,1.005*kF_normal[i],factor = fac)[0]
					#kFQ_E2[i] = fsolve(lambda k: (np.abs(E) - energyPureSDW(0.,np.pi/2 - k*np.cos(theta[i]) + Q[0],np.pi/2 - k*np.sin(theta[i]) + Q[1],1.,t1,t2,M,Q)[1])**2.,1.005*kFQ_normal[i],factor = fac)[0]
					kF_E2[i] = np.abs(fsolve(lambda k: (E + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) + np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,1.005*kF_normal[i],factor = fac)[0])
					kFQ_E2[i] = np.abs(fsolve(lambda k: (E + t2*np.cos(2*k*np.cos(theta[i]))*np.cos(2*k*np.sin(theta[i])) + np.sqrt((t1*np.cos(2*k*np.cos(theta[i]))+t1*np.cos(2*k*np.sin(theta[i])))**2 + M**2))**2.,1.005*kFQ_normal[i],factor = fac)[0])
					kx_E2[i] = kF_E2[i]*np.cos(theta[i])
					ky_E2[i] = kF_E2[i]*np.sin(theta[i])
					kxQ_E2[i] = np.pi/2 - kFQ_E2[i]*np.cos(theta[i])
					kyQ_E2[i] = np.pi/2 - kFQ_E2[i]*np.sin(theta[i])
				else:
					ky_E2[i] = -100
					kyQ_E2[i] = -100
	
	dl_E2 = kF_E2*dtheta/np.cos(dtheta)
	dlQ_E2 = kFQ_E2*dtheta/np.cos(dtheta)
	
	return list(kx_E2),list(ky_E2),list(dl_E2),list(kxQ_E2),list(kyQ_E2),list(dlQ_E2)

def SDW_grid_new(t1,t2,M,Q,E_c,N_E,N_theta):
	err = 1e-8
	
	if E_c > energyPureSDW(0,np.pi/2,0,1,t1,t2,M,Q)[1]:
		E_array = np.linspace(-E_c,energyPureSDW(0,np.pi/2,0,1,t1,t2,M,Q)[1],N_E)
	else:
		E_array = np.linspace(-E_c,E_c,N_E)
	
	theta = np.linspace(np.pi/2-err,err,N_theta)
	
	kx_E2 = []
	ky_E2 = []
	dl_E2 = []
	vx_E2 = []
	vy_E2 = []
	vxQ_E2 = []
	vyQ_E2 = []
	
	kxQ_E2 = []
	kyQ_E2 = []
	dlQ_E2 = []
	vx_E2Q = []
	vy_E2Q = []
	vxQ_E2Q = []
	vyQ_E2Q = []
	
	for i in range(0,N_E):
		kx,ky,dl,kxQ,kyQ,dlQ = SDW_equipotential_E2(t1,t2,M,Q,E_array[i],theta)
		
		kx_E2.append(kx)
		ky_E2.append(ky)
		dl_E2.append(dl)
		
		kxQ_E2.append(kxQ)
		kyQ_E2.append(kyQ)
		dlQ_E2.append(dlQ)
	
	E_array = np.linspace(energyPureSDW(0,np.pi/4,np.pi/4,1,t1,t2,M,Q)[0]+.001,energyPureSDW(0,np.pi/4,np.pi/4,1,t1,t2,M,Q)[0]+E_c,N_E)
	
	kx_E1 = []
	ky_E1 = []
	dl_E1 = []
	vx_E1 = []
	vy_E1 = []
	vxQ_E1 = []
	vyQ_E1 = []
	
	kxQ_E1 = []
	kyQ_E1 = []
	dlQ_E1 = []
	vx_E1Q = []
	vy_E1Q = []
	vxQ_E1Q = []
	vyQ_E1Q = []
	
	for i in range(0,N_E):
		kx,ky,dl,kxQ,kyQ,dlQ = SDW_equipotential_E1(t1,t2,M,Q,E_array[i],theta)
		
		kx_E1.append(kx)
		ky_E1.append(ky)
		dl_E1.append(dl)
		
		kxQ_E1.append(kxQ)
		kyQ_E1.append(kyQ)
		dlQ_E1.append(dlQ)
	
	kx_xi,ky_xi = xi_grid(0,t1,t2,theta)
	kx_xiQ,ky_xiQ = xiQ_grid(0,t1,t2,Q,theta)
	
	vx_xi,vy_xi = velocityTB(kx_xi,ky_xi,1,t1,t2)
	vx_xiQ,vy_xiQ = velocityTB(kx_xiQ,ky_xiQ,1,t1,t2)
	
	vxQ_xi,vyQ_xi = velocityTB(kx_xi+Q[0],ky_xi+Q[1],1,t1,t2)
	vxQ_xiQ,vyQ_xiQ = velocityTB(kx_xiQ+Q[0],ky_xiQ+Q[1],1,t1,t2)
	
	for i in range(0,N_E):
		vx_E2.append(vx_xi)
		vy_E2.append(vy_xi)
		vxQ_E2.append(vxQ_xi)
		vyQ_E2.append(vyQ_xi)
		
		vx_E2Q.append(vx_xiQ)
		vy_E2Q.append(vy_xiQ)
		vxQ_E2Q.append(vxQ_xiQ)
		vyQ_E2Q.append(vyQ_xiQ)
		
		vx_E1.append(vx_xi)
		vy_E1.append(vy_xi)
		vxQ_E1.append(vxQ_xi)
		vyQ_E1.append(vyQ_xi)
		
		vx_E1Q.append(vx_xiQ)
		vy_E1Q.append(vy_xiQ)
		vxQ_E1Q.append(vxQ_xiQ)
		vyQ_E1Q.append(vyQ_xiQ)
	
	kx_E2 = np.array(kx_E2)
	ky_E2 = np.array(ky_E2)
	dl_E2 = np.array(dl_E2)
	kxQ_E2 = np.array(kxQ_E2)
	kyQ_E2 = np.array(kyQ_E2)
	dlQ_E2 = np.array(dlQ_E2)
	
	vx_E2 = np.array(vx_E2)
	vy_E2 = np.array(vy_E2)
	vxQ_E2 = np.array(vxQ_E2)
	vyQ_E2 = np.array(vyQ_E2)
	
	vx_E2Q = np.array(vx_E2Q)
	vy_E2Q = np.array(vy_E2Q)
	vxQ_E2Q = np.array(vxQ_E2Q)
	vyQ_E2Q = np.array(vyQ_E2Q)
	
	kx_E1 = np.array(kx_E1)
	ky_E1 = np.array(ky_E1)
	dl_E1 = np.array(dl_E1)
	kxQ_E1 = np.array(kxQ_E1)
	kyQ_E1 = np.array(kyQ_E1)
	dlQ_E1 = np.array(dlQ_E1)
	
	vx_E1 = np.array(vx_E1)
	vy_E1 = np.array(vy_E1)
	vxQ_E1 = np.array(vxQ_E1)
	vyQ_E1 = np.array(vyQ_E1)
	
	vx_E1Q = np.array(vx_E1Q)
	vy_E1Q = np.array(vy_E1Q)
	vxQ_E1Q = np.array(vxQ_E1Q)
	vyQ_E1Q = np.array(vyQ_E1Q)
	
	dxi_E2 = np.zeros(kx_E2.shape)
	dxiQ_E2 = np.zeros(kxQ_E2.shape)
	
	dxi_E1 = np.zeros(kx_E1.shape)
	dxiQ_E1 = np.zeros(kxQ_E1.shape)
	
	xi_E2 = dispersionTB(0,kx_E2,ky_E2,1,t1,t2)
	xiQ_E2 = dispersionTB(0,kxQ_E2+Q[0],kyQ_E2+Q[1],1,t1,t2)
	
	xi_E1 = dispersionTB(0,kx_E1,ky_E1,1,t1,t2)
	xiQ_E1 = dispersionTB(0,kxQ_E1+Q[0],kyQ_E1+Q[1],1,t1,t2)
	
	for i in range(0,N_theta):
		mask_xi = ky_E2[:,i] != -100
		mask_xiQ = kyQ_E2[:,i] != -100
		
		if len(xi_E2[:,i][mask_xi]) > 1:
			dxi_E2[:,i][mask_xi] = np.abs(np.gradient(xi_E2[:,i][mask_xi]))
		if len(xiQ_E2[:,i][mask_xiQ]) > 1:
			dxiQ_E2[:,i][mask_xiQ] = np.abs(np.gradient(xiQ_E2[:,i][mask_xiQ]))
	
	for i in range(0,N_theta):
		mask_xi = ky_E1[:,i] != -100
		mask_xiQ = kyQ_E1[:,i] != -100
		
		if len(xi_E1[:,i][mask_xi]) > 1:
			dxi_E1[:,i][mask_xi] = np.abs(np.gradient(xi_E1[:,i][mask_xi]))
		if len(xiQ_E2[:,i][mask_xiQ]) > 1:
			dxiQ_E1[:,i][mask_xiQ] = np.abs(np.gradient(xiQ_E1[:,i][mask_xiQ]))
	"""
	plt.scatter(kx_E2[ky_E2 != -100],ky_E2[ky_E2 != -100],c = xi_E2[ky_E2 != -100])
	plt.colorbar()
	plt.scatter(kxQ_E2[kyQ_E2 != -100],kyQ_E2[kyQ_E2 != -100],c = xiQ_E2[kyQ_E2 != -100])
	plt.colorbar()
	plt.show()
	"""
	
	kx2 = np.concatenate((kx_E2,kxQ_E2))
	ky2 = np.concatenate((ky_E2,kyQ_E2))
	dkl2 = np.concatenate((dl_E2,dlQ_E2))
	dxi2 = np.concatenate((dxi_E2,dxiQ_E2))

	vx_n2 = np.concatenate((vx_E2,vx_E2Q))
	vy_n2 = np.concatenate((vy_E2,vy_E2Q))
	vx_nQ2 = np.concatenate((vxQ_E2,vxQ_E2Q))
	vy_nQ2 = np.concatenate((vyQ_E2,vyQ_E2Q))
	
	mask2 = ky2 != -100
	
	kx2 = kx2[mask2]
	ky2 = ky2[mask2]
	dkl2 = dkl2[mask2]
	dxi2 = dxi2[mask2]
	vx_n2 = vx_n2[mask2]
	vy_n2 = vy_n2[mask2]
	vx_nQ2 = vx_nQ2[mask2]
	vy_nQ2 = vy_nQ2[mask2]
	
	kx1 = np.concatenate((kx_E1,kxQ_E1))
	ky1 = np.concatenate((ky_E1,kyQ_E1))
	dkl1 = np.concatenate((dl_E1,dlQ_E1))
	dxi1 = np.concatenate((dxi_E1,dxiQ_E1))

	vx_n1 = np.concatenate((vx_E1,vx_E1Q))
	vy_n1 = np.concatenate((vy_E1,vy_E1Q))
	vx_nQ1 = np.concatenate((vxQ_E1,vxQ_E1Q))
	vy_nQ1 = np.concatenate((vyQ_E1,vyQ_E1Q))
	
	mask1 = ky1 != -100
	
	kx1 = kx1[mask1]
	ky1 = ky1[mask1]
	dkl1 = dkl1[mask1]
	dxi1 = dxi1[mask1]
	vx_n1 = vx_n1[mask1]
	vy_n1 = vy_n1[mask1]
	vx_nQ1 = vx_nQ1[mask1]
	vy_nQ1 = vy_nQ1[mask1]
	
	return kx2.flatten(), ky2.flatten(), dkl2.flatten(), dxi2.flatten(), vx_n2.flatten(), vy_n2.flatten(), vx_nQ2.flatten(), vy_nQ2.flatten(), kx1.flatten(), ky1.flatten(), dkl1.flatten(), dxi1.flatten(), vx_n1.flatten(), vy_n1.flatten(), vx_nQ1.flatten(), vy_nQ1.flatten()

def SDWmixingGridStitched(sc_order,disp,deltaFunc,t1,t2,M,Q,delta,E_c,N_phi,N_E):
	kx_E2_right, ky_E2_right, dkl_E2_right, dxi_E2_right, vx_n2_right, vy_n2_right, vx_nQ2_right, vy_nQ2_right, kx_E1_right, ky_E1_right, dkl_E1_right, dxi_E1_right, vx_n1_right, vy_n1_right, vx_nQ1_right, vy_nQ1_right =  SDW_grid_new(t1,t2,M,Q,E_c,N_E,N_phi)
	
	#kx_E2_mixing, ky_E2_mixing, dkl_E2_mixing, dxi_E2_mixing, vx_n2_mixing, vy_n2_mixing, vx_nQ2_mixing, vy_nQ2_mixing, f1, kxNode1, kyNode1, f2, kxNode2, kyNode2 = mixingNodeGrid(sc_order,disp,deltaFunc,t1,t2,M,Q,delta,.1*E_c/(15.*.75),N_phi,21)
	kx_E2_mixing, ky_E2_mixing, dkl_E2_mixing, dxi_E2_mixing, vx_n2_mixing, vy_n2_mixing, vx_nQ2_mixing, vy_nQ2_mixing, f1, kxNode1, kyNode1, f2, kxNode2, kyNode2 = mixingNodeGrid(sc_order,disp,deltaFunc,t1,t2,M,Q,delta,.4*E_c/(15.*.75),int(N_phi/4),1+int(N_E/4))
	
	kx1_diff = kx_E2_right - kxNode1
	ky1_diff = ky_E2_right - kyNode1
	kr1_diff = np.sqrt(kx1_diff**2 + ky1_diff**2)
	phi1_diff = np.arctan2(ky1_diff,kx1_diff) % (2*np.pi)
	kr1_cutoff = f1(phi1_diff)

	kx2_diff = kx_E2_right - kxNode2
	ky2_diff = ky_E2_right - kyNode2
	kr2_diff = np.sqrt(kx2_diff**2 + ky2_diff**2)
	phi2_diff = np.arctan2(ky2_diff,kx2_diff) % (2*np.pi)
	kr2_cutoff = f2(phi2_diff)
	
	mask = (kr1_diff > kr1_cutoff) & (kr2_diff > kr2_cutoff)
	
	kx_E2 = np.concatenate((kx_E2_right[mask],kx_E2_mixing))
	ky_E2 = np.concatenate((ky_E2_right[mask],ky_E2_mixing))
	dkl_E2 = np.concatenate((dkl_E2_right[mask],dkl_E2_mixing))
	dxi_E2 = np.concatenate((dxi_E2_right[mask],dxi_E2_mixing))
	vx_n2 = np.concatenate((vx_n2_right[mask],vx_n2_mixing))
	vy_n2 = np.concatenate((vy_n2_right[mask],vy_n2_mixing))
	vx_nQ2 = np.concatenate((vx_nQ2_right[mask],vx_nQ2_mixing))
	vy_nQ2 = np.concatenate((vy_nQ2_right[mask],vy_nQ2_mixing))
	"""
	plt.scatter(kx_E2_right[mask],ky_E2_right[mask],c = 'blue')
	plt.scatter(kx_E2_mixing,ky_E2_mixing,c = 'red')
	plt.show()
	"""
	return kx_E2,ky_E2,dkl_E2,dxi_E2,vx_n2,vy_n2,vx_nQ2,vy_nQ2,kx_E1_right,ky_E1_right,dkl_E1_right, dxi_E1_right,vx_n1_right,vy_n1_right,vx_nQ1_right,vy_nQ1_right
	
def FS_grid(dispersion,sc_order,deltaFunc,delta,M_order,M,Q,T,N_xi,N_phi,t1,t2,E_c,whichGrid):
	err = 1e-8
	
	phi = np.linspace(np.pi/2.-err,err,N_phi)
	#x = np.linspace(np.pi-err,0+err,N_phi)
	#phi = .25*np.pi*(np.heaviside(np.pi/2.-x,0)*np.sin(x) + np.heaviside(x-np.pi/2.,1)*(2-np.sin(x)))
	dphi_array = -np.gradient(phi)
	xi_array = np.linspace(-E_c,E_c,N_xi)
	
	dxi = xi_array[1] - xi_array[0]
	
	phi_grid,xi_grid = np.meshgrid(phi,xi_array)
	dphi,tmp = np.meshgrid(dphi_array,xi_array)
	
	kF = np.zeros(xi_grid.shape)
	kFQ = np.zeros(xi_grid.shape)
	
	kx = np.zeros(xi_grid.shape)
	kxQ = np.zeros(xi_grid.shape)
	ky = np.zeros(xi_grid.shape)
	kyQ = np.zeros(xi_grid.shape)
	dkx = np.zeros(xi_grid.shape)
	dkxQ = np.zeros(xi_grid.shape)
	dky = np.zeros(xi_grid.shape)
	dkyQ = np.zeros(xi_grid.shape)

	for i in range(0,N_xi):
		for j in range(0,N_phi):
			kF[i][j] = fsolve(lambda k: (dispersion(0.,k*np.cos(phi_grid[i][j]),k*np.sin(phi_grid[i][j]),1.,t1,t2) - xi_grid[i][j])**2,1.)
			kFQ[i][j] = fsolve(lambda k: (dispersion(0.,np.pi/2. - k*np.cos(phi_grid[i][j])+Q[0],np.pi/2. - k*np.sin(phi_grid[i][j])+Q[1],1.,t1,t2) - xi_grid[i][j])**2,1.)
		
		kx[i] = kF[i]*np.cos(phi_grid[i])
		ky[i] = kF[i]*np.sin(phi_grid[i])
		kxQ[i] = np.pi/2. - kFQ[i]*np.cos(phi_grid[i])
		kyQ[i] = np.pi/2. - kFQ[i]*np.sin(phi_grid[i])
		
		dkx[i] = np.gradient(kx[i])
		dkxQ[i] = np.gradient(kxQ[i])
		dky[i] = np.gradient(ky[i])
		dkyQ[i] = np.gradient(kyQ[i])
	
	xi_k1 = dispersion(0,kx,ky,1.,t1,t2)
	xi_kQ1 = dispersion(0,kx+Q[0],ky+Q[1],1.,t1,t2)
	xi_k2 = dispersion(0,kxQ,kyQ,1.,t1,t2)
	xi_kQ2 = dispersion(0,kxQ+Q[0],kyQ+Q[1],1.,t1,t2)
	
	delta_k1 = deltaFunc(kx,ky,delta)
	delta_kQ1 = deltaFunc(kx+2*Q[0],ky+2*Q[1],delta)
	delta_k2 = deltaFunc(kxQ,kyQ,delta)
	delta_kQ2 = deltaFunc(kxQ+2*Q[0],kyQ+2*Q[1],delta)
    
	M_k1 = M_order(kx,ky,M)
	M_k2 = M_order(kxQ,kyQ,M)

	E1_k = np.zeros(kx.shape)
	E2_k = np.zeros(kx.shape)
	E1_kQ = np.zeros(kxQ.shape)
	E2_kQ = np.zeros(kxQ.shape)

	dkl = np.abs(kF)*dphi/np.cos(dphi)
	dklQ = dkl
	
	vel_x_FS1,vel_y_FS1 = velocityTB(kx[int(N_xi/2)],ky[int(N_xi/2)],1.,t1,t2)
	vel_x_FSQ1,vel_y_FSQ1 = velocityTB(kx[int(N_xi/2)]+Q[0],ky[int(N_xi/2)]+Q[1],1.,t1,t2)
	vel_x_FS2,vel_y_FS2 = velocityTB(kxQ[int(N_xi/2)],kyQ[int(N_xi/2)],1.,t1,t2)
	vel_x_FSQ2,vel_y_FSQ2 = velocityTB(kxQ[int(N_xi/2)]+Q[0],kyQ[int(N_xi/2)]+Q[1],1.,t1,t2)
	
	vel_x_n1 = []
	vel_y_n1 = []
	vel_x_nQ1 = []
	vel_y_nQ1 = []
	
	vel_x_n2 = []
	vel_y_n2 = []
	vel_x_nQ2 = []
	vel_y_nQ2 = []
		
	for i in range(0,N_xi):
		vel_x_n1.append(vel_x_FS1)
		vel_y_n1.append(vel_y_FS1)
		vel_x_nQ1.append(vel_x_FSQ1)
		vel_y_nQ1.append(vel_y_FSQ1)
	
		vel_x_n2.append(vel_x_FS2)
		vel_y_n2.append(vel_y_FS2)
		vel_x_nQ2.append(vel_x_FSQ2)
		vel_y_nQ2.append(vel_y_FSQ2)
		
	vel_x_n1 = np.array(vel_x_n1)
	vel_y_n1 = np.array(vel_y_n1)
	vel_x_nQ1 = np.array(vel_x_nQ1)
	vel_y_nQ1 = np.array(vel_y_nQ1)

	vel_x_n2 = np.array(vel_x_n2)
	vel_y_n2 = np.array(vel_y_n2)
	vel_x_nQ2 = np.array(vel_x_nQ2)
	vel_y_nQ2 = np.array(vel_y_nQ2)
	
	vel_n1 = np.sqrt(vel_x_n1**2. + vel_y_n1**2.)
	vel_nQ1 = np.sqrt(vel_x_nQ1**2. + vel_y_nQ1**2.)
	vel_n2 = np.sqrt(vel_x_n2**2. + vel_y_n2**2.)
	vel_nQ2 = np.sqrt(vel_x_nQ2**2. + vel_y_nQ2**2.)
	
	if whichGrid == 'normal1':
		kx_E1 = kx
		ky_E1 = ky
		dkl_E1 = dkl
		vx_n1 = vel_x_n1
		vy_n1 = vel_y_n1
		vx_nQ1 = vel_x_nQ1
		vy_nQ1 = vel_y_nQ1
		v1 = vel_n1
		dxi_E1 = dxi*np.ones(kx_E1.shape)
		
		kx_E2 = kxQ
		ky_E2 = kyQ
		dkl_E2 = dklQ
		vx_n2 = vel_x_n2
		vy_n2 = vel_y_n2
		vx_nQ2 = vel_x_nQ2
		vy_nQ2 = vel_y_nQ2
		v2 = vel_nQ2
		dxi_E2 = dxi*np.ones(kx_E2.shape)
		
		kx_E1 = kx_E1.flatten()
		ky_E1 = ky_E1.flatten()
		dkl_E1 = dkl_E1.flatten()
		vx_n1 = vx_n1.flatten()
		vy_n1 = vy_n1.flatten()
		vx_nQ1 = vx_nQ1.flatten()
		vy_nQ1 = vy_nQ1.flatten()
		v1 = v1.flatten()
		dxi_E1 = dxi_E1.flatten()
		
		kx_E2 = kx_E2.flatten()
		ky_E2 = ky_E2.flatten()
		dkl_E2 = dkl_E2.flatten()
		vx_n2 = vx_n2.flatten()
		vy_n2 = vy_n2.flatten()
		vx_nQ2 = vx_nQ2.flatten()
		vy_nQ2 = vy_nQ2.flatten()
		v2 = v2.flatten()
		dxi_E2 = dxi_E2.flatten()
		
	elif whichGrid == 'normal2':
		kx_E1 = np.concatenate((kx,-kx))
		ky_E1 = np.concatenate((ky,ky))
		dkl_E1 = np.concatenate((dkl,dkl))
		vx_n1 = np.concatenate((vel_x_n1,-vel_x_n1))
		vy_n1 = np.concatenate((vel_y_n1,vel_y_n1))
		vx_nQ1 = np.concatenate((vel_x_nQ1,-vel_x_nQ1))
		vy_nQ1 = np.concatenate((vel_y_nQ1,vel_y_nQ1))
		v1 = np.concatenate((vel_n1,vel_n1))
		dxi_E1 = dxi*np.ones(kx_E1.shape)

		kx_E2 = np.concatenate((kxQ,-kxQ))
		ky_E2 = np.concatenate((kyQ,kyQ))
		dkl_E2 = np.concatenate((dklQ,dklQ))
		vx_n2 = np.concatenate((vel_x_n2,-vel_x_n2))
		vy_n2 = np.concatenate((vel_y_n2,vel_y_n2))
		vx_nQ2 = np.concatenate((vel_x_nQ2,-vel_x_nQ2))
		vy_nQ2 = np.concatenate((vel_y_nQ2,vel_y_nQ2))
		v2 = np.concatenate((vel_nQ2,vel_nQ2))
		dxi_E2 = dxi*np.ones(kx_E2.shape)
		
		kx_E1 = kx_E1.flatten()
		ky_E1 = ky_E1.flatten()
		dkl_E1 = dkl_E1.flatten()
		vx_n1 = vx_n1.flatten()
		vy_n1 = vy_n1.flatten()
		vx_nQ1 = vx_nQ1.flatten()
		vy_nQ1 = vy_nQ1.flatten()
		v1 = v1.flatten()
		dxi_E1 = dxi_E1.flatten()
		
		kx_E2 = kx_E2.flatten()
		ky_E2 = ky_E2.flatten()
		dkl_E2 = dkl_E2.flatten()
		vx_n2 = vx_n2.flatten()
		vy_n2 = vy_n2.flatten()
		vx_nQ2 = vx_nQ2.flatten()
		vy_nQ2 = vy_nQ2.flatten()
		v2 = v2.flatten()
		dxi_E2 = dxi_E2.flatten()
		
	elif whichGrid == 'SDW1':
		if T <= .75:
			
			kx_E2_right, ky_E2_right, dkl_E2_right, dxi_E2_right, vx_n2_right, vy_n2_right, vx_nQ2_right, vy_nQ2_right, kx_E1_right, ky_E1_right, dkl_E1_right, dxi_E1_right, vx_n1_right, vy_n1_right, vx_nQ1_right, vy_nQ1_right =  SDW_grid_new(t1,t2,M,Q,E_c,N_xi,N_phi)
			
			v1x_right = np.zeros(kx_E1_right.shape)
			v1y_right = np.zeros(kx_E1_right.shape)
			v1x_right[ky_E1_right > np.pi/2. - kx_E1_right] = vx_n1_right[ky_E1_right > np.pi/2. - kx_E1_right]
			v1x_right[ky_E1_right < np.pi/2. - kx_E1_right] = vx_nQ1_right[ky_E1_right < np.pi/2. - kx_E1_right]
			v1y_right[ky_E1_right > np.pi/2. - kx_E1_right] = vy_n1_right[ky_E1_right > np.pi/2. - kx_E1_right]
			v1y_right[ky_E1_right < np.pi/2. - kx_E1_right] = vy_nQ1_right[ky_E1_right < np.pi/2. - kx_E1_right]
			v1_right = np.sqrt(v1x_right**2. + v1y_right**2.)
			
			kx_E1 = kx_E1_right.flatten()
			ky_E1 = ky_E1_right.flatten()
			dkl_E1 = dkl_E1_right.flatten()
			vx_n1 = vx_n1_right.flatten()
			vy_n1 = vy_n1_right.flatten()
			vx_nQ1 = vx_nQ1_right.flatten()
			vy_nQ1 = vy_nQ1_right.flatten()
			v1x = v1x_right.flatten()
			v1y = v1y_right.flatten()
			v1 = v1_right.flatten()
			dxi_E1 = dxi_E1_right.flatten()
			
			v2x_right = np.zeros(kx_E2_right.shape)
			v2y_right = np.zeros(kx_E2_right.shape)
			v2x_right[ky_E2_right < np.pi/2. - kx_E2_right] = vx_n2_right[ky_E2_right < np.pi/2. - kx_E2_right]
			v2x_right[ky_E2_right > np.pi/2. - kx_E2_right] = vx_nQ2_right[ky_E2_right > np.pi/2. - kx_E2_right]
			v2y_right[ky_E2_right < np.pi/2. - kx_E2_right] = vy_n2_right[ky_E2_right < np.pi/2. - kx_E2_right]
			v2y_right[ky_E2_right > np.pi/2. - kx_E2_right] = vy_nQ2_right[ky_E2_right > np.pi/2. - kx_E2_right]
			v2_right = np.sqrt(v2x_right**2. + v2y_right**2.)
			
			kx_E2 = kx_E2_right.flatten()
			ky_E2 = ky_E2_right.flatten()
			dkl_E2 = dkl_E2_right.flatten()
			vx_n2 = vx_n2_right.flatten()
			vy_n2 = vy_n2_right.flatten()
			vx_nQ2 = vx_nQ2_right.flatten()
			vy_nQ2 = vy_nQ2_right.flatten()
			v2x = v2x_right.flatten()
			v2y = v2y_right.flatten()
			v2 = v2_right.flatten()
			dxi_E2 = dxi_E2_right.flatten()
		else:
			
			kx1_new = np.zeros(kx.shape)
			ky1_new = np.zeros(ky.shape)
			kxQ1_new = np.zeros(kxQ.shape)
			kyQ1_new = np.zeros(kyQ.shape)
			
			kx2_new = np.zeros(kx.shape)
			ky2_new = np.zeros(ky.shape)
			kxQ2_new = np.zeros(kxQ.shape)
			kyQ2_new = np.zeros(kyQ.shape)
			
			for i in range(0,N_xi):
				mask = kyQ[i] > np.flip(ky[i])
				
				kx1_new[i] = kx[i]
				ky1_new[i][~mask] = ky[i][~mask]
				
				kxQ1_new[i] = kxQ[i]
				kyQ1_new[i][~mask] = kyQ[i][~mask]
				
				ky1_new[i][mask] = 100
				kyQ1_new[i][mask] = 100
				
				kx2_new[i] = kx[i]
				ky2_new[i][mask] = ky[i][mask]
				
				kxQ2_new[i] = kxQ[i]
				kyQ2_new[i][mask] = kyQ[i][mask]
				
				ky2_new[i][~mask] = 100
				kyQ2_new[i][~mask] = 100
				
			kx_E1 = np.concatenate((kx1_new,kxQ1_new)).flatten()
			ky_E1 = np.concatenate((ky1_new,kyQ1_new)).flatten()
			dkl_E1 = np.concatenate((dkl,dklQ)).flatten()
			vx_n1 = np.concatenate((vel_x_n1,vel_x_n2)).flatten()
			vy_n1 = np.concatenate((vel_y_n1,vel_y_n2)).flatten()
			vx_nQ1 = np.concatenate((vel_x_nQ1,vel_x_nQ2)).flatten()
			vy_nQ1 = np.concatenate((vel_y_nQ1,vel_y_nQ2)).flatten()
			v1x = np.concatenate((vel_x_n1,vel_x_nQ2)).flatten()
			v1y = np.concatenate((vel_y_n1,vel_y_nQ2)).flatten()
			v1 = np.concatenate((vel_n1,vel_nQ2)).flatten()
			dxi_E1 = dxi*np.ones(kx_E1.shape).flatten()
			
			kx_E2 = np.concatenate((kx2_new,kxQ2_new)).flatten()
			ky_E2 = np.concatenate((ky2_new,kyQ2_new)).flatten()
			dkl_E2 = np.concatenate((dkl,dklQ)).flatten()
			vx_n2 = np.concatenate((vel_x_n1,vel_x_n2)).flatten()
			vy_n2 = np.concatenate((vel_y_n1,vel_y_n2)).flatten()
			vx_nQ2 = np.concatenate((vel_x_nQ1,vel_x_nQ2)).flatten()
			vy_nQ2 = np.concatenate((vel_y_nQ1,vel_y_nQ2)).flatten()
			v2x = np.concatenate((vel_x_n1,vel_x_nQ2)).flatten()
			v2y = np.concatenate((vel_y_n1,vel_y_nQ2)).flatten()
			v2 = np.concatenate((vel_n1,vel_nQ2)).flatten()
			dxi_E2 = dxi*np.ones(kx_E2.shape).flatten()

	elif whichGrid == 'SDW2':
		kx1_new = np.zeros(kx.shape)
		ky1_new = np.zeros(ky.shape)
		kxQ1_new = np.zeros(kxQ.shape)
		kyQ1_new = np.zeros(kyQ.shape)

		kx2_new = np.zeros(kx.shape)
		ky2_new = np.zeros(ky.shape)
		kxQ2_new = np.zeros(kxQ.shape)
		kyQ2_new = np.zeros(kyQ.shape)
		
		for i in range(0,N_xi):
			mask = kyQ[i] > np.flip(ky[i])

			kx1_new[i] = kx[i]
			ky1_new[i][~mask] = ky[i][~mask]

			kxQ1_new[i] = kxQ[i]
			kyQ1_new[i][~mask] = kyQ[i][~mask]

			ky1_new[i][mask] = 100
			kyQ1_new[i][mask] = 100
	
			kx2_new[i] = kx[i]
			ky2_new[i][mask] = ky[i][mask]

			kxQ2_new[i] = kxQ[i]
			kyQ2_new[i][mask] = kyQ[i][mask]
			
			ky2_new[i][~mask] = 100
			kyQ2_new[i][~mask] = 100

		kx_E1 = np.concatenate((kx1_new,kxQ1_new,-kx1_new,-kxQ1_new))
		ky_E1 = np.concatenate((ky1_new,kyQ1_new,ky1_new,kyQ1_new))
		dkl_E1 = np.concatenate((dkl,dklQ,dkl,dklQ))
		vx_n1 = np.concatenate((vel_x_n1,vel_x_n2,-vel_x_n1,-vel_x_n2))
		vy_n1 = np.concatenate((vel_y_n1,vel_y_n2,vel_y_n1,vel_y_n2))
		vx_nQ1 = np.concatenate((vel_x_nQ1,vel_x_nQ2,vel_x_nQ1,vel_x_nQ2))
		vy_nQ1 = np.concatenate((vel_y_nQ1,vel_y_nQ2,vel_y_nQ1,vel_y_nQ2))
		v1x = np.concatenate((vel_x_n1,vel_x_nQ2,vel_x_n1,vel_x_nQ2))
		v1y = np.concatenate((vel_y_n1,vel_y_nQ2,vel_y_n1,vel_y_nQ2))
		v1 = np.concatenate((vel_n1,vel_nQ2,vel_n1,vel_nQ2))

		kx_E2 = np.concatenate((kx2_new,kxQ2_new,-kx2_new,-kxQ2_new))
		ky_E2 = np.concatenate((ky2_new,kyQ2_new,ky2_new,kyQ2_new))
		dkl_E2 = np.concatenate((dkl,dklQ,dkl,dklQ))
		vx_n2 = np.concatenate((vel_x_n1,vel_x_n2,-vel_x_n1,-vel_x_n2))
		vy_n2 = np.concatenate((vel_y_n1,vel_y_n2,vel_y_n1,vel_y_n2))
		vx_nQ2 = np.concatenate((vel_x_nQ1,vel_x_nQ2,-vel_x_nQ1,-vel_x_nQ2))
		vy_nQ2 = np.concatenate((vel_y_nQ1,vel_y_nQ2,vel_y_nQ1,vel_y_nQ2))
		v2x = np.concatenate((vel_x_n1,vel_x_nQ2,vel_x_n1,vel_x_nQ2))
		v2y = np.concatenate((vel_y_n1,vel_y_nQ2,vel_y_n1,vel_y_nQ2))
		v2 = np.concatenate((vel_n1,vel_nQ2,vel_n1,vel_nQ2))
	
	elif whichGrid == 'SDW2_aniso':
		if T <= .75:
			if delta > 0. and (sc_order == 'swave' or sc_order == 'dxy'):
				kx_E2_right, ky_E2_right, dkl_E2_right, dxi_E2_right, vx_n2_right, vy_n2_right, vx_nQ2_right, vy_nQ2_right, kx_E1_right, ky_E1_right, dkl_E1_right, dxi_E1_right, vx_n1_right, vy_n1_right, vx_nQ1_right, vy_nQ1_right = SDWmixingGridStitched(sc_order,dispersion,deltaFunc,t1,t2,M,Q,delta,E_c,N_phi,N_xi)
			else:
				kx_E2_right, ky_E2_right, dkl_E2_right, dxi_E2_right, vx_n2_right, vy_n2_right, vx_nQ2_right, vy_nQ2_right, kx_E1_right, ky_E1_right, dkl_E1_right, dxi_E1_right, vx_n1_right, vy_n1_right, vx_nQ1_right, vy_nQ1_right =  SDW_grid_new(t1,t2,M,Q,E_c,N_xi,N_phi)
			v1x_right = np.zeros(kx_E1_right.shape)
			v1y_right = np.zeros(kx_E1_right.shape)
			v1x_right[ky_E1_right > np.pi/2. - kx_E1_right] = vx_n1_right[ky_E1_right > np.pi/2. - kx_E1_right]
			v1x_right[ky_E1_right < np.pi/2. - kx_E1_right] = vx_nQ1_right[ky_E1_right < np.pi/2. - kx_E1_right]
			v1y_right[ky_E1_right > np.pi/2. - kx_E1_right] = vy_n1_right[ky_E1_right > np.pi/2. - kx_E1_right]
			v1y_right[ky_E1_right < np.pi/2. - kx_E1_right] = vy_nQ1_right[ky_E1_right < np.pi/2. - kx_E1_right]
			v1_right = np.sqrt(v1x_right**2. + v1y_right**2.)
			
			kx_E1 = np.concatenate((kx_E1_right,-kx.flatten())).flatten()
			ky_E1 = np.concatenate((ky_E1_right,ky.flatten())).flatten()
			dkl_E1 = np.concatenate((dkl_E1_right,dkl.flatten())).flatten()
			vx_n1 = np.concatenate((vx_n1_right,-vel_x_n1.flatten())).flatten()
			vy_n1 = np.concatenate((vy_n1_right,vel_y_n1.flatten())).flatten()
			vx_nQ1 = np.concatenate((vx_nQ1_right,-vel_x_nQ1.flatten())).flatten()
			vy_nQ1 = np.concatenate((vy_nQ1_right,vel_y_nQ1.flatten())).flatten()
			v1x = np.concatenate((v1x_right,vel_x_n1.flatten())).flatten()
			v1y = np.concatenate((v1y_right,vel_y_n1.flatten())).flatten()
			v1 = np.concatenate((v1_right,vel_n1.flatten())).flatten()
			dxi_E1 = np.concatenate((dxi_E1_right,dxi*np.ones(kx.shape).flatten())).flatten()
			
			v2x_right = np.zeros(kx_E2_right.shape)
			v2y_right = np.zeros(kx_E2_right.shape)
			v2x_right[ky_E2_right < np.pi/2. - kx_E2_right] = vx_n2_right[ky_E2_right < np.pi/2. - kx_E2_right]
			v2x_right[ky_E2_right > np.pi/2. - kx_E2_right] = vx_nQ2_right[ky_E2_right > np.pi/2. - kx_E2_right]
			v2y_right[ky_E2_right < np.pi/2. - kx_E2_right] = vy_n2_right[ky_E2_right < np.pi/2. - kx_E2_right]
			v2y_right[ky_E2_right > np.pi/2. - kx_E2_right] = vy_nQ2_right[ky_E2_right > np.pi/2. - kx_E2_right]
			v2_right = np.sqrt(v2x_right**2. + v2y_right**2.)
			
			kx_E2 = np.concatenate((kx_E2_right,-kxQ.flatten())).flatten()
			ky_E2 = np.concatenate((ky_E2_right,kyQ.flatten())).flatten()
			dkl_E2 = np.concatenate((dkl_E2_right,dklQ.flatten())).flatten()
			vx_n2 = np.concatenate((vx_n2_right,-vel_x_n2.flatten())).flatten()
			vy_n2 = np.concatenate((vy_n2_right,vel_y_n2.flatten())).flatten()
			vx_nQ2 = np.concatenate((vx_nQ2_right,-vel_x_nQ2.flatten())).flatten()
			vy_nQ2 = np.concatenate((vy_nQ2_right,vel_y_nQ2.flatten())).flatten()
			v2x = np.concatenate((v2x_right,vel_x_nQ2.flatten())).flatten()
			v2y = np.concatenate((v2y_right,vel_y_nQ2.flatten())).flatten()
			v2 = np.concatenate((v2_right,vel_nQ2.flatten())).flatten()
			dxi_E2 = np.concatenate((dxi_E2_right,dxi*np.ones(kxQ.shape).flatten())).flatten()
		else:
			
			kx1_new = np.zeros(kx.shape)
			ky1_new = np.zeros(ky.shape)
			kxQ1_new = np.zeros(kxQ.shape)
			kyQ1_new = np.zeros(kyQ.shape)
			
			kx2_new = np.zeros(kx.shape)
			ky2_new = np.zeros(ky.shape)
			kxQ2_new = np.zeros(kxQ.shape)
			kyQ2_new = np.zeros(kyQ.shape)
			
			for i in range(0,N_xi):
				mask = kyQ[i] > np.flip(ky[i])
				
				kx1_new[i] = kx[i]
				ky1_new[i][~mask] = ky[i][~mask]
				
				kxQ1_new[i] = kxQ[i]
				kyQ1_new[i][~mask] = kyQ[i][~mask]
				
				ky1_new[i][mask] = 100
				kyQ1_new[i][mask] = 100
				
				kx2_new[i] = kx[i]
				ky2_new[i][mask] = ky[i][mask]
				
				kxQ2_new[i] = kxQ[i]
				kyQ2_new[i][mask] = kyQ[i][mask]
				
				ky2_new[i][~mask] = 100
				kyQ2_new[i][~mask] = 100
				
			kx_E1 = np.concatenate((kx1_new,kxQ1_new,-kx)).flatten()
			ky_E1 = np.concatenate((ky1_new,kyQ1_new,ky)).flatten()
			dkl_E1 = np.concatenate((dkl,dklQ,dkl)).flatten()
			vx_n1 = np.concatenate((vel_x_n1,vel_x_n2,-vel_x_n1)).flatten()
			vy_n1 = np.concatenate((vel_y_n1,vel_y_n2,vel_y_n1)).flatten()
			vx_nQ1 = np.concatenate((vel_x_nQ1,vel_x_nQ2,-vel_x_nQ1)).flatten()
			vy_nQ1 = np.concatenate((vel_y_nQ1,vel_y_nQ2,vel_y_nQ1)).flatten()
			v1x = np.concatenate((vel_x_n1,vel_x_nQ2,vel_x_n1)).flatten()
			v1y = np.concatenate((vel_y_n1,vel_y_nQ2,vel_y_n1)).flatten()
			v1 = np.concatenate((vel_n1,vel_nQ2,vel_n1)).flatten()
			dxi_E1 = dxi*np.ones(kx_E1.shape).flatten()
			
			kx_E2 = np.concatenate((kx2_new,kxQ2_new,-kxQ)).flatten()
			ky_E2 = np.concatenate((ky2_new,kyQ2_new,kyQ)).flatten()
			dkl_E2 = np.concatenate((dkl,dklQ,dklQ)).flatten()
			vx_n2 = np.concatenate((vel_x_n1,vel_x_n2,-vel_x_n2)).flatten()
			vy_n2 = np.concatenate((vel_y_n1,vel_y_n2,vel_y_n2)).flatten()
			vx_nQ2 = np.concatenate((vel_x_nQ1,vel_x_nQ2,-vel_x_nQ2)).flatten()
			vy_nQ2 = np.concatenate((vel_y_nQ1,vel_y_nQ2,vel_y_nQ2)).flatten()
			v2x = np.concatenate((vel_x_n1,vel_x_nQ2,vel_x_nQ2)).flatten()
			v2y = np.concatenate((vel_y_n1,vel_y_nQ2,vel_y_nQ2)).flatten()
			v2 = np.concatenate((vel_n1,vel_nQ2,vel_nQ2)).flatten()
			dxi_E2 = dxi*np.ones(kx_E2.shape).flatten()
	
	xi_k_E1 = dispersion(0,kx_E1,ky_E1,1.,t1,t2)
	xi_kQ_E1 = dispersion(0,kx_E1+Q[0],ky_E1+Q[1],1.,t1,t2)
	err = 1e-8
	
	xi_k_E1[np.abs(xi_k_E1) < err] = np.zeros(xi_k_E1[np.abs(xi_k_E1) < err].shape)
	xi_kQ_E1[np.abs(xi_kQ_E1) < err] = np.zeros(xi_kQ_E1[np.abs(xi_kQ_E1) < err].shape)
	
	M_k_E1 = M_order(kx_E1,ky_E1,M)
	M_k_E2 = M_order(kx_E2,ky_E2,M)
	
	xi_k_E2 = dispersionTB(0,kx_E2,ky_E2,1,t1,t2)
	xi_kQ_E2 = dispersionTB(0,kx_E2+Q[0],ky_E2+Q[1],1,t1,t2)
		
	delta_k_E1 = deltaFunc(kx_E1,ky_E1,delta)
	if np.abs(delta) > 0:
		delta_k_E1[np.abs(delta_k_E1) < 1e-6] = 1e-6*np.ones(delta_k_E1[np.abs(delta_k_E1) < 1e-6].shape)
	if sc_order == 'dxy':
		#delta_kQ_E1 = delta*deltaFuncQ_dxy(kx_E1,ky_E1,Q)
		delta_kQ_E1 = delta_k_E1
	elif sc_order == 'dx2y2':
		delta_kQ_E1 = -delta_k_E1
	else:
		delta_kQ_E1 = deltaFunc(kx_E1+2*Q[0],ky_E1+2*Q[1],delta)
	
	if delta > 0:
		delta_k_E1[np.abs(delta_k_E1) < 1e-8] = np.sign(delta_k_E1[np.abs(delta_k_E1) < 1e-8])*1e-8
		delta_kQ_E1[np.abs(delta_kQ_E1) < 1e-8] = np.sign(delta_kQ_E1[np.abs(delta_kQ_E1) < 1e-8])*1e-8
	
	delta_k_E2 = deltaFunc(kx_E2,ky_E2,delta)
	if np.abs(delta):
		delta_k_E2[np.abs(delta_k_E2) < 1e-6] = 1e-6*np.ones(delta_k_E2[np.abs(delta_k_E2) < 1e-6].shape)
	if sc_order == 'dxy':
		#delta_kQ_E2 = delta*deltaFuncQ_dxy(kx_E2,ky_E2,Q)
		delta_kQ_E2 = delta_k_E2
	elif sc_order == 'dx2y2':
		delta_kQ_E2 = -delta_k_E2
	else:
		delta_kQ_E2 = deltaFunc(kx_E2+2*Q[0],ky_E2+2*Q[1],delta)
	
	if delta > 0:
		delta_k_E2[np.abs(delta_k_E2) < 1e-8] = np.sign(delta_k_E2[np.abs(delta_k_E2) < 1e-8])*1e-8
		delta_kQ_E2[np.abs(delta_kQ_E2) < 1e-8] = np.sign(delta_kQ_E2[np.abs(delta_kQ_E2) < 1e-8])*1e-8
	
	E1 = np.zeros(kx_E1.shape)
	E2 = np.zeros(kx_E2.shape)

	for i in range(0,kx_E1.shape[0]):
		E1plus,E1minus,E2plus,E2minus = eigenvalues4(xi_k_E1[i], xi_kQ_E1[i], delta_k_E1[i], delta_kQ_E1[i], M_k_E1[i])
		E1[i] = E1plus

	for i in range(0,kx_E2.shape[0]):
		E1plus,E1minus,E2plus,E2minus = eigenvalues4(xi_k_E2[i], xi_kQ_E2[i], delta_k_E2[i], delta_kQ_E2[i], M_k_E2[i])
		E2[i] = E2plus
	
	vx1_E1 = np.zeros(vx_n1.shape)
	vy1_E1 = np.zeros(vy_n1.shape)
	vx2_E1 = np.zeros(vx_nQ1.shape)
	vy2_E1 = np.zeros(vy_nQ1.shape)

	vx1_E2 = np.zeros(vx_n2.shape)
	vy1_E2 = np.zeros(vy_n2.shape)
	vx2_E2 = np.zeros(vx_nQ2.shape)
	vy2_E2 = np.zeros(vy_nQ2.shape)

	if whichGrid == 'normal1' or whichGrid == 'normal2':
		err = 1e-8
		mask1_E1 = np.abs(xi_k_E1) > err
		mask2_E1 = np.abs(xi_kQ_E1) > err
		mask1_E2 = np.abs(xi_k_E2) > err
		mask2_E2 = np.abs(xi_kQ_E2) > err
		
		if np.abs(delta) > 0.:
			vx1_E1[mask1_E1] = (vx_n1*xi_k_E1/E1)[mask1_E1]
			vy1_E1[mask1_E1] = (vy_n1*xi_k_E1/E1)[mask1_E1]
			vx2_E1[mask2_E1] = (vx_nQ1*xi_kQ_E1/E2)[mask2_E1]
			vy2_E1[mask2_E1] = (vy_nQ1*xi_kQ_E1/E2)[mask2_E1]
			
			vx1_E1[~mask1_E1] = vx_n1[~mask1_E1]
			vy1_E1[~mask1_E1] = vy_n1[~mask1_E1]
			vx2_E1[~mask2_E1] = vx_nQ1[~mask2_E1]
			vy2_E1[~mask2_E1] = vy_nQ1[~mask2_E1]
			
			vx1_E2[mask1_E2] = (vx_n2*xi_k_E2/E1)[mask1_E2]
			vy1_E2[mask1_E2] = (vy_n2*xi_k_E2/E1)[mask1_E2]
			vx2_E2[mask2_E2] = (vx_nQ2*xi_kQ_E2/E2)[mask2_E2]
			vy2_E2[mask2_E2] = (vy_nQ2*xi_kQ_E2/E2)[mask2_E2]
			
			vx1_E2[~mask1_E2] = vx_n2[~mask1_E2]
			vy1_E2[~mask1_E2] = vy_n2[~mask1_E2]
			vx2_E2[~mask2_E2] = vx_nQ2[~mask2_E2]
			vy2_E2[~mask2_E2] = vy_nQ2[~mask2_E2]
		
		else:
			vx1_E1 = vx_n1
			vy1_E1 = vy_n1
			vx2_E1 = vx_nQ1
			vy2_E1 = vy_nQ1

			vx1_E2 = vx_n2
			vy1_E2 = vy_n2
			vx2_E2 = vx_nQ2
			vy2_E2 = vy_nQ2
	
	elif whichGrid == 'SDW1' or whichGrid == 'SDW2':
		if np.abs(M) > 0.:
			
			xi_k1_plus = .5*(xi_k_E1 + xi_kQ_E1)
			xi_k1_minus = .5*(xi_k_E1 - xi_kQ_E1)
			xi_k2_plus = .5*(xi_k_E2 + xi_kQ_E2)
			xi_k2_minus = .5*(xi_k_E2 - xi_kQ_E2)
			if delta == 0.:
				Lambda_k1 = np.sqrt((xi_k1_minus)**2. + M_k_E1**2.)
				Lambda_k2 = np.sqrt((xi_k2_minus)**2. + M_k_E2**2.)
	
				vx1_E1 = .5*(vx_n1+vx_nQ1) + xi_k1_minus*(.5*(vx_n1-vx_nQ1))/Lambda_k1
				vy1_E1 = .5*(vy_n1+vy_nQ1) + xi_k1_minus*(.5*(vy_n1-vy_nQ1))/Lambda_k1
				vx2_E2 = .5*(vx_n2+vx_nQ2) - xi_k2_minus*(.5*(vx_n2-vx_nQ2))/Lambda_k2
				vy2_E2 = .5*(vy_n2+vy_nQ2) - xi_k2_minus*(.5*(vy_n2-vy_nQ2))/Lambda_k2
			else:
				delta_k1_plus = .5*(delta_k_E1 + delta_kQ_E1)
				delta_k1_minus = .5*(delta_k_E1 - delta_kQ_E1)
				delta_k2_plus = .5*(delta_k_E2 + delta_kQ_E2)
				delta_k2_minus = .5*(delta_k_E2 - delta_kQ_E2)
				
				Lambda_k2 = np.sqrt((xi_k2_plus*xi_k2_minus + delta_k2_plus*delta_k2_minus)**2. + M_k_E2*M_k_E2*(xi_k2_plus**2. + delta_k2_plus**2.))
	
				Lambda_k1 = np.sqrt((xi_k1_plus*xi_k1_minus + delta_k1_plus*delta_k1_minus)**2. + M_k_E1*M_k_E1*(xi_k1_plus**2. + delta_k1_plus**2.))
				err = dxi/2.
				
				vx1_E1 = (xi_k_E1*vx_n1 + xi_kQ_E1*vx_nQ1)/(2.*E1) + ((xi_k1_plus*xi_k1_minus + delta_k1_plus*delta_k1_minus)*(xi_k_E1*vx_n1-xi_kQ_E1*vx_nQ1) + M_k_E1*M_k_E1*xi_k1_plus*(vx_n1 + vx_nQ1))/(2.*E1*Lambda_k1)
				vx1_E1[Lambda_k1 < 1e-2] = ((xi_k_E1*vx_n1 + xi_kQ_E1*vx_nQ1)/(2.*E1))[Lambda_k1 < 1e-2]
				vx1_E1[E1 < 1e-2] = v1x[E1 < 1e-2]
	
				vy1_E1 = (xi_k_E1*vy_n1 + xi_kQ_E1*vy_nQ1)/(2.*E1) + ((xi_k1_plus*xi_k1_minus + delta_k1_plus*delta_k1_minus)*(xi_k_E1*vy_n1-xi_kQ_E1*vy_nQ1) + M_k_E1*M_k_E1*xi_k1_plus*(vy_n1 + vy_nQ1))/(2.*E1*Lambda_k1)
				vy1_E1[Lambda_k1 < 1e-2] = ((xi_k_E1*vy_n1 + xi_kQ_E1*vy_nQ1)/(2.*E1))[Lambda_k1 < 1e-2]
				vy1_E1[E1 < 1e-2] = v1y[E1 < 1e-2]
	
				vx2_E2 = (xi_k_E2*vx_n2 + xi_kQ_E2*vx_nQ2)/(2.*E2) - ((xi_k2_plus*xi_k2_minus + delta_k2_plus*delta_k2_minus)*(xi_k_E2*vx_n2-xi_kQ_E2*vx_nQ2) + M_k_E2*M_k_E2*xi_k2_plus*(vx_n2 + vx_nQ2))/(2.*E2*Lambda_k2)
				vx2_E2[Lambda_k2 < 1e-2] = ((xi_k_E2*vx_n2 + xi_kQ_E2*vx_nQ2)/(2.*E2))[Lambda_k2 < 1e-2]
				vx2_E2[E2 < 1e-2] = v2x[E2 < 1e-2]

				vy2_E2 = (xi_k_E2*vy_n2 + xi_kQ_E2*vy_nQ2)/(2.*E2) - ((xi_k2_plus*xi_k2_minus + delta_k2_plus*delta_k2_minus)*(xi_k_E2*vy_n2-xi_kQ_E2*vy_nQ2) + M_k_E2*M_k_E2*xi_k2_plus*(vy_n2 + vy_nQ2))/(2.*E2*Lambda_k2)
				vy2_E2[Lambda_k2 < 1e-2] = ((xi_k_E2*vy_n2 + xi_kQ_E2*vy_nQ2)/(2.*E2))[Lambda_k2 < 1e-2]
				vy2_E2[E2 < 1e-2] = v2y[E2 < 1e-2]

		else:
			vx1_E1 = vx_n1
			vy1_E1 = vy_n1
			vx2_E1 = vx_nQ1
			vy2_E1 = vy_nQ1

			vx1_E2 = vx_n2
			vy1_E2 = vy_n2
			vx2_E2 = vx_nQ2
			vy2_E2 = vy_nQ2

	elif whichGrid == 'SDW2_aniso':
		if np.abs(M) > 0.:
			
			xi_k1_plus = .5*(xi_k_E1 + xi_kQ_E1)
			xi_k1_minus = .5*(xi_k_E1 - xi_kQ_E1)
			xi_k2_plus = .5*(xi_k_E2 + xi_kQ_E2)
			xi_k2_minus = .5*(xi_k_E2 - xi_kQ_E2)
			
			if delta == 0.:
				
				Lambda_k1 = np.sqrt((xi_k1_minus)**2. + M_k_E1**2.)
				Lambda_k2 = np.sqrt((xi_k2_minus)**2. + M_k_E2**2.)
				
				vx1_E1 = .5*(vx_n1+vx_nQ1) + xi_k1_minus*(.5*(vx_n1-vx_nQ1))/Lambda_k1
				vy1_E1 = .5*(vy_n1+vy_nQ1) + xi_k1_minus*(.5*(vy_n1-vy_nQ1))/Lambda_k1
				vx2_E2 = .5*(vx_n2+vx_nQ2) - xi_k2_minus*(.5*(vx_n2-vx_nQ2))/Lambda_k2
				vy2_E2 = .5*(vy_n2+vy_nQ2) - xi_k2_minus*(.5*(vy_n2-vy_nQ2))/Lambda_k2
			else:
				delta_k1_plus = .5*(delta_k_E1 + delta_kQ_E1)
				delta_k1_minus = .5*(delta_k_E1 - delta_kQ_E1)
				delta_k2_plus = .5*(delta_k_E2 + delta_kQ_E2)
				delta_k2_minus = .5*(delta_k_E2 - delta_kQ_E2)
				
				Lambda_k2 = np.sqrt((xi_k2_plus*xi_k2_minus + delta_k2_plus*delta_k2_minus)**2. + M_k_E2*M_k_E2*(xi_k2_plus**2. + delta_k2_plus**2.))
	
				Lambda_k1 = np.sqrt((xi_k1_plus*xi_k1_minus + delta_k1_plus*delta_k1_minus)**2. + M_k_E1*M_k_E1*(xi_k1_plus**2. + delta_k1_plus**2.))
				err = dxi/2.
				
				vx1_E1 = (xi_k_E1*vx_n1 + xi_kQ_E1*vx_nQ1)/(2.*E1) + ((xi_k1_plus*xi_k1_minus + delta_k1_plus*delta_k1_minus)*(xi_k_E1*vx_n1-xi_kQ_E1*vx_nQ1) + M_k_E1*M_k_E1*xi_k1_plus*(vx_n1 + vx_nQ1))/(2.*E1*Lambda_k1)
				vx1_E1[Lambda_k1 < 1e-2] = ((xi_k_E1*vx_n1 + xi_kQ_E1*vx_nQ1)/(2.*E1))[Lambda_k1 < 1e-2]
				vx1_E1[E1 < 1e-2] = v1x[E1 < 1e-2]

				vy1_E1 = (xi_k_E1*vy_n1 + xi_kQ_E1*vy_nQ1)/(2.*E1) + ((xi_k1_plus*xi_k1_minus + delta_k1_plus*delta_k1_minus)*(xi_k_E1*vy_n1-xi_kQ_E1*vy_nQ1) + M_k_E1*M_k_E1*xi_k1_plus*(vy_n1 + vy_nQ1))/(2.*E1*Lambda_k1)
				vy1_E1[Lambda_k1 < 1e-2] = ((xi_k_E1*vy_n1 + xi_kQ_E1*vy_nQ1)/(2.*E1))[Lambda_k1 < 1e-2]
				vy1_E1[E1 < 1e-2] = v1y[E1 < 1e-2]
	
				vx2_E2 = (xi_k_E2*vx_n2 + xi_kQ_E2*vx_nQ2)/(2.*E2) - ((xi_k2_plus*xi_k2_minus + delta_k2_plus*delta_k2_minus)*(xi_k_E2*vx_n2-xi_kQ_E2*vx_nQ2) + M_k_E2*M_k_E2*xi_k2_plus*(vx_n2 + vx_nQ2))/(2.*E2*Lambda_k2)
				vx2_E2[Lambda_k2 < 1e-2] = ((xi_k_E2*vx_n2 + xi_kQ_E2*vx_nQ2)/(2.*E2))[Lambda_k2 < 1e-2]
				vx2_E2[E2 < 1e-2] = v2x[E2 < 1e-2]

				vy2_E2 = (xi_k_E2*vy_n2 + xi_kQ_E2*vy_nQ2)/(2.*E2) - ((xi_k2_plus*xi_k2_minus + delta_k2_plus*delta_k2_minus)*(xi_k_E2*vy_n2-xi_kQ_E2*vy_nQ2) + M_k_E2*M_k_E2*xi_k2_plus*(vy_n2 + vy_nQ2))/(2.*E2*Lambda_k2)
				vy2_E2[Lambda_k2 < 1e-2] = ((xi_k_E2*vy_n2 + xi_kQ_E2*vy_nQ2)/(2.*E2))[Lambda_k2 < 1e-2]
				vy2_E2[E2 < 1e-2] = v2y[E2 < 1e-2]

			if np.abs(delta) > 0.:
				err = 1e-8
				mask1_E1 = (np.abs(xi_k_E1) < err)
				mask2_E1 = (np.abs(xi_kQ_E1) < err)
				mask1_E2 = (np.abs(xi_k_E2) < err)
				mask2_E2 = (np.abs(xi_kQ_E2) < err)
				
				vx1_E1[kx_E1 < 0.] = (vx_n1*xi_k_E1/E1)[kx_E1 < 0.]
				vy1_E1[kx_E1 < 0.] = (vy_n1*xi_k_E1/E1)[kx_E1 < 0.]
				vx1_E1[mask1_E1][kx_E1[mask1_E1] < 0.] = vx_n1[mask1_E1][kx_E1[mask1_E1] < 0.]
				vy1_E1[mask1_E1][kx_E1[mask1_E1] < 0.] = vy_n1[mask1_E1][kx_E1[mask1_E1] < 0.]
				vx2_E2[kx_E2 < 0.] = (vx_nQ2*xi_kQ_E2/E2)[kx_E2 < 0.]
				vy2_E2[kx_E2 < 0.] = (vy_nQ2*xi_kQ_E2/E2)[kx_E2 < 0.]
				vx2_E2[mask2_E2][kx_E2[mask2_E2] < 0.] = vx_nQ2[mask2_E2][kx_E2[mask2_E2] < 0.]
				vy2_E2[mask2_E2][kx_E2[mask2_E2] < 0.] = vy_nQ2[mask2_E2][kx_E2[mask2_E2] < 0.]
				
			else:
				vx1_E1[kx_E1 < 0.] = vx_n1[kx_E1 < 0.]
				vy1_E1[kx_E1 < 0.] = vy_n1[kx_E1 < 0.]
				
				vx2_E2[kx_E2 < 0.] = vx_nQ2[kx_E2 < 0.]
				vy2_E2[kx_E2 < 0.] = vy_nQ2[kx_E2 < 0.]
		else:
			vx1_E1 = vx_n1
			vy1_E1 = vy_n1
			vx2_E1 = vx_nQ1
			vy2_E1 = vy_nQ1

			vx1_E2 = vx_n2
			vy1_E2 = vy_n2
			vx2_E2 = vx_nQ2
			vy2_E2 = vy_nQ2

	vx1_E1[np.isnan(vx1_E1)] = vx_n1[np.isnan(vx1_E1)]
	vy1_E1[np.isnan(vy1_E1)] = vy_n1[np.isnan(vy1_E1)]
	vx2_E2[np.isnan(vx2_E2)] = vx_nQ2[np.isnan(vx2_E2)]
	vy2_E2[np.isnan(vy2_E2)] = vy_nQ2[np.isnan(vy2_E2)]

	mask1 = (ky_E1 != 100)
	mask2 = (ky_E2 != 100)

	kx_E1 = kx_E1[mask1].flatten()
	ky_E1 = ky_E1[mask1].flatten()
	kx_E2 = kx_E2[mask2].flatten()
	ky_E2 = ky_E2[mask2].flatten()
	xi_k_E1 = xi_k_E1[mask1].flatten()
	xi_kQ_E1 = xi_kQ_E1[mask1].flatten()
	xi_k_E2 = xi_k_E2[mask2].flatten()
	xi_kQ_E2 = xi_kQ_E2[mask2].flatten()
	delta_k_E1 = delta_k_E1[mask1].flatten()
	delta_kQ_E1 = delta_kQ_E1[mask1].flatten()
	delta_k_E2 = delta_k_E2[mask2].flatten()
	delta_kQ_E2 = delta_kQ_E2[mask2].flatten()
	M_k_E1 = M_k_E1[mask1].flatten()
	M_k_E2 = M_k_E2[mask2].flatten()
	E1_k = E1[mask1].flatten()
	E2_k = E2[mask2].flatten()
	dkl_E1 = dkl_E1[mask1].flatten()
	dkl_E2 = dkl_E2[mask2].flatten()
	v1x = vx1_E1[mask1].flatten()
	v1y = vy1_E1[mask1].flatten()
	v2x = vx2_E2[mask2].flatten()
	v2y = vy2_E2[mask2].flatten()
	v1 = v1[mask1].flatten()
	v2 = v2[mask2].flatten()
	dxi_E1 = dxi_E1[mask1].flatten()
	dxi_E2 = dxi_E2[mask2].flatten()
	
	sort1 = np.argsort(np.abs(v1x))
	sort2 = np.argsort(np.abs(E2_k))
	
	fermi_deriv1 = fermi_deriv(E1_k,T)
	fermi_deriv2 = fermi_deriv(E2_k,T)
	
	return kx_E1, ky_E1, kx_E2, ky_E2, xi_k_E1, xi_kQ_E1, xi_k_E2, xi_kQ_E2, delta_k_E1, delta_kQ_E1, delta_k_E2, delta_kQ_E2, M_k_E1, M_k_E2, E1_k, E2_k, v1x, v1y, v2x, v2y, v1, v2, dkl_E1, dkl_E2, fermi_deriv1, fermi_deriv2, dxi_E1, dxi_E2

def n_scattering(f,xi,xi_Q,dl,v,dxi):

	m = int(len(xi)/31.)

	xi_gpu = cuda.mem_alloc(xi.nbytes)
	xi_Q_gpu = cuda.mem_alloc(xi_Q.nbytes)
	dl_gpu = cuda.mem_alloc(dl.nbytes)
	v_gpu = cuda.mem_alloc(v.nbytes)

	cuda.memcpy_htod(xi_gpu,xi)
	cuda.memcpy_htod(xi_Q_gpu,xi_Q)
	cuda.memcpy_htod(dl_gpu,dl)
	cuda.memcpy_htod(v_gpu,v)
	
	Tau_inv = np.zeros(len(xi)*(int(len(xi)/(m))+1))
	Tau_inv_gpu = cuda.mem_alloc(Tau_inv.nbytes)
	cuda.memcpy_htod(Tau_inv_gpu,Tau_inv)

	n_scattering_gpu(np.int32(len(xi)),np.int32(m),np.double(f),xi_gpu,xi_Q_gpu,dl_gpu,v_gpu,Tau_inv_gpu, grid = (int(len(xi)/16)+1,int((int(len(xi)/m))/16)+1,1), block = (16,16,1))
	
	cuda.memcpy_dtoh(Tau_inv,Tau_inv_gpu)
	
	Tau = []
	for i in range(0,len(xi)):
		Tau_inv_sum = np.sum(Tau_inv[i*(int(len(xi)/m)+1):(i+1)*(int(len(xi)/m)+1)])
		
		if np.abs(Tau_inv_sum) > 0.:
			Tau.append(1./(Tau_inv_sum*dxi[i]))
		else:
			Tau.append(0.)
	xi_gpu.free()
	xi_Q_gpu.free()
	dl_gpu.free()
	v_gpu.free()
	Tau_inv_gpu.free()

	return np.array(Tau)

def SCSDW_scattering(f,M1,M2,delta1,delta_Q1,delta2,delta_Q2,xi1,xi_Q1,xi2,xi_Q2,E1,E2,dl1,dl2,v1,v2,dxi1,dxi2):
	
	m1 = int(len(xi1)/63.)
	m2 = int(len(xi2)/63.)
	
	xi1_gpu = cuda.mem_alloc(xi1.nbytes)
	xi_Q1_gpu = cuda.mem_alloc(xi_Q1.nbytes)
	xi2_gpu = cuda.mem_alloc(xi2.nbytes)
	xi_Q2_gpu = cuda.mem_alloc(xi_Q2.nbytes)
	delta1_gpu = cuda.mem_alloc(delta1.nbytes)
	delta_Q1_gpu = cuda.mem_alloc(delta_Q1.nbytes)
	delta2_gpu = cuda.mem_alloc(delta2.nbytes)
	delta_Q2_gpu = cuda.mem_alloc(delta_Q2.nbytes)
	M1_gpu = cuda.mem_alloc(M1.nbytes)
	M2_gpu = cuda.mem_alloc(M2.nbytes)
	dl1_gpu = cuda.mem_alloc(dl1.nbytes)
	dl2_gpu = cuda.mem_alloc(dl2.nbytes)
	v1_gpu = cuda.mem_alloc(v1.nbytes)
	v2_gpu = cuda.mem_alloc(v2.nbytes)
	dxi1_gpu = cuda.mem_alloc(dxi1.nbytes)
	dxi2_gpu = cuda.mem_alloc(dxi2.nbytes)

	cuda.memcpy_htod(xi1_gpu,xi1)
	cuda.memcpy_htod(xi_Q1_gpu,xi_Q1)
	cuda.memcpy_htod(xi2_gpu,xi2)
	cuda.memcpy_htod(xi_Q2_gpu,xi_Q2)
	cuda.memcpy_htod(delta1_gpu,delta1)
	cuda.memcpy_htod(delta_Q1_gpu,delta_Q1)
	cuda.memcpy_htod(delta2_gpu,delta2)
	cuda.memcpy_htod(delta_Q2_gpu,delta_Q2)
	cuda.memcpy_htod(M1_gpu,M1)
	cuda.memcpy_htod(M2_gpu,M2)
	cuda.memcpy_htod(dl1_gpu,dl1)
	cuda.memcpy_htod(dl2_gpu,dl2)
	cuda.memcpy_htod(v1_gpu,v1)
	cuda.memcpy_htod(v2_gpu,v2)
	cuda.memcpy_htod(dxi1_gpu,dxi1)
	cuda.memcpy_htod(dxi2_gpu,dxi2)
	
	Tau11_inv = np.zeros(len(xi1)*(int(len(xi1)/(m1))+1))
	Tau11_inv_gpu = cuda.mem_alloc(Tau11_inv.nbytes)
	cuda.memcpy_htod(Tau11_inv_gpu,Tau11_inv)

	Tau12_inv = np.zeros(len(xi1)*(int(len(xi2)/(m2))+1))
	Tau12_inv_gpu = cuda.mem_alloc(Tau12_inv.nbytes)
	cuda.memcpy_htod(Tau12_inv_gpu,Tau12_inv)
	
	SCSDW_scattering_gpu(np.int32(len(xi1)),np.int32(len(xi2)),np.int32(m1),np.double(f),M1_gpu,M2_gpu,delta1_gpu,delta_Q1_gpu,delta2_gpu,delta_Q2_gpu,xi1_gpu,xi_Q1_gpu,xi2_gpu,xi_Q2_gpu,dl1_gpu,dl2_gpu,dxi1_gpu,dxi2_gpu,v1_gpu,v2_gpu,Tau11_inv_gpu,np.int32(11), grid = (int(len(xi1)/16)+1,int((int(len(xi1)/m1))/16)+1,1), block = (16,16,1))
	SCSDW_scattering_gpu(np.int32(len(xi1)),np.int32(len(xi2)),np.int32(m2),np.double(f),M1_gpu,M2_gpu,delta1_gpu,delta_Q1_gpu,delta2_gpu,delta_Q2_gpu,xi1_gpu,xi_Q1_gpu,xi2_gpu,xi_Q2_gpu,dl1_gpu,dl2_gpu,dxi1_gpu,dxi2_gpu,v1_gpu,v2_gpu,Tau12_inv_gpu,np.int32(12), grid = (int(len(xi1)/16)+1,int((int(len(xi2)/m2))/16)+1,1), block = (16,16,1))
	
	cuda.memcpy_dtoh(Tau11_inv,Tau11_inv_gpu)
	cuda.memcpy_dtoh(Tau12_inv,Tau12_inv_gpu)

	Tau11_inv_sum = np.zeros(xi1.shape)
	Tau12_inv_sum = np.zeros(xi1.shape)
	for i in range(0,len(xi1)):
		Tau11_inv_sum[i] = np.sum(Tau11_inv[i*(int(len(xi1)/m1)+1):(i+1)*(int(len(xi1)/m1)+1)]) #+ np.sum(Tau12_inv[i*(int(len(xi1)/m1)+1):(i+1)*(int(len(xi1)/m1)+1)])
		Tau12_inv_sum[i] = np.sum(Tau12_inv[i*(int(len(xi2)/m2)+1):(i+1)*(int(len(xi2)/m2)+1)])
	Tau1 = []
	for i in range(0,len(xi1)):
		Tau1_inv_sum = Tau11_inv_sum[i] + Tau12_inv_sum[i]
		if Tau1_inv_sum > 1e-8:
			Tau1.append(1./(Tau1_inv_sum))
		else:
			Tau1.append(0.)
	
	Tau21_inv = np.zeros(len(xi2)*(int(len(xi1)/m1)+1))
	Tau21_inv_gpu = cuda.mem_alloc(Tau21_inv.nbytes)
	cuda.memcpy_htod(Tau21_inv_gpu,Tau21_inv)
	
	Tau22_inv = np.zeros(len(xi2)*(int(len(xi2)/m2)+1))
	Tau22_inv_gpu = cuda.mem_alloc(Tau22_inv.nbytes)
	cuda.memcpy_htod(Tau22_inv_gpu,Tau22_inv)

	SCSDW_scattering_gpu(np.int32(len(xi1)),np.int32(len(xi2)),np.int32(m2),np.double(f),M1_gpu,M2_gpu,delta1_gpu,delta_Q1_gpu,delta2_gpu,delta_Q2_gpu,xi1_gpu,xi_Q1_gpu,xi2_gpu,xi_Q2_gpu,dl1_gpu,dl2_gpu,dxi1_gpu,dxi2_gpu,v1_gpu,v2_gpu,Tau22_inv_gpu,np.int32(22), grid = (int(len(xi2)/16)+1,int((int(len(xi2)/m2))/16)+1,1), block = (16,16,1))
	SCSDW_scattering_gpu(np.int32(len(xi1)),np.int32(len(xi2)),np.int32(m1),np.double(f),M1_gpu,M2_gpu,delta1_gpu,delta_Q1_gpu,delta2_gpu,delta_Q2_gpu,xi1_gpu,xi_Q1_gpu,xi2_gpu,xi_Q2_gpu,dl1_gpu,dl2_gpu,dxi1_gpu,dxi2_gpu,v1_gpu,v2_gpu,Tau21_inv_gpu,np.int32(21), grid = (int(len(xi2)/16)+1,int((int(len(xi1)/m1))/16)+1,1), block = (16,16,1))
	
	cuda.memcpy_dtoh(Tau21_inv,Tau21_inv_gpu)
	cuda.memcpy_dtoh(Tau22_inv,Tau22_inv_gpu)
	
	Tau22_inv_sum = np.zeros(xi2.shape)
	Tau21_inv_sum = np.zeros(xi2.shape)
	for i in range(0,len(xi2)):
		Tau22_inv_sum[i] = np.sum(Tau22_inv[i*(int(len(xi2)/m2)+1):(i+1)*(int(len(xi2)/m2)+1)])
		Tau21_inv_sum[i] = np.sum(Tau21_inv[i*(int(len(xi1)/m1)+1):(i+1)*(int(len(xi1)/m1)+1)])
	Tau2 = []
	for i in range(0,len(xi2)):
		Tau2_inv_sum = Tau22_inv_sum[i] + Tau21_inv_sum[i]
		if Tau2_inv_sum > 1e-8:
			Tau2.append(1./(Tau2_inv_sum))
		else:
			Tau2.append(0.)
	xi1_gpu.free()
	xi_Q1_gpu.free()
	xi2_gpu.free()
	xi_Q2_gpu.free()
	delta1_gpu.free()
	delta_Q1_gpu.free()
	delta2_gpu.free()
	delta_Q2_gpu.free()
	M1_gpu.free()
	M2_gpu.free()
	dl1_gpu.free()
	dl2_gpu.free()
	dxi1_gpu.free()
	dxi2_gpu.free()
	v1_gpu.free()
	v2_gpu.free()
	Tau11_inv_gpu.free()
	Tau12_inv_gpu.free()
	Tau21_inv_gpu.free()
	Tau22_inv_gpu.free()
	
	return np.array(Tau1),np.array(Tau2)

def n_conductivity(dispersion,delta_function,N_xi,N_phi,E_cutoff,f,Q,t1,t2,T):
	
	kx,ky,kxQ,kyQ,xi_k1,xi_kQ1,xi_k2,xi_kQ2,delta_k1,delta_kQ1,delta_k2,delta_kQ2,M_k1,M_k2,Ek1,Ek2,v1x,v1y,v2x,v2y,v1,v2,dl1,dl2,fermi_deriv1,fermi_deriv2,dxi1,dxi2 = FS_grid(dispersion,'swave',delta_function,0.,M_aniso,0.,Q,T,N_xi,N_phi,t1,t2,E_cutoff,'normal2')
	
	badDeltaMask1 = np.abs(Ek1) < 0.8 * max(Ek1)
	badDeltaMask2 = np.abs(Ek2) < 0.8 * max(Ek2)
	
	Tau1_n = n_scattering(f,xi_k1,xi_kQ1,dl1,v1,dxi1)
	
	k1 = np.sum((v1x[badDeltaMask1]**2.)*(xi_k1[badDeltaMask1]**2.)*(Tau1_n[badDeltaMask1])*fermi_deriv1[badDeltaMask1]*dl1[badDeltaMask1]*dxi1[badDeltaMask1]/v1[badDeltaMask1])/T
	
	kxx1 = k1
	
	sort1 = np.argsort(Tau1_n[badDeltaMask1])
	
	TauFig = plt.figure(figsize=(24,24))
	plt.title(r'$\tau_n$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx[badDeltaMask1][sort1],ky[badDeltaMask1][sort1],c = Tau1_n[badDeltaMask1][sort1],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/TauNormal_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	
	TauFig.clear()
	plt.close(TauFig)
	del TauFig

	EFig = plt.figure(figsize=(24,24))
	plt.title(r'$\xi_k$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx[badDeltaMask1],ky[badDeltaMask1],c = xi_k1[badDeltaMask1],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/xi_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	
	EFig.clear()
	plt.close(EFig)
	del EFig

	vxFig = plt.figure(figsize=(24,24))
	plt.title(r'$v_{x}^2$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx[badDeltaMask1],ky[badDeltaMask1],c = v1x[badDeltaMask1]**2,cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/v2Normal_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')

	vxFig.clear()
	plt.close(vxFig)
	del vxFig

	sort1 = np.argsort(fermi_deriv1[badDeltaMask1])

	fermiDerivFig = plt.figure(figsize=(24,24))
	plt.title(r'$\frac{\partial f}{\partial \xi} (\xi_k)$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{SDW}$' + ')')
	plt.scatter(kx[badDeltaMask1][sort1],ky[badDeltaMask1][sort1],c = fermi_deriv1[badDeltaMask1][sort1],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/fermiDerivNormal_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	fermiDerivFig.clear()
	plt.close(fermiDerivFig)
	del fermiDerivFig
	
	plt.close("all")

	return kxx1

def SCSDW_conductivity(dispersion,delta_function,M_order,sc_order,N_xi,N_phi,E_cutoff,f,Q,t1,t2,T,delta,M):
	kx_E1,ky_E1,kx_E2,ky_E2,xi_k1,xi_kQ1,xi_k2,xi_kQ2,delta_k1,delta_kQ1,delta_k2,delta_kQ2,M_k1,M_k2,Ek1,Ek2,v1x,v1y,v2x,v2y,v1,v2,dl1,dl2,fermi_deriv1,fermi_deriv2,dxi1,dxi2 = FS_grid(dispersion,sc_order,delta_function,delta,M_order,M,Q,T,N_xi,N_phi,t1,t2,E_cutoff,'SDW2_aniso')
	
	badDeltaMask1 = (np.abs(Ek1) < .8 * max(Ek1)) & (kx_E1 > 0) & (kx_E1 > np.pi/2 - ky_E1)
	badDeltaMask2 = (np.abs(Ek2) < .8 * max(Ek2)) & ((kx_E2 < 0) | ((kx_E2 > 0) & (kx_E2 > np.pi/2 - ky_E2)))
	kx_tmp = np.abs(kx_E2)
	ky_tmp = np.abs(ky_E2)
	kparr = 2*(kx_tmp+ky_tmp)/np.pi
	kperp = 2*(-kx_tmp+ky_tmp)/np.pi
	kparr_new = kparr - 1
	kx_new = -np.pi/4. + np.pi*(kparr_new-kperp)/4.
	ky_new = -np.pi/4. + np.pi*(kparr_new+kperp)/4.
	kx_E2_plot = np.sign(kx_E2)*np.abs(kx_new)
	ky_E2_plot = np.sign(ky_E2)*np.abs(ky_new)

	kx_tmp = np.abs(kx_E1)
	ky_tmp = np.abs(ky_E1)
	kparr = 2*(kx_tmp+ky_tmp)/np.pi
	kperp = 2*(-kx_tmp+ky_tmp)/np.pi
	kparr_new = kparr - 1
	kx_new = -np.pi/4. + np.pi*(kparr_new-kperp)/4.
	ky_new = -np.pi/4. + np.pi*(kparr_new+kperp)/4.
	kx_E1_plot = np.sign(kx_E1)*np.abs(kx_new)
	ky_E1_plot = np.sign(ky_E1)*np.abs(ky_new)

	Tau1_SCSDW,Tau2_SCSDW = SCSDW_scattering(f,M_k1,M_k2,delta_k1,delta_kQ1,delta_k2,delta_kQ2,xi_k1,xi_kQ1,xi_k2,xi_kQ2,Ek1,Ek2,dl1,dl2,v1,v2,dxi1,dxi2)
	
	#Tau1_SCSDW[kx_E1 < 0.] = Tau2_SCSDW[kx_E2 < 0.]

	if len(Ek1[badDeltaMask1]) > 0:
		k1 = np.sum((v1x[badDeltaMask1]**2.)*(Ek1[badDeltaMask1]**2.)*(Tau1_SCSDW[badDeltaMask1])*fermi_deriv1[badDeltaMask1]*dl1[badDeltaMask1]*dxi1[badDeltaMask1]/(v1[badDeltaMask1]))/T
		kxx1 = k1
	else:
		kxx1 = 0
	k2 = np.sum((v2x[badDeltaMask2]**2.)*(Ek2[badDeltaMask2]**2.)*(Tau2_SCSDW[badDeltaMask2])*fermi_deriv2[badDeltaMask2]*dl2[badDeltaMask2]*dxi2[badDeltaMask2]/(v2[badDeltaMask2]))/T
	
	kxx2 = k2
	
	if len(Ek1[badDeltaMask1]) > 0:
		k1 = np.sum((v1y[badDeltaMask1]**2.)*(Ek1[badDeltaMask1]**2.)*(Tau1_SCSDW[badDeltaMask1])*fermi_deriv1[badDeltaMask1]*dl1[badDeltaMask1]*dxi1[badDeltaMask1]/(v1[badDeltaMask1]))/T
		kyy1 = k1
	else:
		kyy1 = 0
	k2 = np.sum((v2y[badDeltaMask2]**2.)*(Ek2[badDeltaMask2]**2.)*(Tau2_SCSDW[badDeltaMask2])*fermi_deriv2[badDeltaMask2]*dl2[badDeltaMask2]*dxi2[badDeltaMask2]/(v2[badDeltaMask2]))/T
	
	kyy2 = k2
	
	if len(Ek1[badDeltaMask1]) > 0:
		k1 = np.sum((v1x[badDeltaMask1]*v1y[badDeltaMask1])*(Ek1[badDeltaMask1]**2.)*(Tau1_SCSDW[badDeltaMask1])*fermi_deriv1[badDeltaMask1]*dl1[badDeltaMask1]*dxi1[badDeltaMask1]/(v1[badDeltaMask1]))/T
		kxy1 = k1 
	else:
		kxy1 = 0 
	k2 = np.sum((v2x[badDeltaMask2]*v2y[badDeltaMask2])*(Ek2[badDeltaMask2]**2.)*(Tau2_SCSDW[badDeltaMask2])*fermi_deriv2[badDeltaMask2]*dl2[badDeltaMask2]*dxi2[badDeltaMask2]/(v2[badDeltaMask2]))/T
	
	kxy2 = k2
	
	if len(Ek1[badDeltaMask1]) > 0:
		sort1 = np.argsort(Tau1_SCSDW[badDeltaMask1])
	sort2 = np.argsort(Tau2_SCSDW[badDeltaMask2])
	
	TauFig = plt.figure(figsize=(32,24))
	plt.subplot(1,2,1)
	plt.title(r'$\tau_1$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	if len(Ek1[badDeltaMask1]) > 0:
		plt.scatter(kx_E1_plot[badDeltaMask1][sort1],ky_E1_plot[badDeltaMask1][sort1],c = Tau1_SCSDW[badDeltaMask1][sort1],cmap = 'inferno')
		plt.colorbar()
	plt.subplot(1,2,2)
	plt.title(r'$\tau_2$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx_E2_plot[badDeltaMask2][sort2],ky_E2_plot[badDeltaMask2][sort2],c = Tau2_SCSDW[badDeltaMask2][sort2],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/Tau_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	TauFig.clear()
	plt.close(TauFig)
	del TauFig
	
	EFig = plt.figure(figsize=(32,24))
	plt.subplot(1,2,1)
	plt.title(r'$E^{(1)}_k$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	if len(Ek1[badDeltaMask1]) > 0:
		plt.scatter(kx_E1_plot[badDeltaMask1],ky_E1_plot[badDeltaMask1],c = Ek1[badDeltaMask1],cmap = 'inferno')
		plt.colorbar()
	plt.subplot(1,2,2)
	plt.title(r'$E^{(2)}_k$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx_E2_plot[badDeltaMask2],ky_E2_plot[badDeltaMask2],c = Ek2[badDeltaMask2],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/E_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	
	EFig.clear()
	plt.close(EFig)
	del EFig
	
	vxFig = plt.figure(figsize=(32,24))
	plt.subplot(1,2,1)
	plt.title(r'$v^{(1)}_{x}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	if len(Ek1[badDeltaMask1]) > 0:
		plt.scatter(kx_E1_plot[badDeltaMask1],ky_E1_plot[badDeltaMask1],c = v1x[badDeltaMask1],cmap = 'inferno')
		plt.colorbar()
	plt.subplot(1,2,2)
	plt.title(r'$v^{(2)}_{x}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx_E2_plot[badDeltaMask2],ky_E2_plot[badDeltaMask2],c = v2x[badDeltaMask2],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/vx_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	
	vxFig.clear()
	plt.close(vxFig)
	del vxFig
	
	vyFig = plt.figure(figsize=(32,24))
	plt.subplot(1,2,1)
	plt.title(r'$v^{(1)}_{y}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	if len(Ek1[badDeltaMask1]) > 0:
		plt.scatter(kx_E1_plot[badDeltaMask1],ky_E1_plot[badDeltaMask1],c = v1y[badDeltaMask1],cmap = 'inferno')
		plt.colorbar()
	plt.subplot(1,2,2)
	plt.title(r'$v^{(2)}_{y}$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx_E2_plot[badDeltaMask2],ky_E2_plot[badDeltaMask2],c = v2y[badDeltaMask2],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/vy_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')

	vyFig.clear()
	plt.close(vyFig)
	del vyFig

	if len(Ek1[badDeltaMask1]) > 0:
		sort1 = np.argsort(fermi_deriv1[badDeltaMask1])
	sort2 = np.argsort(fermi_deriv2[badDeltaMask2])

	fermiDerivFig = plt.figure(figsize=(32,24))
	plt.subplot(1,2,1)
	plt.title(r'$\frac{\partial f}{\partial E} (E^{(1)}_k)$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	if len(Ek1[badDeltaMask1]) > 0:
		plt.scatter(kx_E1_plot[badDeltaMask1][sort1],ky_E1_plot[badDeltaMask1][sort1],c = fermi_deriv1[badDeltaMask1][sort1],cmap = 'inferno')
		plt.colorbar()
	plt.subplot(1,2,2)
	plt.title(r'$\frac{\partial f}{\partial E} (E^{(2)}_k)$' + r'($T$=' + str(float(int(1000*T))/1000) + r'$T_{N}$' + ')')
	plt.scatter(kx_E2_plot[badDeltaMask2][sort2],ky_E2_plot[badDeltaMask2][sort2],c = fermi_deriv2[badDeltaMask2][sort2],cmap = 'inferno')
	plt.colorbar()
	plt.savefig('./picsHeatmapSCSDW/fermiDeriv_' + str(N_xi) + '_' + str(int(1000*T)) + '.png')
	
	fermiDerivFig.clear()
	plt.close(fermiDerivFig)
	del fermiDerivFig
	
	plt.close("all")
	return kxx1,kxx2,kyy1,kyy2,kxy1,kxy2
	
def deltaIntegrandEven(phi,r1,r2,delta,deltaFunc,M,n,T):
	Y_k = deltaFunc(phi)
	delta_k = delta*Y_k
	
	wn = 2*np.pi*T*(n + .5)
	
	ret = r2*(delta/wn - (delta_k*Y_k)/np.sqrt(wn**2 + delta_k**2)) + r1*(delta/wn - .5*(delta_k + M)*Y_k/np.sqrt(wn**2 + (M+delta_k)**2) - .5*(delta_k - M)*Y_k/np.sqrt(wn**2 + (M-delta_k)**2))
	
	return ret

def MIntegrandEven(phi,delta,deltaFunc,M,n,T):
	Y_k = deltaFunc(phi)
	delta_k = delta*Y_k
	
	wn = 2*np.pi*T*(n + .5)
	
	ret = M/wn - .5*(M+delta_k)/np.sqrt(wn**2 + (M+delta_k)**2) - .5*(M-delta_k)/np.sqrt(wn**2 + (M-delta_k)**2)
	
	return ret

def deltaSelfConsistencyEven(r1,r2,delta,deltaFunc,M,T,Tc,m):
	
	ret = delta*np.log(T/Tc)
	
	for n in range(0,int(m/T)):
		ret = ret + 2.*np.pi*T*quad(lambda phi: deltaIntegrandEven(phi,r1,r2,delta,deltaFunc,M,n,T),0.,2.*np.pi)[0]/(2.*np.pi)
	
	return ret

def MSelfConsistencyEven(delta,deltaFunc,M,T,TSDW,m):
	
	ret = M*np.log(T/TSDW)
	
	for n in range(0,int(m/T)):
		ret = ret + 2.*np.pi*T*quad(lambda phi: MIntegrandEven(phi,delta,deltaFunc,M,n,T),0,2*np.pi)[0]/(2.*np.pi)
	
	return ret

def deltaMSelfConsistencyEven(r1,r2,delta,deltaFunc,M,T,Tc,TSDW,m1,m2):
	deltaResidual = deltaSelfConsistencyEven(r1,r2,delta,deltaFunc,M,T,Tc,m1)
	MResidual = MSelfConsistencyEven(delta,deltaFunc,M,T,TSDW,m2)
	#print(T,delta,M,deltaResidual,MResidual)
	return deltaResidual, MResidual

def deltaSelfConsistencyOdd(sc_order,mu,disp,delta,deltaFunc,M,MFunc,Q,T,Tc,t1,t2,num_k,m):
	
	kx = np.linspace(-np.pi/2.,np.pi/2.,num_k)
	ky = np.linspace(-np.pi/2.,np.pi/2.,num_k)
	dk = np.pi/(num_k)
	Kx,Ky = np.meshgrid(kx,ky)
	
	Y_k = deltaFunc(Kx,Ky,1.)
	
	M_k = MFunc(Kx,Ky,M)
	
	xi_k_plus = (t2*np.cos(2.*Kx)*np.cos(2.*Ky))
	xi_k_minus = (mu - t1*(np.cos(2.*Kx)+np.cos(2.*Ky)))
	
	ret = 0.
	
	for n in range(0,int(m/T)):
		if sc_order == 'dx2y2':
			num = (2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. + xi_k_plus**2. + M_k**2.
			denom = ((2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. + xi_k_plus**2. + M_k**2.)**2. - 4.*(xi_k_plus**2.)*(xi_k_minus**2.+ M_k**2.)
		elif sc_order == 'swave' or sc_order == 'dxy':
			num = (2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. + xi_k_plus**2. + M_k**2.
			denom = ((2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. + xi_k_plus**2. + M_k**2.)**2. - 4.*(xi_k_plus**2.)*(xi_k_minus**2.+ M_k**2.) - 4.*(delta*Y_k*M_k)**2.
		
		ret = ret - T*np.sum((Y_k**2)*(num/denom))*dk*dk
		if n < m:
			num_Tc = (2.*np.pi*Tc*(float(n)+.5))**2. + xi_k_minus**2. + xi_k_plus**2.
			denom_Tc = ((2.*np.pi*Tc*(float(n)+.5))**2. + xi_k_minus**2. + xi_k_plus**2.)**2. - 4.*(xi_k_plus**2.)*(xi_k_minus**2.)
			ret = ret + Tc*np.sum((Y_k**2.)*(num_Tc/denom_Tc))*dk*dk
	return ret

def MSelfConsistencyOdd(sc_order,mu,disp,delta,deltaFunc,M,MFunc,Q,T,TSDW,t1,t2,num_k,m):
	
	kx = np.linspace(-np.pi/2.,np.pi/2.,num_k)
	ky = np.linspace(-np.pi/2.,np.pi/2.,num_k)
	dk = np.pi/(num_k)
	Kx,Ky = np.meshgrid(kx,ky)
	
	Y_k = deltaFunc(Kx,Ky,1.)
	
	M_k = MFunc(Kx,Ky,M)
	
	xi_k_plus = t2*np.cos(2.*Kx)*np.cos(2.*Ky)
	xi_k_minus = (mu - t1*(np.cos(2.*Kx)+np.cos(2.*Ky)))
	
	ret = 0.
	
	for n in range(0,int(m/T)):
		if sc_order == 'dx2y2':
			num = (2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. - xi_k_plus**2. + M_k**2.
			denom = ((2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. + xi_k_plus**2. + M_k**2.)**2. - 4.*(xi_k_plus**2.)*(xi_k_minus**2.+ M_k**2.)
		elif sc_order == 'swave' or 'dxy':
			num = (2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. - xi_k_plus**2. + M_k**2.
			denom = ((2.*np.pi*T*(float(n)+.5))**2. + delta**2.*Y_k**2. + xi_k_minus**2. + xi_k_plus**2. + M_k**2.)**2. - 4.*(xi_k_plus**2.)*(xi_k_minus**2.+ M_k**2.) - 4.*(delta*Y_k*M_k)**2.

		ret = ret - T*np.sum((num/denom))*dk*dk
		if n < m:
			num_TSDW = (2.*np.pi*TSDW*(float(n)+.5))**2. + xi_k_minus**2. - xi_k_plus**2.
			denom_TSDW = ((2.*np.pi*TSDW*(float(n)+.5))**2. + xi_k_minus**2. + xi_k_plus**2.)**2. - 4.*(xi_k_plus**2.)*(xi_k_minus**2.)
			ret = ret + TSDW*np.sum((num_TSDW/denom_TSDW))*dk*dk

	#print(T,M,ret)
	return ret

def deltaMSelfConsistencyOdd(sc_order,mu,disp,delta,deltaFunc,M,MFunc,Q,T,Tc,TSDW,t1,t2,num_k,m1,m2):
	deltaResidual = deltaSelfConsistencyOdd(sc_order,mu,disp,delta,deltaFunc,M,MFunc,Q,T,Tc,t1,t2,num_k,m1)
	MResidual = MSelfConsistencyOdd(sc_order,mu,disp,delta,deltaFunc,M,MFunc,Q,T,TSDW,t1,t2,num_k,m2)
	#print(T,delta,M,deltaResidual,MResidual)
	return deltaResidual, MResidual

def deltaT_even(T,r1,r2,deltaFunc,Tc,TSDW):
	m1 = 90
	m2 = 90
	n = len(T[T < .01])
	T = T[T >= .01]
	
	delta = np.zeros(T.shape)
	M = np.zeros(T.shape)
	M_pure = np.zeros(T.shape)
	
	for i in range(0,len(T)):
		
		if i < 1:
			solution = fsolve(lambda x: deltaMSelfConsistencyEven(r1,r2,np.abs(x[0]),deltaFunc,np.abs(x[1]),T[i],Tc,TSDW,m1,m2),[1.,1.7]) #[.1,1.] worked for swave
			delta[i] = np.abs(solution[0])
			M[i] = np.abs(solution[1])
			M_pure[i] = np.abs(fsolve(lambda x: MSelfConsistencyEven(0.,deltaFunc,np.abs(x[0]),T[i],TSDW,m2),M[i]))
		else:
			if delta[i-1] > 1e-3:
				solution = fsolve(lambda x: deltaMSelfConsistencyEven(r1,r2,np.abs(x[0]),deltaFunc,np.abs(x[1]),T[i],Tc,TSDW,m1,m2),[delta[i-1],M[i-1]])
				delta[i] = np.abs(solution[0])
				M[i] = np.abs(solution[1])
				M_pure[i] = np.abs(fsolve(lambda x: MSelfConsistencyEven(0.,deltaFunc,np.abs(x[0]),T[i],TSDW,m2),M_pure[i-1]))
			else:
				delta[i-1] = 0.
				delta[i] = 0.
				M[i-1] = M_pure[i-1]
				M[i] = np.abs(fsolve(lambda x: MSelfConsistencyEven(0.,deltaFunc,np.abs(x[0]),T[i],TSDW,m2),M_pure[i-1]))
				M_pure[i] = M[i]
			
		#print('solution: ',T[i],delta[i],M[i],M_pure[i])
	
	if n != 0:
		delta = np.concatenate((delta[0]*np.ones(n),delta))
		M = np.concatenate((M[0]*np.ones(n),M))
	"""
	plt.scatter(T,delta,c = 'blue')
	plt.scatter(T,M,c = 'red')
	plt.scatter(T,M_pure,c = 'black')
	plt.show()
	"""
	return delta,M

def deltaT_odd(T,sc_order,disp,deltaFunc,MFunc,Q,mu,Tc,TSDW,t1,t2,num_k):
	m1 = 30
	m2 = 175
	n = len(T[T < .01])
	T = T[T >= .01]
	
	delta = np.zeros(T.shape)
	M = np.zeros(T.shape)
	M_pure = np.zeros(T.shape)
	
	for i in range(0,len(T)):
		
		if i < 1:
			solution = fsolve(lambda x: deltaMSelfConsistencyOdd(sc_order,mu,disp,np.abs(x[0]),deltaFunc,np.abs(x[1]),MFunc,Q,T[i],Tc,TSDW,t1,t2,num_k,m1,m2),[.1,1.])
			delta[i] = np.abs(solution[0])
			M[i] = np.abs(solution[1])
			M_pure[i] = np.abs(fsolve(lambda x: MSelfConsistencyOdd(sc_order,mu,disp,0.,deltaFunc,np.abs(x[0]),MFunc,Q,T[i],TSDW,t1,t2,num_k,m2),M[i]))
		else:
			if delta[i-1] > 1e-3:
				solution = fsolve(lambda x: deltaMSelfConsistencyOdd(sc_order,mu,disp,np.abs(x[0]),deltaFunc,np.abs(x[1]),MFunc,Q,T[i],Tc,TSDW,t1,t2,num_k,m1,m2),[delta[i-1],M[i-1]])
				delta[i] = np.abs(solution[0])
				M[i] = np.abs(solution[1])
				M_pure[i] = np.abs(fsolve(lambda x: MSelfConsistencyOdd(sc_order,mu,disp,0.,deltaFunc,np.abs(x[0]),MFunc,Q,T[i],TSDW,t1,t2,num_k,m2),M_pure[i-1]))
			else:
				delta[i-1] = 0.
				delta[i] = 0.
				M[i-1] = M_pure[i-1]
				M[i] = np.abs(fsolve(lambda x: MSelfConsistencyOdd(sc_order,mu,disp,0.,deltaFunc,np.abs(x[0]),MFunc,Q,T[i],TSDW,t1,t2,num_k,m2),M_pure[i-1]))
				M_pure[i] = M[i]
			
		#print(T[i],delta[i],M[i],M_pure[i])
	
	if n != 0:
		delta = np.concatenate((delta[0]*np.ones(n),delta))
		M = np.concatenate((M[0]*np.ones(n),M))
	"""
	plt.scatter(T,delta,c = 'blue')
	plt.scatter(T,M,c = 'red')
	plt.scatter(T,M_pure,c = 'black')
	plt.show()
	"""
	return delta,M

def deltaT(T,sc_order,disp,deltaFunc,MFunc,Q,mu,Tc,TSDW,t1,t2):
	if sc_order == 'dx2y2':
		return deltaT_odd(T,sc_order,disp,deltaFunc_dx2y2,MFunc,Q,mu,Tc,TSDW,t1,t2,350)
	elif sc_order == 'swave':
		return deltaT_even(T,.03,.97,deltaFunc_swaveCircular,Tc,TSDW)
	elif sc_order == 'dxy':
		return deltaT_even(T,.03,.97,deltaFunc_dxyCircular,Tc,TSDW)

def MT(T,disp,MFunc,Q,mu,TSDW,t1,t2,num_k):
	m = 175
	n = len(T[T < .01])
	T = T[T >= .01]
	
	M = np.zeros(T.shape)
	
	for i in range(0,len(T)):
		
		if i < 1:
			M[i] = np.abs(fsolve(lambda x: MSelfConsistencyOdd('dx2y2',mu,disp,0.,deltaFunc_dx2y2,np.abs(x[0]),MFunc,Q,T[i],TSDW,t1,t2,num_k,m),3.))
		else:
				M[i] = np.abs(fsolve(lambda x: MSelfConsistencyOdd('dx2y2',mu,disp,0.,deltaFunc_dx2y2,np.abs(x[0]),MFunc,Q,T[i],TSDW,t1,t2,num_k,m),M[i-1]))
			
		#print(T[i],M[i])
	if n != 0:
		M = np.concatenate((M[0]*np.ones(n),M))
	"""
	plt.scatter(T,M,c = 'red')
	plt.show()
	"""
	return M
	
def main(dispersion,sc_order,p,M_order,num_T,N_xi,N_phi,E_cutoff,f,Q,t1,t2):
	File = open('SCSDW_conductivity_' + str(sc_order) + '_' + str(p) + '_' + str(N_xi) + '_' + str(E_cutoff) + '_' + str(t1) + '_' + str(t2) + '.csv', 'w')
	writer = csv.writer(File)
	print('Writing data to file: ' + 'SCSDW_conductivity_' + str(sc_order) + '_' + str(p) + '_' + str(N_xi) + '_' + str(E_cutoff) + '_' + str(t1) + '_' + str(t2) + '.csv')

	kxx_SC = []
	kyy_SC = []
	kxy_SC = []
	kparr_SC = []
	kperp_SC = []
	k_n = []
	delta = []
	M = []
	#T = np.array([.125,.15,.175,.2,.225,.25,.275,.3,.325,.35])
	#T = np.array([.375,.4,.425,.45,.475,.5,.525,.55,.575,.6])
	#T = np.array([.675,.7,.725,.75,.775,.8,.825,.85,.875,.9])
	T = np.array([.3,.31,.32,.33,.34,.35,.36,.37,.38,.39])
	#T = np.array([.05,.06,.07,.08,.09,.1,.125,.15,.175,.2])
	#T = np.array([.925,.95,.975,.99,.999,.999,.999,.999,.999,.999])
	
	"""
	if sc_order == 'pureSC_dx2y2':
		delta_function = deltaFunc_dx2y2
		delta = deltaT(T,dispersion,deltaFunc_dx2y2,0.,1,t1,t2,500)
		M = np.zeros(T.shape)
	"""
	if sc_order == 'dx2y2':
		delta_function = deltaFunc_dx2y2
		delta,M = deltaT(T,'dx2y2',dispersionTB,deltaFunc_dx2y2,M_iso,Q,0.,p,1.,t1,t2)
	elif sc_order == 'dxy':
		delta_function = deltaFunc_dxy
		delta,M = deltaT(T,'dxy',dispersionTB,deltaFunc_dxyCircular,M_iso,Q,0.,p,1.,t1,t2)
	elif sc_order == 'swave':
		delta_function = deltaFunc_swave
		delta,M = deltaT(T,'swave',dispersionTB,deltaFunc_swaveCircular,M_iso,Q,0.,p,1.,t1,t2)
	elif sc_order == 'pureSDW':
		delta_function = deltaFunc_swave
		delta = np.zeros(T.shape)
		M = MT(T,dispersionTB,M_iso,Q,0.,1.,t1,t2,350)

	for i in range(0,num_T):
		kxx1, kxx2, kyy1, kyy2, kxy1, kxy2 = SCSDW_conductivity(dispersion,delta_function,M_order,sc_order,N_xi,N_phi,T[i]*E_cutoff,f*T[i],Q,t1,t2,T[i],delta[i],M[i])
		kxx_SC.append(kxx1 + kxx2)
		kyy_SC.append(kyy1 + kyy2)
		kxy_SC.append(kxy1 + kxy2)
		kparr_SC.append(kxx_SC[i] + kxy_SC[i])
		kperp_SC.append(kxx_SC[i] - kxy_SC[i])
		kn1 = n_conductivity(dispersion,delta_function,N_xi,N_phi,T[i]*E_cutoff,f*T[i],Q,t1,t2,T[i])
		k_n.append(kn1)
		writer.writerow([T[i],M[i],delta[i],kxx_SC[i]/k_n[i],kyy_SC[i]/k_n[i],kxy_SC[i]/k_n[i],kparr_SC[i]/k_n[i],kperp_SC[i]/k_n[i],kxx1,kxx2,kn1])
		print(T[i],M[i],delta[i],kxx_SC[i]/k_n[i],kyy_SC[i]/k_n[i],kxy_SC[i]/k_n[i],kparr_SC[i]/k_n[i],kperp_SC[i]/k_n[i],kxx1,kxx2,kn1)
		gc.collect()
	File.close()

SCSDW_scattering_gpu = gpu_code.get_function("SCSDW_scattering_gpu")
n_scattering_gpu = gpu_code.get_function("n_scattering_gpu")
main(dispersionTB,'dxy',.35,M_aniso,10,301,300,15.,.25,[np.pi/2.,np.pi/2.],100.,10.)
