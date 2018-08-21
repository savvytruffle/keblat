#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "helpers.h"
//#include "polifitgsl.h"
#include <gsl/gsl_multifit.h>
#include <stdbool.h>



#define NMAX 20
#define NDIM 3
#define G 19.94
#define RMIN 0.01

#define moddiff(a,b) ((a>b) ? (a-b) : (b-a))
#define NELEMS(x)  (sizeof(x) / sizeof((x)[0]))
#define TWOPI 6.28318531
#define PI 3.14159265
#define MAX(a,b) ((a) > (b) ? a : b)
#define MIN(a,b) ((a) < (b) ? a : b)
#define ROWMINUS1(r) ((r-1)%3 < 0 ? 3+(r-1)%3 : (r-1)%3) // QATS
#define PARITY 1 // QATS: Parity of box (down = -1, up = 1)
#define TRSZ 133//35

// occultquad fns
double rc(double x, double y);
double rj(double x, double y, double z, double p);
double ellec(double k);
double ellk(double k);
double rf(double x, double y, double z);
double max(double a, double b);
double min(double a, double b);

double transit_x[TRSZ] = {-1.        , -0.99999997, -0.99999975, -0.99999893, -0.99999673,
       -0.99999187, -0.99998243, -0.99996575, -0.99993828, -0.99989547,
       -0.99983165, -0.99973989, -0.99961185, -0.9994377 , -0.99920598,
       -0.99890342, -0.99851485, -0.99802306, -0.99740861, -0.99664973,
       -0.99572212, -0.9945988 , -0.99324991, -0.99164249, -0.98974027,
       -0.98750342, -0.98488824, -0.98184684, -0.97832675, -0.97427047,
       -0.969615  , -0.96429112, -0.95822276, -0.95132602, -0.94350812,
       -0.93466609, -0.92468507, -0.91343624, -0.90077419, -0.88653352,
       -0.87052436, -0.85252643, -0.83228096, -0.80947943, -0.78374739,
       -0.75462061, -0.72150825, -0.68363324, -0.63992968, -0.6       ,
       -0.56470588, -0.52941176, -0.49411765, -0.45882353, -0.42352941,
       -0.38823529, -0.35294118, -0.31764706, -0.28235294, -0.24705882,
       -0.21176471, -0.17647059, -0.14117647, -0.10588235, -0.07058824,
       -0.03529412,  0.        ,  0.03529412,  0.07058824,  0.10588235,
        0.14117647,  0.17647059,  0.21176471,  0.24705882,  0.28235294,
        0.31764706,  0.35294118,  0.38823529,  0.42352941,  0.45882353,
        0.49411765,  0.52941176,  0.56470588,  0.6       ,  0.63992968,
        0.68363324,  0.72150825,  0.75462061,  0.78374739,  0.80947943,
        0.83228096,  0.85252643,  0.87052436,  0.88653352,  0.90077419,
        0.91343624,  0.92468507,  0.93466609,  0.94350812,  0.95132602,
        0.95822276,  0.96429112,  0.969615  ,  0.97427047,  0.97832675,
        0.98184684,  0.98488824,  0.98750342,  0.98974027,  0.99164249,
        0.99324991,  0.9945988 ,  0.99572212,  0.99664973,  0.99740861,
        0.99802306,  0.99851485,  0.99890342,  0.99920598,  0.9994377 ,
        0.99961185,  0.99973989,  0.99983165,  0.99989547,  0.99993828,
        0.99996575,  0.99998243,  0.99999187,  0.99999673,  0.99999893,
        0.99999975,  0.99999997,  1.        };
double transit_y[TRSZ] = {0.01836735,  0.03673469,  0.05510204,  0.07346939,  0.09183673,
        0.11020408,  0.12857143,  0.14693878,  0.16530612,  0.18367347,
        0.20204082,  0.22040816,  0.23877551,  0.25714286,  0.2755102 ,
        0.29387755,  0.3122449 ,  0.33061224,  0.34897959,  0.36734694,
        0.38571429,  0.40408163,  0.42244898,  0.44081633,  0.45918367,
        0.47755102,  0.49591837,  0.51428571,  0.53265306,  0.55102041,
        0.56938776,  0.5877551 ,  0.60612245,  0.6244898 ,  0.64285714,
        0.66122449,  0.67959184,  0.69795918,  0.71632653,  0.73469388,
        0.75306122,  0.77142857,  0.78979592,  0.80816327,  0.82653061,
        0.84489796,  0.86326531,  0.88163265,  0.9       ,  0.9146101 ,
        0.92606848,  0.93633897,  0.94555515,  0.95382419,  0.96123311,
        0.96785322,  0.97374342,  0.97895253,  0.98352113,  0.98748293,
        0.99086579,  0.99369256,  0.99598168,  0.99774766,  0.99900147,
        0.99975074,  1.        ,  0.99975074,  0.99900147,  0.99774766,
        0.99598168,  0.99369256,  0.99086579,  0.98748293,  0.98352113,
        0.97895253,  0.97374342,  0.96785322,  0.96123311,  0.95382419,
        0.94555515,  0.93633897,  0.92606848,  0.9146101 ,  0.9       ,
        0.88163265,  0.86326531,  0.84489796,  0.82653061,  0.80816327,
        0.78979592,  0.77142857,  0.75306122,  0.73469388,  0.71632653,
        0.69795918,  0.67959184,  0.66122449,  0.64285714,  0.6244898 ,
        0.60612245,  0.5877551 ,  0.56938776,  0.55102041,  0.53265306,
        0.51428571,  0.49591837,  0.47755102,  0.45918367,  0.44081633,
        0.42244898,  0.40408163,  0.38571429,  0.36734694,  0.34897959,
        0.33061224,  0.3122449 ,  0.29387755,  0.2755102 ,  0.25714286,
        0.23877551,  0.22040816,  0.20204082,  0.18367347,  0.16530612,
        0.14693878,  0.12857143,  0.11020408,  0.09183673,  0.07346939,
        0.05510204,  0.03673469,  0.01836735};

/* ================================================== */
/* =============== occultquad algs ================== */
/* =============  Mandel & Agol 2002  =============== */
/* ================================================== */

int occultquad(double *mu0, double *muo1, double *z0, double u1, double u2, double p, int nz) 
{
/*	Input: *************************************
 
	 z0   impact parameter in units of rs
	 u1   linear    limb-darkening coefficient (gamma_1 in paper)
	 u2   quadratic limb-darkening coefficient (gamma_2 in paper)
	 p    occulting star size in units of rs
	 nz   number of points in the light curve (# elements in z0)
	
	 Output: ***********************************
	
	 default:  muo1, fraction of flux at each z0 for a limb-darkened source
	 optional:  mu0, fraction of flux at each z0 for a uniform source
	
	 Limb darkening has the form:
	 I(r)=[1-u1*(1-sqrt(1-(r/rs)^2))-u2*(1-sqrt(1-(r/rs)^2))^2]/(1-u1/3-u2/6)/pi
*/
    int i;
    double *lambdad, *etad, *lambdae, pi, x1, x2, x3, z, omega, kap0 = 0.0, kap1 = 0.0, \
		q, Kk=0., Ek=0., Pk=0., n;//, tol = 1e-8;

    lambdad = (double *)malloc(nz*sizeof(double));
    lambdae = (double *)malloc(nz*sizeof(double));
    etad = (double *)malloc(nz*sizeof(double));

    if(fabs(p - 0.5) < 1.0e-4) p = 0.5;

    omega=1.0-u1/3.0-u2/6.0;
    pi=acos(-1.0);

    // trivial case of no planet
    if(p<=0.) {for(i=0;i<nz;i++) {muo1[i]=1.; mu0[i]=1.;}}

    for(i = 0; i < nz; i++)
    {	
        z = z0[i];

//        if(fabs(p - z) < tol) z = p;
//        if(fabs((p-1.)-z) < tol) z = p-1.;
//        if(fabs((1.-p)-z) < tol) z = 1.-p;
//        if(z < tol) z = 0.;

        x1 = pow((p - z), 2.0);
        x2 = pow((p + z), 2.0);
        x3 = p*p - z*z;

        //source is unocculted:
        if(z >= 1.0 + p)					
        {
            //printf("zone 1\n");
            lambdad[i] = 0.0;
            etad[i] = 0.0;
            lambdae[i] = 0.0;
            muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                            (u1 + 2.0 * u2) * (lambdad[i] + 2./3. * (p > z)) + \
                            u2 * etad[i])/omega;
            mu0[i] = 1.0 - lambdae[i];
            if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
            {
                printf("zone1: %+.3e %+.3e %+.3e %+.3e %+.4e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i]);
            }
            continue;
        }
        //source is completely occulted:
        if(p >= 1.0 && z <= p - 1.0)			
        {
            //printf("zone 2\n");
            lambdad[i] = 0.0; //from Errata
            etad[i] = 0.5;		//error in Fortran code corrected here, following Eastman's python code
            lambdae[i] = 1.0;
            muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                            (u1 + 2.0 * u2) * (lambdad[i] + 2./3. * (p > z)) + \
                            u2 * etad[i])/omega;
            mu0[i] = 1.0-lambdae[i];
            if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
            {
                printf("zone2: %+.3e %+.3e %+.3e %+.3e %+.4e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i]);
            }
            continue;
        }

        //source is partly occulted and occulting object crosses the limb:
        if(z >= fabs(1.0 - p) && z <= 1.0 + p)	
        {				
            //printf("zone 3\n");
            kap1 = acos(min((1.0-p*p+z*z)/2.0/z,1.0));
            kap0 = acos(min((p*p+z*z-1.0)/2.0/p/z,1.0));
            lambdae[i] = p*p*kap0+kap1;
            lambdae[i] = (lambdae[i] - 0.50*sqrt(max(4.0*z*z-pow((1.0+z*z-p*p), 2.0),0.0)))/pi;
            //etad[i] = 0.5/pi * (kap1 + p*p * (p*p + 2.*z*z) * kap0 - 1. + 5.*p*p + z*z)/4. * sqrt((1.-x1)*(x2-1.))); //added from Eastman's code
            etad[i] = 0.5;
            lambdad[i] = 0.0;
        }
        //occulting object transits the source but doesn't completely cover it:
        if(z <= 1.0 - p)				
        {					
            //printf("zone 4\n");
            lambdae[i] = p*p;
        }
        //edge of the occulting star lies at the origin
        if(fabs(z-p) < 1e-4*(z+p))
        {
            //printf("zone 5\n");
            if(p == 0.5)	
            {
                //printf("zone 6\n");
                lambdad[i] = 1.0/3.0-4.0/pi/9.0;
                etad[i] = 3.0/32.0;
                muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                                (u1 + 2.0 * u2) * (lambdad[i] + 2./3. * (p > z)) + \
                                u2 * etad[i])/omega;
                mu0[i] = 1.0-lambdae[i];
                if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
                {
                    printf("zone6: %+.3e %+.3e %+.3e %+.3e %+.4e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i]);
                }
                continue;
            }
            if(z >= 0.5)
            {
                //printf("zone 5.1\n");
                q = 0.5/p;
                Kk = ellk(q);
                Ek = ellec(q);
                // lambda3, eta1 in MA02; subtract 2/3 HSF(p-z)?! verify this.
                lambdad[i] = 1.0/3.0+16.0*p/9.0/pi*(2.0*p*p-1.0)*Ek- \
				 	 	(32.0*pow(p, 4.0)-20.0*p*p+3.0)/9.0/pi/p*Kk - 2./3.*(p>z);
                etad[i] = 0.5/pi*(kap1+p*p*(p*p+2.0*z*z)*kap0 - \
				              	(1.0+5.0*p*p+z*z)/4.0*sqrt((1.0-x1)*(x2-1.0)));
				//printf("zone 5.1: %+.3e +%.3e +%.3e +%.3e +%.3e +%.3e\n", z, Kk, Ek, lambdad[i], etad[i], muo1[i]);
				//continue;
            }
            if(z<0.5)	
            {
                //printf("zone 5.2\n");
                q = 2.0*p;
                Kk = ellk(q);
                Ek = ellec(q);
                //lambda4, eta2 in MA02; NO HSF(p-z)...
                lambdad[i] = 1.0/3.0+2.0/9.0/pi*(4.0*(2.0*p*p-1.0)*Ek + (1.0-4.0*p*p)*Kk);
                etad[i] = p*p/2.0*(p*p+2.0*z*z);
                muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                                (u1 + 2.0 * u2) * (lambdad[i]) + \
                                u2 * etad[i])/omega;
                mu0[i] = 1.0-lambdae[i];
                if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
                {
                    printf("zone5.2: %+.3e %+.3e %+.3e %+.3e %+.4e %+.3e %+.3e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i], Kk, Ek);
                }
                continue;
            }
            muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                            (u1 + 2.0 * u2) * (lambdad[i] + 2./3. * (p > z)) + \
                            u2 * etad[i])/omega;
            mu0[i] = 1.0-lambdae[i];
            if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
            {
                printf("zone3,4,5,5.1: %+.3e %+.3e %+.3e %+.3e %+.4e %+.3e %+.3e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i], Kk, Ek);
            }
            //printf("zone 5.1: %+.3e +%.3e +%.3e +%.3e +%.3e +%.3e\n", z, Kk, Ek, lambdad[i], etad[i], muo1[i]);
            continue;
        }
        //occulting star partly occults the source and crosses the limb:
        if((z > 0.5 + fabs(p - 0.5) && z < 1.0 + p) || (p > 0.5 \
            && z >= fabs(1.0-p) && z < p))
        {
            //printf("zone 3.1\n");
            q = sqrt((1.0-pow((p-z), 2.0))/4.0/z/p);
            Kk = ellk(q);
            Ek = ellec(q);
            n = 1.0/x1-1.0;
            Pk = Kk-n/3.0*rj(0.0,1.0-q*q,1.0,1.0+n);
            lambdad[i] = 1.0/9.0/pi/sqrt(p*z)*(((1.0-x2)*(2.0*x2+ \
			        x1-3.0)-3.0*x3*(x2-2.0))*Kk+4.0*p*z*(z*z+ \
			        7.0*p*p-4.0)*Ek-3.0*x3/x1*Pk);
            etad[i] = 1.0/2.0/pi*(kap1+p*p*(p*p+2.0*z*z)*kap0- \
				(1.0+5.0*p*p+z*z)/4.0*sqrt((1.0-x1)*(x2-1.0)));
            muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                            (u1 + 2.0 * u2) * (lambdad[i] + 2./3.*(p > z)) + \
                            u2 * etad[i])/omega;

            mu0[i] = 1.0-lambdae[i];
            //printf("z, Kk, Ek, Pk, lambdad, etad, muo1: %+.3e %+.3e %+.3e %+.3e %+.3e %+.3e %+.3e\n", z, Kk, Ek, Pk, lambdad[i], etad[i], muo1[i]);
            if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
            {
                printf("zone3.1: %+.3e %+.3e %+.3e %+.3e %+.4e %+.3e %+.3e %+.3e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i], Kk, Ek, Pk);
            }
            continue;
        }
		//occulting star transits the source:
        if(p <= 1.0  && z <= (1.0 - p)*1.0001)	
        {
            //printf("zone 4.1\n");
            q = sqrt((x2-x1)/(1.0-x1));
            Kk = ellk(q);
            Ek = ellec(q);
            n = x2/x1-1.0;
            Pk = Kk-n/3.0*rj(0.0,1.0-q*q,1.0,1.0+n);
            lambdad[i] = 2.0/9.0/pi/sqrt(1.0-x1)*((1.0-5.0*z*z+p*p+ \
			         x3*x3)*Kk+(1.0-x1)*(z*z+7.0*p*p-4.0)*Ek-3.0*x3/x1*Pk);
            if(fabs(p+z-1.0) <= 1.0e-4)
            {
                // need the 2/3 HEAVISIDE STEP FUNCTION (p-1/2) lambda5 in MA02
                lambdad[i] = 2.0/3.0/pi*acos(1.0-2.0*p)-4.0/9.0/pi* \
				            sqrt(p*(1.0-p))*(3.0+2.0*p-8.0*p*p) - 2./3.*(p>0.5); 
            }
            etad[i] = p*p/2.0*(p*p+2.0*z*z);
        }
        muo1[i] = 1.0 - ((1.0 - u1 - 2.0 * u2) * lambdae[i] + \
                        (u1 + 2.0 * u2) * (lambdad[i] + 2./3. * (p > z)) + \
                        u2 * etad[i])/omega;
        mu0[i] = 1.0-lambdae[i];
        if((muo1[i]-1. > 1e-4) || muo1[i]<-1e-4)
        {
            printf("zone4.1: %+.3e %+.3e %+.3e %+.3e %+.4e %+.3e %+.3e %+.3e\n", z, lambdae[i], lambdad[i], etad[i], muo1[i], Kk, Ek, Pk);
        }
    }
    free(lambdae);
    free(lambdad);
    free(etad);
    //free(mu);
    return 0;
}


double min(double a, double b)
{
	if(a < b) return a;
	else return b;
}

double max(double a, double b)
{
	if(a > b) return a;
	else return b;
}

double rc(double x, double y)
{
	double rc, ERRTOL,TINY,SQRTNY,BIG,TNBG,COMP1,COMP2,THIRD,C1,C2, C3,C4;
	ERRTOL=0.04; TINY=1.69e-38; SQRTNY=1.3e-19; BIG=3.0e37;
	TNBG=TINY*BIG; COMP1=2.236/SQRTNY; COMP2=TNBG*TNBG/25.0;
	THIRD=1.0/3.0; C1=0.3; C2=1.0/7.0; C3=0.375; C4=9.0/22.0;

	double alamb,ave,s,w,xt,yt;
	if(x < 0.0 || y == 0.0 || (x+fabs(y)) < TINY || (x+fabs(y)) > BIG || (y < -COMP1 && x > 0 && x < COMP2)){
		printf("Invalid argument(s) in rc\n");
		return 0;
	}
	if(y > 0.0)
	{
		xt=x;
		yt=y;
		w=1.0;
	}
	else
	{
		xt=x-y;
		yt=-y;
		w=sqrt(x)/sqrt(xt);
	}
	s = ERRTOL*10.0;
	while(fabs(s) > ERRTOL)
	{
		alamb = 2.0*sqrt(xt)*sqrt(yt)+yt;
		xt = 0.25*(xt+alamb);
		yt  =0.25*(yt+alamb);
		ave = THIRD*(xt+yt+yt);
		s = (yt-ave)/ave;
	}
	rc = w*(1.0+s*s*(C1+s*(C2+s*(C3+s*C4))))/sqrt(ave);
	return rc;
}

double rj(double x, double y, double z, double p)
{
	double rj, ERRTOL,TINY,BIG,C1,C2,C3,C4,C5,C6,C7,C8, tempmax;
	
	ERRTOL=0.05; TINY=2.5e-13; BIG=9.0e11; C1=3.0/14.0;
	C2=1.0/3.0; C3=3.0/22.0; C4=3.0/26.0; C5=.750*C3;
     	C6=1.50*C4; C7=.50*C2; C8=C3+C3;
	
	double  a = 0.0,alamb,alpha,ave,b = 0.0,beta,delp,delx,dely,delz,ea,eb,ec,ed,ee, \
     		fac,pt,rcx = 0.0,rho,sqrtx,sqrty,sqrtz,sum,tau,xt,yt,zt;
      
	if(x < 0.0 || y < 0.0 || z < 0.0 || (x+y) < TINY || (x+z) < TINY || (y+z) < TINY || fabs(p) < TINY \
		|| x > BIG || y > BIG || z > BIG || fabs(p) > BIG)
	{
		return 0;
	}
	sum=0.0;
	fac=1.0;
	if(p > 0.0)
	{
		xt=x;
		yt=y;
		zt=z;
		pt=p;
	}
	else
	{
		xt = min(x, y);
		xt = min(xt,z);
		zt = max(x, y);
		zt = max(zt, z);
		yt = x+y+z-xt-zt;
		a = 1.0/(yt-p);
		b = a*(zt-yt)*(yt-xt);
		pt = yt+b;
		rho = xt*zt/yt;
		tau = p*pt/yt;
		rcx = rc(rho,tau);
	}
	tempmax = ERRTOL*10.0;
	while(tempmax > ERRTOL)
	{
		sqrtx = sqrt(xt);
		sqrty = sqrt(yt);
		sqrtz = sqrt(zt);
		alamb = sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
		alpha = pow((pt*(sqrtx+sqrty+sqrtz)+sqrtx*sqrty*sqrtz), 2.0);
		beta = pt*(pt+alamb)*(pt + alamb);
		sum = sum+fac*rc(alpha,beta);
		fac = 0.25*fac;
		xt = 0.25*(xt+alamb);
		yt = 0.25*(yt+alamb);
		zt = 0.250*(zt+alamb);
		pt = 0.25*(pt+alamb);
		ave = 0.2*(xt+yt+zt+pt+pt);
		delx = (ave-xt)/ave;
		dely = (ave-yt)/ave;
		delz = (ave-zt)/ave;
		delp = (ave-pt)/ave;
		tempmax = max(fabs(delx), fabs(dely));
		tempmax = max(tempmax, fabs(delz));
		tempmax = max(tempmax, fabs(delp));
	}
	ea = delx*(dely+delz)+dely*delz;
	eb = delx*dely*delz;
	ec = delp*delp;
	ed = ea-3.0*ec;
	ee = eb+2.0*delp*(ea-ec);
	rj = 3.0*sum+fac*(1.0+ed*(-C1+C5*ed-C6*ee)+eb*(C7+delp*(-C8+delp*C4)) +\
		delp*ea*(C2-delp*C3)-C2*delp*ec)/(ave*sqrt(ave));
	if(p < 0.0) rj=a*(b*rj+3.0*(rcx-rf(xt,yt,zt)));
	return rj;  
}
	
double ellec(double k)
{

	double m1,a1,a2,a3,a4,b1,b2,b3,b4,ee1,ee2,ellec;
	// Computes polynomial approximation for the complete elliptic
	// integral of the second kind (Hasting's approximation):
	m1=1.0-k*k;
	a1=0.44325141463;
	a2=0.06260601220;
	a3=0.04757383546;
	a4=0.01736506451;
	b1=0.24998368310;
	b2=0.09200180037;
	b3=0.04069697526;
	b4=0.00526449639;
	ee1=1.0+m1*(a1+m1*(a2+m1*(a3+m1*a4)));
	ee2=m1*(b1+m1*(b2+m1*(b3+m1*b4)))*log(1.0/m1);
	ellec=ee1+ee2;
    /*
	double rd(double x, double y, double z);
	double rf(double x, double y, double z);
	double q;

	q=(1.0-k)*(1.0+k);
	ellec=rf(0, q, 1.0)-sqrt(k)*rd(0, q, 1.0)/3.0;
    */
	return ellec;
}

double ellk(double k)
{

	double a0,a1,a2,a3,a4,b0,b1,b2,b3,b4,ellk, ek1,ek2,m1;
	// Computes polynomial approximation for the complete elliptic
	// integral of the first kind (Hasting's approximation):
	m1=1.0-k*k;
	a0=1.38629436112;
	a1=0.09666344259;
	a2=0.03590092383;
	a3=0.03742563713;
	a4=0.01451196212;
	b0=0.5;
	b1=0.12498593597;
	b2=0.06880248576;
	b3=0.03328355346;
	b4=0.00441787012;
	ek1=a0+m1*(a1+m1*(a2+m1*(a3+m1*a4)));
	ek2=(b0+m1*(b1+m1*(b2+m1*(b3+m1*b4))))*log(m1);
	ellk=ek1-ek2;
     /*
	double rf(double x, double y, double z);
	ellk=rf(0, (1.0-k)*(1.0+k), 1.0);
     */
	return ellk;
}

double rf(double x, double y, double z)
{
	double rf, ERRTOL,TINY,BIG,THIRD,C1,C2,C3,C4, tempmax;
	
	ERRTOL=0.08; TINY=1.5e-38; BIG=3.0e37; THIRD=1.0/3.0;
	C1=1.0/24.0; C2=0.1; C3=3.0/44.0; C4=1.0/14.0;
	
	double alamb,ave,delx,dely,delz,e2,e3,sqrtx,sqrty,sqrtz,xt,yt,zt;

	if(min(x,y) < 0.0 || z < 0.0 || min(x+y, x+z) < TINY || y+z < TINY || max(x,y) > BIG || z > BIG)
	{
		printf("Invalid argument(s) in rf\n");
		return 0;
	}
	xt=x;
	yt=y;
	zt=z;
	tempmax = ERRTOL*10.0;
	while(tempmax > ERRTOL)
	{
		sqrtx=sqrt(xt);
		sqrty=sqrt(yt);
		sqrtz=sqrt(zt);
		alamb=sqrtx*(sqrty+sqrtz)+sqrty*sqrtz;
		xt=0.25*(xt+alamb);
		yt=0.25*(yt+alamb);
		zt=0.25*(zt+alamb);
		ave=THIRD*(xt+yt+zt);
		delx=(ave-xt)/ave;
		dely=(ave-yt)/ave;
		delz=(ave-zt)/ave;
		tempmax = max(fabs(delx), fabs(dely));
		tempmax = max(tempmax, fabs(delz));
	}
	e2=delx*dely-delz*delz;
	e3=delx*dely*delz;
	rf=(1.0+(C1*e2-C2-C3*e3)*e2+C4*e3)/sqrt(ave);
	return rf;
}


/* ================================================== */
/* =============== QATS algorithms ================== */
/* =============  Carter & Agol 2012  =============== */
/* ================================================== */
int test(int a, int b, int *c) {
  if (a < b) {*c=a;}
  else if (a > b) {*c=b;}
  return 0;
}


void shConvol(double *y, int N, int q, double *d) {
  // Box convolution (computes D in eqn 15.
  // Assumed d has size d[N-q+1] and y has size y[N]
  int n, j;
  for (n=0; n<=N-q; n++) {
    d[n] = 0;
    for (j=0; j<q; j++) {
      d[n] += PARITY*y[j+n];
    }
  }
}

double maximum(double *v, int lb, int rb, int *index) {
  // Maximum of vector v btw. indices lb and rb. Index of maximum is index.
  double m = v[lb];
  int i;
  *index = lb;
  for (i=lb; i<=rb; i++) {
    if (v[i] > m) {
      m = v[i];
      *index = i;
    }
  }
  return m;
}

void omegaBounds(int m, int M, int DeltaMin, int DeltaMax, int N, int q, int *lb, int *rb) {
  // Returns bounds for omega set as listed in paper (eqns 16-18)
  *lb = ((m-1)*DeltaMin > N-DeltaMax-(M-m)*DeltaMax ? (m-1)*DeltaMin : N-DeltaMax-(M-m)*DeltaMax);
  *rb = (DeltaMax-q+(m-1)*DeltaMax < N-q-(M-m)*DeltaMin ? DeltaMax-q+(m-1)*DeltaMax : N-q-(M-m)*DeltaMin);
}

void gammaBounds(int m, int n, int DeltaMin, int DeltaMax, int q, int *lb, int *rb) {
  // Returns bounds for gamma set as listed in paper (eqns 19-21)
  *lb = ((m-1)*DeltaMin > n-DeltaMax ? (m-1)*DeltaMin : n-DeltaMax);
  *rb = (DeltaMax-q+(m-1)*DeltaMax < n-DeltaMin ? DeltaMax-q+(m-1)*DeltaMax : n-DeltaMin);
}


void computeSmnRow(double *d, int M, int m, int DeltaMin, int DeltaMax, int N, int q, double **Smn) {
  // Recursive generation of Smn (eqns 22, 23) called by computeSmn
  int omegaLb, omegaRb, gammaLb, gammaRb, index, n;
  if (m !=1 ) {computeSmnRow(d, M, m-1, DeltaMin, DeltaMax, N, q, Smn);}
  omegaBounds(m, M, DeltaMin, DeltaMax, N, q, &omegaLb, &omegaRb);
  for (n=omegaLb; n <= omegaRb; n++) {
    if (m==1) {
      Smn[m-1][n] = d[n];
    }
    else {
      gammaBounds(m-1, n, DeltaMin, DeltaMax, q, &gammaLb, &gammaRb);
      Smn[m-1][n] = d[n]+maximum(Smn[m-2], gammaLb, gammaRb, &index);
    }
  }
}

void computeSmn(double *d, int M, int DeltaMin, int DeltaMax, int N, int q, double **Smn, double *Sbest) {
  // Assumed Smn has been allocated as M X N-q+1, d is data, others are from paper (eqns 22, 23).
  // Sbest holds merit for this set of parameters. Memory Intensive; refer to computeSmnInPlace.
  int omegaLb, omegaRb, index;
  computeSmnRow(d, M, M, DeltaMin, DeltaMax, N, q, Smn);
  omegaBounds(M, M, DeltaMin, DeltaMax, N, q, &omegaLb, &omegaRb);
  *Sbest = maximum(Smn[M-1], omegaLb, omegaRb, &index) / sqrt(((double)M) * q);
}

void computeSmnInPlace(double *d, int M, int DeltaMin, int DeltaMax, int N, int q, double **Smn, double *Sbest) {
  // Assumed Smn has been allocated as 3 x N-q+1. Use this algorithm to compute 'Sbest' from Smn
  // (eqns 22, 23) when returned indices are not important. This computation should be used to
  // produce the "spectrum." Much less memory intensive than computing Smn in full.
  int omegaLb, omegaRb, gammaLb, gammaRb, index, m, n;
  int r = 1;
  for (m=1; m <= M; m++) {
    omegaBounds(m, M, DeltaMin, DeltaMax, N, q, &omegaLb, &omegaRb);
    if (m != 1) {
      for (n=omegaLb; n<=omegaRb; n++) {
        gammaBounds(m-1, n, DeltaMin, DeltaMax, q, &gammaLb, &gammaRb);
        Smn[r][n] = d[n] + maximum(Smn[ROWMINUS1(r)], gammaLb, gammaRb, &index);
      }
    }
    else {
      for (n=omegaLb; n<=omegaRb; n++) {
        Smn[r][n] = d[n] ;
      }
    }
    r = (r+1) % 3;
  }

  omegaBounds(M, M, DeltaMin, DeltaMax, N, q, &omegaLb, &omegaRb);
  *Sbest = maximum(Smn[ROWMINUS1(r)], omegaLb, omegaRb, &index) / sqrt(((double)M) * q);
}

void optIndices(int M, int DeltaMin, int DeltaMax, int N, int q, double **Smn, int *indices) {
  // Optimal starting indices for M, DeltaMin, DeltaMax, q (according to eqns. 25, 26).
  // Given Smn (Eqns. 22,23)
  // Assumed indices has been allocated as having M values. Called by qats_indices.
  int omegaLb, omegaRb, gammaLb, gammaRb, index, m;
  omegaBounds(M, M, DeltaMin, DeltaMax, N, q, &omegaLb, &omegaRb);
  maximum(Smn[M-1], omegaLb, omegaRb, &index);
  indices[M-1] = index;
  for (m=M-1; m>=1; m--) {
    gammaBounds(m, indices[m], DeltaMin, DeltaMax, q, &gammaLb, &gammaRb);
    maximum(Smn[m-1], gammaLb, gammaRb, &index);
    indices[m-1] = index;
  }
}

void qats(double *d, int DeltaMin, int DeltaMax, int N, int q, double *Sbest, int *Mbest) {
  // Determine highest likelihood for this set of input parameters
  // (Algorithm 1, for a single value of q).  Start indices are not returned.
  // Useful for calculating the QATS "spectrum."
  // Assumed d is pre-convolved data (D in Eqn. 15  -- see shConvol above)
  int MMin = floor((N+q-1.0)/DeltaMax);
  int MMax = floor((N-q-0.0)/DeltaMin) + 1;
  double c_Sbest;
  int M, i;

  *Sbest = -100000000.;

  for (M=MMin; M <= MMax; M++) {
    double **Smn = malloc(3 * sizeof(*Smn));
    for (i=0;i<3;i++) {
      Smn[i] = malloc((N-q+1) * sizeof(*(Smn[i])));
    }
    computeSmnInPlace(d, M, DeltaMin, DeltaMax, N, q, Smn, &c_Sbest);
    if (c_Sbest > *Sbest) {
      *Sbest = c_Sbest;
      *Mbest = M;
    }
    //else{
    //  printf("c_Sbest<=Sbest %d %12.6f %12.6f\n", M, c_Sbest, Sbest);
    //}
    for (i=0;i<3;i++) {
      free(Smn[i]);
    }
    free(Smn);
  }
}

void qats_indices(double *d, int M, int DeltaMin, int DeltaMax, int N, int q, double *Sbest, int *indices) {
  // For a specific collection of M, DeltaMin, DeltaMax, and q find the optimal starting indices of transit.
  // Use after finding the most likely parameters with detectMqNoTimes or even wider search on DeltaMin, DeltaMax or q.
  double **Smn = malloc(M * sizeof(*Smn));
  int i;
  for (i = 0; i < M; i++) {
    Smn[i] = malloc((N-q+1) * sizeof(*(Smn[i])));
  }
  computeSmn(d, M, DeltaMin, DeltaMax, N, q, Smn, Sbest);
  optIndices(M, DeltaMin, DeltaMax, N, q, Smn, indices);
  for (i = 0; i < M; i++) {
    free(Smn[i]);
  }
  free(Smn);
}


int get_qats_indices(int M, int DeltaMin, int DeltaMax, int q, int N, double *y, int *indices, double *Sbest) {
  //since q=1, no need for convolution
  //double *d = malloc(N*sizeof(double));
  //shConvol(y, N, q, d);
  qats_indices(y, M, DeltaMin, DeltaMax, N, q, Sbest, indices);
  //free(d);
  return 0;
}

int get_qats_likelihood(int DeltaMin, int DeltaMax, int q, int N, double *y, double *Sbest, int *Mbest){
  //since q=1, no need for convolution
  //double *d = malloc(N*sizeof(double));
  //shConvol(y, N, q, d); // Calculate D_n
  qats(y, DeltaMin, DeltaMax, N, q, Sbest, Mbest);
  //free(d);
  return 0;
}


/* =========================================================== */
/* =============== polynomial, transit fits ================== */
/* =========================================================== */
double poly_eval(double *coeffs, int Ncoeffs, double x) {
  // Evaluates polynomial given coefficients, Ncoeff = degree + 1, and data array x
  int i;
  double ans = 0.0;
  /*
  if (Ncoeffs<1) {
    printf("Ncoeffs = %in", Ncoeffs);
  }
  */
  for (i=0; i<Ncoeffs; i++) {
    ans = ans + coeffs[i]*pow(x, i);
  }
  return ans;
}

int binarysearch(double *A, double key, long ilo, long ihi, long sz) {
    if (ihi < ilo) {
        if (ilo > sz) {
            return sz;
        }
        return ilo;
    }
    long imid = (ilo + ihi) / 2;
    if (A[imid] > key) {
        return binarysearch(A, key, ilo, imid-1, sz);
    }
    else if (A[imid] < key) {
        return binarysearch(A, key, imid+1, ihi, sz);
    }
    else {
        return imid;
    }
}

double interp1d(double x0, double x1, double y0, double y1, double xnew) {
    int i;
    double ynew;
    ynew = y0*(1. - (xnew-x0)/(x1-x0)) + y1*(xnew - x0)/(x1-x0);
    return ynew;
}

int resample_time(double *t, double *x, double *tbary, double *xshift, long sz) {
  /* resample time from t to t-tbary, assuming delchisq at transits are PEAKS */

  long i, j, i_lo, i_hi, Npts;
  double t_lo, t_hi, xnew, temp;
  for (i = 0; i < sz-1; i++){
    t_lo = t[i] - tbary[i];
    t_hi = t[i+1] - tbary[i+1];
    if (t_lo>t_hi) {
        temp = t_lo;
        t_lo = t_hi;
        t_hi = temp;
    }
    i_lo = binarysearch(t, t_lo, 0, sz, sz);
    i_hi = binarysearch(t, t_hi, 0, sz, sz);
/*    printf("tlo,t_hi,i_lo,i_hi = %12.6f, %12.6f, %i, %i\n", t_lo, t_hi, i_lo, i_hi); */
    Npts = i_hi-i_lo;
    if (Npts>0) {
        for (j = 0; j < Npts; j++) {
            xnew = interp1d(t_lo, t_hi, x[i], x[i+1], t[i_lo+j]);
            // change >= to <= if delchisq at transits are DIPS
            if (xnew>=xshift[i_lo+j]) {
                xshift[i_lo+j] = xnew;
            /*printf("%li, %li, %12.6f, %12.6f, %12.6f\n", i, i_lo+j, t[i_lo+j], xnew, xshift[i_lo+j]);*/
            }
        }
    }
  }
  
  return 0;
}

bool polynomialfit_w(int obs, int degree, 
		   double *dx, double *dy, double *derr, double *store) /* n, p */
{
  /*
  if (obs<1) {printf("Nobs = %i; ", obs);}
  if (degree<1) {printf("degree = %i\n", degree);}
  */
  gsl_multifit_linear_workspace *ws;
  gsl_matrix *cov, *X;
  gsl_vector *y, *w, *c;
  double chisq;
 
  int i, j;
 
  X = gsl_matrix_alloc(obs, degree);
  y = gsl_vector_alloc(obs);
  w = gsl_vector_alloc(obs);
  c = gsl_vector_alloc(degree);
  cov = gsl_matrix_alloc(degree, degree);
 
  for(i=0; i < obs; i++) {
    for(j=0; j < degree; j++) {
      gsl_matrix_set(X, i, j, pow(dx[i], j));
    }
    gsl_vector_set(y, i, dy[i]);
    gsl_vector_set(w, i, 1.0/(derr[i]*derr[i]));
  }
 
  ws = gsl_multifit_linear_alloc(obs, degree);
  gsl_multifit_wlinear(X, w, y, c, cov, &chisq, ws);
 
  /* store result ... */
  for(i=0; i < degree; i++)
  {
    store[i] = gsl_vector_get(c, i);
  }
 
  gsl_multifit_linear_free(ws);
  gsl_matrix_free(X);
  gsl_matrix_free(cov);
  gsl_vector_free(y);
  gsl_vector_free(w);
  gsl_vector_free(c);
  return true; /* we do not "analyse" the result (cov matrix mainly)
		  to know if the fit is "good" */
}

bool polynomialfit(int obs, int degree, 
		   double *dx, double *dy, double *store) /* n, p */
{
  gsl_multifit_linear_workspace *ws;
  gsl_matrix *cov, *X;
  gsl_vector *y, *c;
  double chisq;
 
  int i, j;
 
  X = gsl_matrix_alloc(obs, degree);
  y = gsl_vector_alloc(obs);
  c = gsl_vector_alloc(degree);
  cov = gsl_matrix_alloc(degree, degree);
 
  for(i=0; i < obs; i++) {
    for(j=0; j < degree; j++) {
      gsl_matrix_set(X, i, j, pow(dx[i], j));
    }
    gsl_vector_set(y, i, dy[i]);
  }
 
  ws = gsl_multifit_linear_alloc(obs, degree);
  gsl_multifit_linear(X, y, c, cov, &chisq, ws);
 
  /* store result ... */
  for(i=0; i < degree; i++)
  {
    store[i] = gsl_vector_get(c, i);
  }
 
  gsl_multifit_linear_free(ws);
  gsl_matrix_free(X);
  gsl_matrix_free(cov);
  gsl_vector_free(y);
  gsl_vector_free(c);
  return true; /* we do not "analyse" the result (cov matrix mainly)
		  to know if the fit is "good" */
}

double mean(double *x, unsigned long sz) {
  double meanval = 0.0;
  unsigned long i;
  for (i=0;i<sz;i++) {
    meanval = meanval + x[i];
  }
  meanval = meanval / (double)sz;
  return meanval;
}

/* sorts the input array in place... 
void quick_sort(int *a, int n) {
    int i, j, p, t;
    if (n < 2)
        return;
    p = a[n / 2];
    for (i = 0, j = n - 1;; i++, j--) {
        while (a[i] < p)
            i++;
        while (p < a[j])
            j--;
        if (i >= j)
            break;
        t = a[i];
        a[i] = a[j];
        a[j] = t;
    }
    quick_sort(a, i);
    quick_sort(a + i, n - i);
}
*/

double utransit(double *x, double *y, double xnew) {
  double *coeffs;
  double ynew;
  coeffs = malloc(3*sizeof(double));
  polynomialfit(3, 3, x, y, coeffs);
  ynew = poly_eval(coeffs, 3, xnew);
  free(coeffs);
  return ynew;
}


/* computing delta chi-square (chi_polyonly - chi_poly*tran) NO transit masking
int dchi_fn(double *dchi, double *t, long *jumps, double *f, double *ef, double *depth, 
		long *duration, int ndep, int ndur, int njumps, long *porder, long *cwidth, long Ntot) {
  long lw, mt, rw;
  int i, j, k, ii, npts;
  double *tt, *ff, *eff, *fmod, *efmod, *coeff_tran, *coeff_notran;
  double chisq_notran, chisq_tran, tmp_notran, tmp_tran, tmean;
  double *u_x, *u_y;
  u_x = malloc(3*sizeof(double));
  u_y = malloc(3*sizeof(double));
  u_y[0] = 1.0;
  u_y[2] = 1.0;
  for (i=0; i < ndep; i++) {
    u_y[1] = 1.0 - depth[i];
    for (j=0; j < ndur; j++) {
      npts = duration[j]+2*cwidth[j];
      tt = malloc(npts*sizeof(double));
      ff = malloc(npts*sizeof(double));
      eff = malloc(npts*sizeof(double));
      fmod = malloc(npts*sizeof(double));
      efmod = malloc(npts*sizeof(double));
      coeff_tran = malloc((porder[j]+1)*sizeof(double));
      coeff_notran = malloc((porder[j]+1)*sizeof(double));
      for (k=0; k < njumps-1; k++) {

        //lt = jumps[k]+MIN(10, cwidth[j]);
        //lw = lt - cwidth[j];
        //rt = lt + duration[j] - 1;
        //mt = lt + duration[j]/2;
        //rw = rt + cwidth[j];

        lw = jumps[k];
        mt = jumps[k] + cwidth[j] + duration[j]/2;
        rw = jumps[k] + 2*cwidth[j] + duration[j] - 1;
        //printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);
        while (rw < jumps[k+1]) {
        //while (rt < jumps[k+1]-MIN(10, cwidth[j])-1) {
          chisq_notran=0;
          chisq_tran=0;
          lw = MAX(lw, jumps[k]);
          rw = MIN(rw, jumps[k+1]-1);
          npts = rw-lw+1;
          for (ii=0; ii < npts; ii++) {
            tt[ii] = t[lw+ii];
            ff[ii] = f[lw+ii];
            eff[ii] = ef[lw+ii];

            //fmod[ii] = f[lw+ii];
            //efmod[ii] = ef[lw+ii];
            //if (cwidth[j]+1 <= ii && ii<=duration[j]+cwidth[j]-2) {
              //fmod[ii] = fmod[ii] / (1.-depth[i]);
              //efmod[ii] = ef[lw+ii] / (1.-depth[i]);
            //}

          }
          tmean = mean(tt, npts);
          for (ii=0; ii<npts;ii++) {
            tt[ii] = tt[ii] - tmean;
          }
          u_x[0]=tt[cwidth[j]];
          u_x[1]=tt[cwidth[j]+duration[j]/2];
          u_x[2]=tt[duration[j]+cwidth[j]-1];
          for (ii=0; ii<npts;ii++) {
            fmod[ii] = 1.0;
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
              fmod[ii] = utransit(u_x, u_y, tt[ii]);
            }
            efmod[ii] = eff[ii] / fmod[ii];
            fmod[ii] = ff[ii] / fmod[ii];
          }
          //if (i==0 && j==0 && k%10==0 && lw==jumps[0]) { printf("\n"); }
          polynomialfit_w(npts, porder[j]+1, tt, ff, eff, coeff_notran);
          polynomialfit_w(npts, porder[j]+1, tt, fmod, efmod, coeff_tran);

          //for (ii=0;ii<porder[j]+1; ii++) {
            //printf("%.8e ", coeff_notran[ii]);
          //}
          //printf("\n");

	      for (ii=0; ii < npts; ii++) {
            //printf("%10.8f ", poly_eval(coeff_notran, porder[j]+1, tt[ii]));
            tmp_notran = (poly_eval(coeff_notran, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
            tmp_tran = (poly_eval(coeff_tran, porder[j]+1, tt[ii]) - fmod[ii])/(efmod[ii]);
            chisq_notran = chisq_notran + tmp_notran*tmp_notran;
            chisq_tran = chisq_tran + tmp_tran*tmp_tran;
            //if (ii % 6==0) {printf("\n");}
          }
          dchi[i*ndur*Ntot + j*Ntot + mt] = chisq_notran-chisq_tran;

          //lt+=1;
          //lw=lt-cwidth[j];
          //mt+=1;
          //rt+=1;
          //rw=rt+cwidth[j];

          lw=lw+1;
          mt=mt+1;
          rw=rw+1; 
        }
        //printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);
      }
      free(tt);
      free(ff);
      free(eff);
      free(fmod);
      free(efmod);
    }
  }
  return 0;
}
*/

int dchiChoosePC(double *dchi, double *t, int *rinds, double *f, double *ef, double depth, int *duration,
                 int ndur, int Nrinds, int porder, int cwidth) {
  int ii, jj, j, k, npts, i_lo, lw, mt;
  double *tt, *ff, *eff, *fmod, *efmod, *coeffs, *ttt, *fff, *efff;
  double chisq_notran, chisq_tran, t_tmp, tmp_notran, tmp_tran, tmean;
  coeffs = malloc((porder+1)*sizeof(double));
  printf("ok here we go\n");
  printf("%d, %d, %d = \n", Nrinds, porder, cwidth);
  for (j=0;j<ndur;j++) {
    npts = duration[j] + 2*cwidth;
    tt = malloc(npts*sizeof(double));
    ff = malloc(npts*sizeof(double));
    eff = malloc(npts*sizeof(double));
    fmod = malloc(npts*sizeof(double));
    efmod = malloc(npts*sizeof(double));
    ttt = malloc((npts-duration[j])*sizeof(double));
    fff = malloc((npts-duration[j])*sizeof(double));
    efff = malloc((npts-duration[j])*sizeof(double));
    printf("j=%d\n", j);
    for (k=0;k<Nrinds;k++) {
      mt = rinds[k];
      lw = rinds[k] - cwidth - duration[j]/2;
      //rw = rinds[k] + cwidth + duration[j]/2 - 1;
      chisq_notran=0;
      chisq_tran=0;
      for (ii=0;ii<npts;ii++) {
        tt[ii] = t[lw+ii];
        ff[ii] = f[lw+ii];
        eff[ii] = ef[lw+ii];
      }
      tmean = mean(tt, npts);
      for (ii=0; ii<npts;ii++) {
        tt[ii] = tt[ii] - tmean;
      }
      for (ii=0; ii<npts;ii++) {
        fmod[ii] = 1.0;
        if (cwidth <= ii && ii <=duration[j]+cwidth-1) {
          t_tmp = tt[ii]/tt[cwidth];
          if (t_tmp <= transit_x[0] | t_tmp >= transit_x[TRSZ-1]) {
            fmod[ii] = 1.0;
          }
          else {
            i_lo = binarysearch(transit_x, t_tmp, 0, TRSZ, TRSZ);
            fmod[ii] = 1. - depth * interp1d(transit_x[i_lo], transit_x[i_lo+1], transit_y[i_lo], transit_y[i_lo+1], t_tmp);
            //pow((1.-(tt[ii]*tt[ii])/(tt[cwidth[j]]*tt[cwidth[j]])), 0.2);//utransit(u_x, u_y, tt[ii]);
          }
        }
        efmod[ii] = eff[ii] / fmod[ii];
        fmod[ii] = ff[ii] / fmod[ii];
      }
      jj=0;
      for (ii=0; ii<npts;ii++) {
        if (cwidth <= ii && ii <=duration[j]+cwidth-1) {
        }
        else {
          ttt[jj] = tt[ii];
          fff[jj] = ff[ii];
          efff[jj] = eff[ii];
          jj+=1;
        }
      }
      polynomialfit_w(npts-duration[j], porder+1, ttt, fff, efff, coeffs);
      for (ii=0; ii < npts; ii++) {
        /*printf("%10.8f ", poly_eval(coeff_notran, porder[j]+1, tt[ii]));*/
        tmp_notran = (poly_eval(coeffs, porder+1, tt[ii]) - ff[ii])/eff[ii];
        tmp_tran = (poly_eval(coeffs, porder+1, tt[ii]) - fmod[ii])/(efmod[ii]);
        chisq_notran = chisq_notran + tmp_notran*tmp_notran;
        chisq_tran = chisq_tran + tmp_tran*tmp_tran;
        /*if (ii % 6==0) {
          printf("\n");
        }*/
      }
      dchi[j*Nrinds + k] = chisq_notran - chisq_tran;
    }
    printf("got to here");
    free(tt);
    free(ff);
    free(eff);
    free(fmod);
    free(efmod);
    free(ttt);
    free(fff);
    free(efff);
  }
  printf("before free(coeffs); ");
  free(coeffs);
  printf("after free(coeffs)");
  return 0;
}

// dchi computation (chi_polyonly - chi_poly*tran) with transits NOT MASKED in polyfitting
// testing single cwidth sides
int dchi_fn(double *dchi, double *t, long *jumps, double *f, double *ef, double *depth,
		long *duration, int ndep, int ndur, int njumps, long *porder, long *cwidth, long Ntot) {
  long lw, mt, rw, lt, rt;
  int i, j, k, ii, npts, jj, i_lo, i_hi;
  double *tt, *ff, *eff, *fmod, *efmod, *coeffs_tran, *coeffs_notran;//, *ttt, *fff, *efff;
  double chisq_notran, chisq_tran, tmp_notran, tmp_tran, tmean, t_tmp;
  for (i=0; i < ndep; i++) {
    for (j=0; j < ndur; j++) {
      //npts = duration[j]+2*cwidth[j];
      coeffs_notran = malloc((porder[j]+1)*sizeof(double));
      coeffs_tran = malloc((porder[j]+1)*sizeof(double));

      for (k=0; k < njumps-1; k++) {
        lw = jumps[k];
        mt = jumps[k] + 5 + duration[j]/2;
        lt = jumps[k] + 5;
        rt = jumps[k] + 5 + duration[j] - 1;
        //printf("%d %d %d %d %d\n", duration[j], cwidth[j], MIN(cwidth[j], duration[j]), lw, mt);
        rw = rt + cwidth[j];//jumps[k] + 2*cwidth[j] + duration[j] - 1;
        jj=0;
        while (rt < jumps[k+1]-5) {
          chisq_notran=0;
          chisq_tran=0;
          //lw = MAX(lw, jumps[k]);
          //rw = MIN(rw, jumps[k+1]-1);
          npts = rw-lw+1;
          tt = malloc(npts*sizeof(double));
          ff = malloc(npts*sizeof(double));
          eff = malloc(npts*sizeof(double));
          fmod = malloc(npts*sizeof(double));
          efmod = malloc(npts*sizeof(double));

          for (ii=0; ii < npts; ii++) {
            tt[ii] = t[lw+ii] - t[mt];
            ff[ii] = f[lw+ii];
            eff[ii] = ef[lw+ii];

          }
          //tmean = mean(tt, npts);
          //for (ii=0; ii<npts;ii++) {
          //  tt[ii] = tt[ii] - tmean;
          //  // no weighting
          //  eff[ii] = ef[lw+ii];
          //}
          for (ii=0; ii<npts;ii++) {
            fmod[ii] = 1.0;
            if (t[lt] <= t[lw+ii] && t[lw+ii] <= t[rt]) {
              t_tmp = tt[ii]/(duration[j]*0.0204340278*0.5);
              if (t_tmp <= transit_x[0] | t_tmp >= transit_x[TRSZ-1]) {
              }
              else {
                i_lo = binarysearch(transit_x, t_tmp, 0, TRSZ, TRSZ);
                fmod[ii] = 1. - depth[i] * interp1d(transit_x[i_lo], transit_x[i_lo+1], transit_y[i_lo], transit_y[i_lo+1], t_tmp);
              }
            }
            efmod[ii] = eff[ii] / fmod[ii];
            fmod[ii] = ff[ii] / fmod[ii];
          }
          polynomialfit_w(npts, porder[j]+1, tt, ff, eff, coeffs_notran);

          polynomialfit_w(npts, porder[j]+1, tt, fmod, efmod, coeffs_tran);


	     for (ii=0; ii < npts; ii++) {
            tmp_notran = (poly_eval(coeffs_notran, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
            tmp_tran = (poly_eval(coeffs_tran, porder[j]+1, tt[ii]) - fmod[ii])/(efmod[ii]);
            chisq_notran = chisq_notran + tmp_notran*tmp_notran;
            chisq_tran = chisq_tran + tmp_tran*tmp_tran;
          }
          dchi[i*ndur*Ntot + j*Ntot + mt] = (chisq_notran/(npts-1) - chisq_tran/(npts-3));// /npts;
          mt=mt+1;
          lt=lt+1;
          rt=rt+1;
          jj+=1;
          if (mt <= jumps[k]+cwidth[j]+duration[j]/2) {lw = jumps[k];}
          else {lw=lw+1;}
          if (mt >= jumps[k+1]-1-cwidth[j]-duration[j]/2) {rw = jumps[k+1]-1;}
          else {rw=rw+1;}      
          free(tt);
          free(ff);
          free(eff);
          free(fmod);
          free(efmod);
        }
      }

      free(coeffs_notran);
      free(coeffs_tran);

    }
  }
  return 0;
}


// dchi computation (chi_polyonly - chi_poly*tran) with transits NOT MASKED in polyfitting
int dchi_fn2(double *dchi, double *t, long *jumps, double *f, double *ef, double *depth,
		long *duration, int ndep, int ndur, int njumps, long *porder, long *cwidth, long Ntot) {
  long lw, mt, rw;
  int i, j, k, ii, npts, jj, i_lo, i_hi;
  double *tt, *ff, *eff, *fmod, *efmod, *coeffs_tran, *coeffs_notran;//, *ttt, *fff, *efff;
  double chisq_notran, chisq_tran, tmp_notran, tmp_tran, tmean, t_tmp;
  //double *u_x, *u_y;
  //u_x = malloc(3*sizeof(double));
  //u_y = malloc(3*sizeof(double));
  //u_y[0] = 1.0;
  //u_y[2] = 1.0;
  for (i=0; i < ndep; i++) {
    //u_y[1] = 1.0 - depth[i];
    for (j=0; j < ndur; j++) {
      npts = duration[j]+2*cwidth[j];
      tt = malloc(npts*sizeof(double));
      ff = malloc(npts*sizeof(double));
      eff = malloc(npts*sizeof(double));
      fmod = malloc(npts*sizeof(double));
      efmod = malloc(npts*sizeof(double));
      coeffs_notran = malloc((porder[j]+1)*sizeof(double));
      coeffs_tran = malloc((porder[j]+1)*sizeof(double));
      //ttt = malloc((npts-duration[j])*sizeof(double));
      //fff = malloc((npts-duration[j])*sizeof(double));
      //efff = malloc((npts-duration[j])*sizeof(double));

      for (k=0; k < njumps-1; k++) {
        /*
        lt = jumps[k]+MIN(10, cwidth[j]);
        lw = lt - cwidth[j];
        rt = lt + duration[j] - 1;
        mt = lt + duration[j]/2;
        rw = rt + cwidth[j];
        */

        lw = jumps[k];
        mt = jumps[k] + cwidth[j] + duration[j]/2;
        rw = jumps[k] + 2*cwidth[j] + duration[j] - 1;
        jj=0;
        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
        while (rw < jumps[k+1]) {
        /*while (rt < jumps[k+1]-MIN(10, cwidth[j])-1) {*/
          chisq_notran=0;
          chisq_tran=0;
          lw = MAX(lw, jumps[k]);
          rw = MIN(rw, jumps[k+1]-1);
          npts = rw-lw+1;
          for (ii=0; ii < npts; ii++) {
            tt[ii] = t[lw+ii];
            ff[ii] = f[lw+ii];
            //eff[ii] = ef[lw+ii];
            /*
            fmod[ii] = f[lw+ii];
            efmod[ii] = ef[lw+ii];
            if (cwidth[j]+1 <= ii && ii<=duration[j]+cwidth[j]-2) {
              fmod[ii] = fmod[ii] / (1.-depth[i]);
              efmod[ii] = ef[lw+ii] / (1.-depth[i]);
            }
            */
          }
          tmean = mean(tt, npts);
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("\n%d %d %d %d %d\n", i, j, k, lw, npts);
//          }
          for (ii=0; ii<npts;ii++) {
            tt[ii] = tt[ii] - tmean;
            // no weighting
            eff[ii] = ef[lw+ii];
            // Gaussian weighting
            //eff[ii] = ef[lw+ii] / exp(-tt[ii]*tt[ii] / (2. * (duration[j]+cwidth[j]) * (duration[j]+cwidth[j]) * 0.00018188455877));
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", tt[ii]);
//            }
          }
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {printf("\n");}

          //u_x[0]=tt[cwidth[j]];
          //u_x[1]=tt[cwidth[j]+duration[j]/2];
          //u_x[2]=tt[duration[j]+cwidth[j]-1];
          for (ii=0; ii<npts;ii++) {
            fmod[ii] = 1.0;
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
              t_tmp = tt[ii]/(duration[j]*0.0204340278*0.5);
              if (t_tmp <= transit_x[0] | t_tmp >= transit_x[TRSZ-1]) {
                //fmod[ii] = 1.0;
              }
              else {
                i_lo = binarysearch(transit_x, t_tmp, 0, TRSZ, TRSZ);
                fmod[ii] = 1. - depth[i] * interp1d(transit_x[i_lo], transit_x[i_lo+1], transit_y[i_lo], transit_y[i_lo+1], t_tmp);
                //pow((1.-(tt[ii]*tt[ii])/(tt[cwidth[j]]*tt[cwidth[j]])), 0.2);//utransit(u_x, u_y, tt[ii]);
              }
            }
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", ff[ii]);
//            }
            efmod[ii] = eff[ii] / fmod[ii];
            fmod[ii] = ff[ii] / fmod[ii];
          }

//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("\n");
//            for (ii=0;ii<npts;ii++) {
//              printf("%.7f ", fmod[ii]);
//            }
//            printf("\n");
//          }
//
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            for (ii=0;ii<npts;ii++) {
//              printf("%.7f ", eff[ii]);
//            }
//            printf("\n");
//          }

          /*
          jj=0;
          for (ii=0; ii<npts;ii++) {
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
            }
            else {
              ttt[jj] = tt[ii];
              fff[jj] = ff[ii];
              efff[jj] = eff[ii];
              jj+=1;
            }
          }
          */
          //polynomialfit_w(npts-duration[j], porder[j]+1, ttt, fff, efff, coeffs);
          polynomialfit_w(npts, porder[j]+1, tt, ff, eff, coeffs_notran);

          polynomialfit_w(npts, porder[j]+1, tt, fmod, efmod, coeffs_tran);

//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            for (ii=0;ii<porder[j]+1;ii++) {
//              printf("%.7e ", coeffs_tran[ii]);
//            }
//            printf("\n");
//          }

	      for (ii=0; ii < npts; ii++) {
            /*printf("%10.8f ", poly_eval(coeff_notran, porder[j]+1, tt[ii]));*/
            tmp_notran = (poly_eval(coeffs_notran, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
            tmp_tran = (poly_eval(coeffs_tran, porder[j]+1, tt[ii]) - fmod[ii])/(efmod[ii]);
            chisq_notran = chisq_notran + tmp_notran*tmp_notran;
            chisq_tran = chisq_tran + tmp_tran*tmp_tran;
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", poly_eval(coeffs_tran, porder[j]+1, tt[ii]));
//            }

          }
          dchi[i*ndur*Ntot + j*Ntot + mt] = chisq_notran - chisq_tran;
          /*
          lt+=1;
          lw=lt-cwidth[j];
          mt+=1;
          rt+=1;
          rw=rt+cwidth[j];
          */
          lw=lw+1;
          mt=mt+1;
          rw=rw+1;
          jj+=1;
        }
        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
      }
      free(tt);
      free(ff);
      free(eff);
      free(fmod);
      free(efmod);
      //free(ttt);
      //free(fff);
      //free(efff);
      free(coeffs_notran);
      free(coeffs_tran);

    }
  }
  //free(u_x);
  //free(u_y);
  return 0;
}

// dchi computation (chi_polyonly - chi_poly*tran) with transits MASKED in polyfitting
int dchi_fn_mask(double *dchi, double *t, long *jumps, double *f, double *ef, double *depth,
		long *duration, int ndep, int ndur, int njumps, long *porder, long *cwidth, long Ntot) {
  long lw, mt, rw;
  int i, j, k, ii, npts, jj, i_lo, i_hi;
  double *tt, *ff, *eff, *fmod, *efmod, *coeffs, *ttt, *fff, *efff; //*coeffs_tran, *coeffs_notran;
  double chisq_notran, chisq_tran, tmp_notran, tmp_tran, tmean, t_tmp;
  //double *u_x, *u_y;
  //u_x = malloc(3*sizeof(double));
  //u_y = malloc(3*sizeof(double));
  //u_y[0] = 1.0;
  //u_y[2] = 1.0;
  for (i=0; i < ndep; i++) {
    //u_y[1] = 1.0 - depth[i];
    for (j=0; j < ndur; j++) {
      npts = duration[j]+2*cwidth[j];
      tt = malloc(npts*sizeof(double));
      ff = malloc(npts*sizeof(double));
      eff = malloc(npts*sizeof(double));
      fmod = malloc(npts*sizeof(double));
      efmod = malloc(npts*sizeof(double));
      //coeffs_notran = malloc((porder[j]+1)*sizeof(double));
      //coeffs_tran = malloc((porder[j]+1)*sizeof(double));
      coeffs = malloc((porder[j]+1)*sizeof(double));
      ttt = malloc((npts-duration[j])*sizeof(double));
      fff = malloc((npts-duration[j])*sizeof(double));
      efff = malloc((npts-duration[j])*sizeof(double));

      for (k=0; k < njumps-1; k++) {
        /*
        lt = jumps[k]+MIN(10, cwidth[j]);
        lw = lt - cwidth[j];
        rt = lt + duration[j] - 1;
        mt = lt + duration[j]/2;
        rw = rt + cwidth[j];
        */

        lw = jumps[k];
        mt = jumps[k] + cwidth[j] + duration[j]/2;
        rw = jumps[k] + 2*cwidth[j] + duration[j] - 1;
        jj=0;
        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
        while (rw < jumps[k+1]) {
        /*while (rt < jumps[k+1]-MIN(10, cwidth[j])-1) {*/
          chisq_notran=0;
          chisq_tran=0;
          lw = MAX(lw, jumps[k]);
          rw = MIN(rw, jumps[k+1]-1);
          npts = rw-lw+1;
          for (ii=0; ii < npts; ii++) {
            tt[ii] = t[lw+ii];
            ff[ii] = f[lw+ii];
            //eff[ii] = ef[lw+ii];
            /*
            fmod[ii] = f[lw+ii];
            efmod[ii] = ef[lw+ii];
            if (cwidth[j]+1 <= ii && ii<=duration[j]+cwidth[j]-2) {
              fmod[ii] = fmod[ii] / (1.-depth[i]);
              efmod[ii] = ef[lw+ii] / (1.-depth[i]);
            }
            */
          }
          tmean = mean(tt, npts);
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("\n%d %d %d %d %d\n", i, j, k, lw, npts);
//          }
          for (ii=0; ii<npts;ii++) {
            tt[ii] = tt[ii] - tmean;
            // no weighting
            eff[ii] = ef[lw+ii];
            // Gaussian weighting
            //eff[ii] = ef[lw+ii] / exp(-tt[ii]*tt[ii] / (2. * (duration[j]+cwidth[j]) * (duration[j]+cwidth[j]) * 0.00018188455877));
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", tt[ii]);
//            }
          }
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {printf("\n");}

          //u_x[0]=tt[cwidth[j]];
          //u_x[1]=tt[cwidth[j]+duration[j]/2];
          //u_x[2]=tt[duration[j]+cwidth[j]-1];
          jj=0;
          for (ii=0; ii<npts;ii++) {
            fmod[ii] = 1.0;
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
              t_tmp = tt[ii]/(duration[j]*0.0204340278*0.5);
              if (t_tmp <= transit_x[0] | t_tmp >= transit_x[TRSZ-1]) {
                //fmod[ii] = 1.0;
              }
              else {
                i_lo = binarysearch(transit_x, t_tmp, 0, TRSZ, TRSZ);
                fmod[ii] = 1. - depth[i] * interp1d(transit_x[i_lo], transit_x[i_lo+1], transit_y[i_lo], transit_y[i_lo+1], t_tmp);
                //pow((1.-(tt[ii]*tt[ii])/(tt[cwidth[j]]*tt[cwidth[j]])), 0.2);//utransit(u_x, u_y, tt[ii]);
              }
            }
            else {
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", ff[ii]);
//            }
              ttt[jj] = tt[ii];
              fff[jj] = ff[ii];
              efff[jj] = ef[ii];
              jj+=1;
            }
            efmod[ii] = eff[ii] / fmod[ii];
            fmod[ii] = ff[ii] / fmod[ii];
          }

//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("\n");
//            for (ii=0;ii<npts;ii++) {
//              printf("%.7f ", fmod[ii]);
//            }
//            printf("\n");
//          }
//
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            for (ii=0;ii<npts;ii++) {
//              printf("%.7f ", eff[ii]);
//            }
//            printf("\n");
//          }

          /*
          jj=0;
          for (ii=0; ii<npts;ii++) {
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
            }
            else {
              ttt[jj] = tt[ii];
              fff[jj] = ff[ii];
              efff[jj] = eff[ii];
              jj+=1;
            }
          }
          */
          polynomialfit_w(npts-duration[j], porder[j]+1, ttt, fff, efff, coeffs);
          //polynomialfit_w(npts, porder[j]+1, tt, ff, eff, coeffs_notran);

          //polynomialfit_w(npts, porder[j]+1, tt, fmod, efmod, coeffs_tran);

//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            for (ii=0;ii<porder[j]+1;ii++) {
//              printf("%.7e ", coeffs_tran[ii]);
//            }
//            printf("\n");
//          }

	      for (ii=0; ii < npts; ii++) {
            /*printf("%10.8f ", poly_eval(coeff_notran, porder[j]+1, tt[ii]));*/
            tmp_notran = (poly_eval(coeffs, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
            tmp_tran = (poly_eval(coeffs, porder[j]+1, tt[ii]) - fmod[ii])/(efmod[ii]);
            //tmp_notran = (poly_eval(coeffs_notran, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
            //tmp_tran = (poly_eval(coeffs_tran, porder[j]+1, tt[ii]) - fmod[ii])/(efmod[ii]);
            chisq_notran = chisq_notran + tmp_notran*tmp_notran;
            chisq_tran = chisq_tran + tmp_tran*tmp_tran;
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", poly_eval(coeffs_tran, porder[j]+1, tt[ii]));
//            }

          }
          dchi[i*ndur*Ntot + j*Ntot + mt] = chisq_notran/(npts-porder[j]-1) - chisq_tran/(npts-porder[j]-1-2);
          /*
          lt+=1;
          lw=lt-cwidth[j];
          mt+=1;
          rt+=1;
          rw=rt+cwidth[j];
          */
          lw=lw+1;
          mt=mt+1;
          rw=rw+1;
          jj+=1;
        }
        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
      }
      free(tt);
      free(ff);
      free(eff);
      free(fmod);
      free(efmod);
      free(ttt);
      free(fff);
      free(efff);
      //free(coeffs_notran);
      //free(coeffs_tran);
      free(coeffs);

    }
  }
  //free(u_x);
  //free(u_y);
  return 0;
}

// dchi computation but w/ gaussian weighting for polynomial fit
int dchi_fn_gs(double *dchi, double *t, long *jumps, double *f, double *ef, double *depth,
		long *duration, int ndep, int ndur, int njumps, long *porder, long *cwidth, long Ntot) {
  long lw, mt, rw;
  int i, j, k, ii, npts, jj, i_lo, i_hi;
  double *tt, *ff, *eff, *fmod, *efmod, *coeffs_tran, *coeffs_notran;//, *ttt, *fff, *efff;
  double chisq_notran, chisq_tran, tmp_notran, tmp_tran, tmean, t_tmp;
  //double *u_x, *u_y;
  //u_x = malloc(3*sizeof(double));
  //u_y = malloc(3*sizeof(double));
  //u_y[0] = 1.0;
  //u_y[2] = 1.0;
  for (i=0; i < ndep; i++) {
    //u_y[1] = 1.0 - depth[i];
    for (j=0; j < ndur; j++) {
      npts = duration[j]+2*cwidth[j];
      tt = malloc(npts*sizeof(double));
      ff = malloc(npts*sizeof(double));
      eff = malloc(npts*sizeof(double));
      fmod = malloc(npts*sizeof(double));
      efmod = malloc(npts*sizeof(double));
      coeffs_notran = malloc((porder[j]+1)*sizeof(double));
      coeffs_tran = malloc((porder[j]+1)*sizeof(double));
      //ttt = malloc((npts-duration[j])*sizeof(double));
      //fff = malloc((npts-duration[j])*sizeof(double));
      //efff = malloc((npts-duration[j])*sizeof(double));

      for (k=0; k < njumps-1; k++) {
        /*
        lt = jumps[k]+MIN(10, cwidth[j]);
        lw = lt - cwidth[j];
        rt = lt + duration[j] - 1;
        mt = lt + duration[j]/2;
        rw = rt + cwidth[j];
        */

        lw = jumps[k];
        mt = jumps[k] + cwidth[j] + duration[j]/2;
        rw = jumps[k] + 2*cwidth[j] + duration[j] - 1;
        jj=0;
        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
        while (rw < jumps[k+1]) {
        /*while (rt < jumps[k+1]-MIN(10, cwidth[j])-1) {*/
          chisq_notran=0;
          chisq_tran=0;
          lw = MAX(lw, jumps[k]);
          rw = MIN(rw, jumps[k+1]-1);
          npts = rw-lw+1;
          for (ii=0; ii < npts; ii++) {
            tt[ii] = t[lw+ii];
            ff[ii] = f[lw+ii];
            //eff[ii] = ef[lw+ii];
            /*
            fmod[ii] = f[lw+ii];
            efmod[ii] = ef[lw+ii];
            if (cwidth[j]+1 <= ii && ii<=duration[j]+cwidth[j]-2) {
              fmod[ii] = fmod[ii] / (1.-depth[i]);
              efmod[ii] = ef[lw+ii] / (1.-depth[i]);
            }
            */
          }
          tmean = mean(tt, npts);
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("\n%d %d %d %d %d\n", i, j, k, lw, npts);
//          }
          for (ii=0; ii<npts;ii++) {
            tt[ii] = tt[ii] - tmean;
            // no weighting
            //eff[ii] = ef[lw+ii];
            // Gaussian weighting
            eff[ii] = ef[lw+ii] / exp(-tt[ii]*tt[ii] / (2. * (duration[j]+cwidth[j]) * (duration[j]+cwidth[j]) * 0.00018188455877));
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", tt[ii]);
//            }
          }
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {printf("\n");}

          //u_x[0]=tt[cwidth[j]];
          //u_x[1]=tt[cwidth[j]+duration[j]/2];
          //u_x[2]=tt[duration[j]+cwidth[j]-1];
          for (ii=0; ii<npts;ii++) {
            fmod[ii] = 1.0;
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
              t_tmp = tt[ii]/(duration[j]*0.0204340278*0.5);
              if (t_tmp <= transit_x[0] | t_tmp >= transit_x[TRSZ-1]) {
                //fmod[ii] = 1.0;
              }
              else {
                i_lo = binarysearch(transit_x, t_tmp, 0, TRSZ, TRSZ);
                fmod[ii] = 1. - depth[i] * interp1d(transit_x[i_lo], transit_x[i_lo+1], transit_y[i_lo], transit_y[i_lo+1], t_tmp);
                //pow((1.-(tt[ii]*tt[ii])/(tt[cwidth[j]]*tt[cwidth[j]])), 0.2);//utransit(u_x, u_y, tt[ii]);
              }
            }
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", ff[ii]);
//            }
            efmod[ii] = eff[ii] / fmod[ii];
            fmod[ii] = ff[ii] / fmod[ii];
          }

//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("\n");
//            for (ii=0;ii<npts;ii++) {
//              printf("%.7f ", fmod[ii]);
//            }
//            printf("\n");
//          }
//
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            for (ii=0;ii<npts;ii++) {
//              printf("%.7f ", eff[ii]);
//            }
//            printf("\n");
//          }

          /*
          jj=0;
          for (ii=0; ii<npts;ii++) {
            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
            }
            else {
              ttt[jj] = tt[ii];
              fff[jj] = ff[ii];
              efff[jj] = eff[ii];
              jj+=1;
            }
          }
          */
          //polynomialfit_w(npts-duration[j], porder[j]+1, ttt, fff, efff, coeffs);
          polynomialfit_w(npts, porder[j]+1, tt, ff, eff, coeffs_notran);

          polynomialfit_w(npts, porder[j]+1, tt, fmod, efmod, coeffs_tran);

//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            for (ii=0;ii<porder[j]+1;ii++) {
//              printf("%.7e ", coeffs_tran[ii]);
//            }
//            printf("\n");
//          }

	      for (ii=0; ii < npts; ii++) {
            /*printf("%10.8f ", poly_eval(coeff_notran, porder[j]+1, tt[ii]));*/
            tmp_notran = (poly_eval(coeffs_notran, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
            tmp_tran = (poly_eval(coeffs_tran, porder[j]+1, tt[ii]) - fmod[ii])/(efmod[ii]);
            chisq_notran = chisq_notran + tmp_notran*tmp_notran;
            chisq_tran = chisq_tran + tmp_tran*tmp_tran;
//            if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//              printf("%.7f ", poly_eval(coeffs_tran, porder[j]+1, tt[ii]));
//            }

          }
          dchi[i*ndur*Ntot + j*Ntot + mt] = chisq_notran - chisq_tran;
          /*
          lt+=1;
          lw=lt-cwidth[j];
          mt+=1;
          rt+=1;
          rw=rt+cwidth[j];
          */
          lw=lw+1;
          mt=mt+1;
          rw=rw+1;
          jj+=1;
        }
        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
      }
      free(tt);
      free(ff);
      free(eff);
      free(fmod);
      free(efmod);
      //free(ttt);
      //free(fff);
      //free(efff);
      free(coeffs_notran);
      free(coeffs_tran);

    }
  }
  //free(u_x);
  //free(u_y);
  return 0;
}

//// new dchi, (N_dof - chi_poly*tran), w/ transits masked during polyfitting
//int dchi_fn(double *dchi, double *t, long *jumps, double *f, double *ef, double *depth,
//		long *duration, int ndep, int ndur, int njumps, long *porder, long *cwidth, long Ntot) {
//  long lw, mt, rw;
//  int i, j, k, ii, npts, jj;
//  double *mod, *tt, *ff, *eff, *coeffs;
//  double chisq_tran, tmp_tran, tmean;
//  double *u_x, *u_y;
//  u_x = malloc(3*sizeof(double));
//  u_y = malloc(3*sizeof(double));
//  u_y[0] = 1.0;
//  u_y[2] = 1.0;
//  for (i=0; i < ndep; i++) {
//    u_y[1] = 1.0 - depth[i];
//    for (j=0; j < ndur; j++) {
//      npts = duration[j]+2*cwidth[j];
//      tt = malloc(npts*sizeof(double));
//      mod = malloc(npts*sizeof(double));
//      coeffs = malloc((porder[j]+1)*sizeof(double));
//      //tt = malloc((npts-duration[j])*sizeof(double));
//      ff = malloc((npts)*sizeof(double));
//      eff = malloc((npts)*sizeof(double));
//
//      for (k=0; k < njumps-1; k++) {
//        /*
//        lt = jumps[k]+MIN(10, cwidth[j]);
//        lw = lt - cwidth[j];
//        rt = lt + duration[j] - 1;
//        mt = lt + duration[j]/2;
//        rw = rt + cwidth[j];
//        */
//
//        lw = jumps[k];
//        mt = jumps[k] + cwidth[j] + duration[j]/2;
//        rw = jumps[k] + 2*cwidth[j] + duration[j] - 1;
//
//        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
//        while (rw < jumps[k+1]) {
//        /*while (rt < jumps[k+1]-MIN(10, cwidth[j])-1) {*/
//          chisq_tran=0;
//          lw = MAX(lw, jumps[k]);
//          rw = MIN(rw, jumps[k+1]-1);
//          npts = rw-lw+1;
//          for (ii=0; ii < npts; ii++) {
//            tt[ii] = t[lw+ii];
//            ff[ii] = f[lw+ii];
//            eff[ii] = ef[lw+ii];
//            /*
//            fmod[ii] = f[lw+ii];
//            efmod[ii] = ef[lw+ii];
//            if (cwidth[j]+1 <= ii && ii<=duration[j]+cwidth[j]-2) {
//              fmod[ii] = fmod[ii] / (1.-depth[i]);
//              efmod[ii] = ef[lw+ii] / (1.-depth[i]);
//            }
//            */
//          }
//          tmean = mean(tt, npts);
//          for (ii=0; ii<npts;ii++) {
//            tt[ii] = tt[ii] - tmean;
//          }
//          u_x[0]=tt[cwidth[j]];
//          u_x[1]=tt[cwidth[j]+duration[j]/2];
//          u_x[2]=tt[duration[j]+cwidth[j]-1];
//          for (ii=0; ii<npts;ii++) {
//            mod[ii] = 1.0;
//            if (cwidth[j] <= ii && ii <=duration[j]+cwidth[j]-1) {
//              mod[ii] = utransit(u_x, u_y, tt[ii]);
//              //printf("tran %d %d %d %.5e\n", k, ii, jj, mod[ii]);
//            }
//            ff[ii] = ff[ii] / mod[ii];
//            eff[ii] = eff[ii] / mod[ii];
//              //printf("NOtran %d %d %d %.5e %.5e %.5e\n", k, ii, jj, tt[jj], tt0[ii], f[lw+ii]);
//
//          }
//
//          polynomialfit_w(npts, porder[j]+1, tt, ff, eff, coeffs);
//
//          /*
//          for (ii=0;ii<porder[j]+1; ii++) {
//            printf("%.8e ", coeff_notran[ii]);
//          }
//          printf("\n");
//          */
//	      for (ii=0; ii < npts; ii++) {
//            /*printf("%10.8f ", poly_eval(coeff_notran, porder[j]+1, tt[ii]));*/
//            tmp_tran = (poly_eval(coeffs, porder[j]+1, tt[ii]) - ff[ii])/eff[ii];
//            chisq_tran = chisq_tran + tmp_tran*tmp_tran;
//            /*if (ii % 6==0) {
//              printf("\n");
//            }*/
//          }
//          if (i==0 && j==0 && k%10==0 && lw==jumps[0]) {
//            printf("%d %d %d %ld %d \n", i, j, k, lw, npts);
//            for (ii=0; ii<npts;ii++) {
//              printf("%10.8f ", tt[ii]);
//            }
//            printf("\n");
//            for (ii=0; ii<npts;ii++) {
//              printf("%10.8f ", ff[ii]);
//            }
//            printf("\n");
//            for (ii=0; ii<npts;ii++) {
//              printf("%10.8f ", eff[ii]);
//            }
//            printf("\n");
//            for (ii=0;ii<porder[j]+1; ii++) {
//              printf("%.8e ", coeffs[ii]);
//            }
//            printf("\n");
//            for (ii=0; ii<npts;ii++) {
//              printf("%10.8f ", poly_eval(coeffs, porder[j]+1, tt[ii]));
//            }
//            printf("\n");
//            printf("%.8e\n", chisq_tran);
//          }
//          dchi[i*ndur*Ntot + j*Ntot + mt] = npts - porder[j] - 1 - chisq_tran;
//          /*
//          lt+=1;
//          lw=lt-cwidth[j];
//          mt+=1;
//          rt+=1;
//          rw=rt+cwidth[j];
//          */
//          lw=lw+1;
//          mt=mt+1;
//          rw=rw+1;
//        }
//        /*printf("%d %d %d %ld %ld %ld\n", i, j, k, lw, mt, rw);*/
//      }
//      free(tt);
//      free(ff);
//      free(eff);
//      free(mod);
//      free(coeffs);
//    }
//  }
//  free(u_x);
//  free(u_y);
//  return 0;
//}

int poly_lc(double *polvals, double *t, double *f, double *ef, double *model,
            long *jumps, int porder, int njumps) {
  int i, j, k, npts, ooe, porder_orig;
  double *tt0, *tt, *fmod, *efmod, *coeffs;
  double t0mean, tmean;

  for (i=0; i < njumps-1; i++) {
      npts = jumps[i+1]-jumps[i];
      ooe = 0;
      tt0 = malloc(npts*sizeof(double));

      for (j=0;j<npts;j++) {
        tt0[j] = t[jumps[i]+j];
        if (model[jumps[i]+j]>=1.) {
          ooe+=1;
        }
      }

      if (ooe<porder+1) {
        //printf("%i, %i, %.5f, %.5f\n", ooe, porder+1, model[jumps[i]], model[jumps[i+1]-1]);
        free(tt0);
        /*printf("Npts out of eclipse = %i", ooe);*/
      }
      else {
        //if ((model[jumps[i]]<1.) || (model[jumps[i+1]-1]<1.)) {porder_orig=porder; porder=1;}
        coeffs = malloc((porder+1)*sizeof(double));

        t0mean = mean(tt0, npts);

        tt = malloc(ooe*sizeof(double));
        fmod = malloc(ooe*sizeof(double));
        efmod = malloc(ooe*sizeof(double));

        k=0;
        for (j=0; j<npts; j++) {
          tt0[j] = tt0[j] - t0mean;
          if (model[jumps[i]+j]>=1.) {
            tt[k] = t[jumps[i]+j];
            fmod[k] = f[jumps[i]+j] / model[jumps[i]+j];
            efmod[k] = ef[jumps[i]+j] / model[jumps[i]+j];
            k+=1;
          }
        }

        tmean = mean(tt, ooe);
        for (k=0;k<ooe;k++) {
          tt[k] = tt[k] - tmean;
        }


        polynomialfit_w(ooe, porder+1, tt, fmod, efmod, coeffs);

        for (j=0;j<npts;j++) {
          polvals[jumps[i]+j] = poly_eval(coeffs, porder+1, tt0[j]);
        }
        //if ((model[jumps[i]]<1.) || (model[jumps[i+1]-1]<1.)) {porder=porder_orig;}
        free(tt0);
        free(tt);
        free(fmod);
        free(efmod);
        free(coeffs);
      }
    }
  return 0;
}

int poly_lc_ooe(double *polvals, double *t, double *f, double *ef, double *model,
            long *jumps, int porder, int njumps) {
  int i, j, k, npts, ooe, porder_orig;
  double *tt0, *tt, *fmod, *efmod, *coeffs;
  double t0mean, tmean;

  for (i=0; i < njumps-1; i++) {
      npts = jumps[i+1]-jumps[i];
      ooe = 0;
      tt0 = malloc(npts*sizeof(double));

      for (j=0;j<npts;j++) {
        tt0[j] = t[jumps[i]+j];
        if (model[jumps[i]+j]>0.) {
          ooe+=1;
        }
      }

      if (ooe<porder+1) {
        /*printf("%.5f, %.5f\n", model[jumps[i]], model[jumps[i+1]-1]);*/
        //printf("ooe<porder+1; %i %i\n", ooe, porder+1);
        free(tt0);
        /*printf("Npts out of eclipse = %i", ooe);*/
      }
      else {
        //if ((model[jumps[i]]<1.) || (model[jumps[i+1]-1]<1.)) {porder_orig=porder; porder=1;}
        coeffs = malloc((porder+1)*sizeof(double));

        t0mean = mean(tt0, npts);

        tt = malloc(ooe*sizeof(double));
        fmod = malloc(ooe*sizeof(double));
        efmod = malloc(ooe*sizeof(double));

        k=0;
        for (j=0; j<npts; j++) {
          tt0[j] = tt0[j] - t0mean;
          if (model[jumps[i]+j]>0.) {
            tt[k] = t[jumps[i]+j];
            fmod[k] = f[jumps[i]+j] / model[jumps[i]+j];
            efmod[k] = ef[jumps[i]+j] / model[jumps[i]+j];
            k+=1;
          }
        }

        tmean = mean(tt, ooe);
        for (k=0;k<ooe;k++) {
          tt[k] = tt[k] - tmean;
        }

        //printf("\nBefore polyfit, porder=%i, ooe=%i, npts=%i, t=%.5f\n", porder, ooe, npts, tt[0]);
        polynomialfit_w(ooe, porder+1, tt, fmod, efmod, coeffs);
        //printf("After polyfit\n");
        for (j=0;j<npts;j++) {
          polvals[jumps[i]+j] = poly_eval(coeffs, porder+1, tt0[j]);
          //printf("%.5f ", polvals[jumps[i]+j]);
        }
        //if ((model[jumps[i]]<1.) || (model[jumps[i+1]-1]<1.)) {porder=porder_orig;}
        free(tt0);
        free(tt);
        free(fmod);
        free(efmod);
        free(coeffs);
      }
    }
  return 0;
}

double getE(double M, double e)
{
    double E = M, eps = 1.0e-7;
    int niter;
    while(fabs(E - e*sin(E) - M) > eps && niter<30) {
        E = E - (E - e*sin(E) - M) / (1.0 - e*cos(E));
    }
    return E;
}

int rsky(double *t, double *f, double e, double P, double t0, double eps, int Npts)
{
    int ii;
    double M, E;
    double n=TWOPI/P;
    for (ii=0;ii<Npts;ii++) {
        if (e > eps) {
            M = n*(t[ii] - t0);
            E = getE(M, e);
            f[ii] = 2. * atan((sqrt((1.+e)/(1.-e))) * tan(E/2.0));
        }
        else {
            f[ii] = (t[ii]-t0)/P - (int)((t[ii]-t0)/P)*TWOPI;
        }
    }
    return 0;
}

