/*
This code tests the functions and subroutines you have written
associated with integrating the wquations of motions for N bodies
interacting via gravitational forces.

Compile and execute the same as the previous assignments.

*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NMAX 20
#define NDIM 3
#define G 19.94
#define RMIN 0.01
#define TWOPI 6.28318531
#define PI 3.14159265

// occultquad fns
double rc(double x, double y);
double rj(double x, double y, double z, double p);
double ellec(double k);
double ellk(double k);
double rf(double x, double y, double z);
double max(double a, double b);
double min(double a, double b);

// rsky fns
double getE(double M, double e);

// nbody fns
void center_of_mass(int N, double mass[NMAX], double r[NMAX][NDIM], double rcm[NDIM]);
double kinetic_energy(int N, double mass[NMAX], double v[NMAX][NDIM]);
double potential(int N, double mass[NMAX], double r[NMAX][NDIM]);
void move_runge_kutta(int N, double mass[NMAX], double r[NMAX][NDIM], 
		      double v[NMAX][NDIM], int nsteps, double dt);
void move_velocity_verlet(int N, double mass[NMAX], double r[NMAX][NDIM], 
		      double v[NMAX][NDIM], int nsteps, double dt);

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



double getE(double M, double e)
{
    double E = M, eps = 1.0e-7;
    while(fabs(E - e*sin(E) - M) > eps) {
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
        if (e > eps)
	{
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


void center_of_mass(int N, double mass[NMAX], double r[NMAX][NDIM], 
		    double rcm[NDIM]) 
{
  int i, k;
  double mtot;
	
  mtot = 0;
  for (k=0; k<NDIM; k++) {
    rcm[k] = 0;
  }

  for (i=0; i<N; i++) {
    mtot += mass[i];
    for (k=0; k<NDIM; k++) {
      rcm[k] += mass[i]*r[i][k];
    }
  }

  for (k=0; k<NDIM; k++)
    rcm[k] /= mtot;

}


/*
Kinetic energy.  
KE = 1/2 * m * v (dot) v
*/

double kinetic_energy(int N, double mass[NMAX], double v[NMAX][NDIM])
{
  int i, k;
  double ke;

  ke = 0;
  for (i=0; i<N; i++) {
    for (k=0; k<NDIM; k++) {
      ke += mass[i]*v[i][k]*v[i][k];
    }
  }

  ke *= 0.5;

  return ke;
}

/*
Potential energy.  
PE = sum i=1 to n, j=1 to i-1 {-G * m(i)*m(j) / rij}
*/

double potential(int N, double mass[NMAX], double r[NMAX][NDIM])
{
  int i, j, k;
  double pe, rij, r2;

  pe = 0;

  for (i=0; i<N-1; i++) {
    for (j=i+1; j<N; j++) {

      r2 = 0;
      for (k=0; k<NDIM; k++) {
	r2 += (r[i][k]-r[j][k])*(r[i][k]-r[j][k]);
      }
      rij = sqrt(r2);
      if (rij < RMIN)
	rij = RMIN;
      pe -= G*mass[i]*mass[j]/rij;
    }
  }

  return pe;
}
 
/*
Second-order Runge-Kutta integration algorithm. 
*/

void move_runge_kutta(int N, double mass[NMAX], double r[NMAX][NDIM], 
		      double v[NMAX][NDIM], int nsteps, double dt)
{
  int i, j, k, n;
  double a[NDIM], rij[NDIM], r1, r3;
  double deltav[NMAX][NDIM], rmid[NMAX][NDIM], vmid[NMAX][NDIM];

  for (n=0; n<nsteps; n++) {	

    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	   deltav[i][k] = 0;
      }
    }

    /* double loops over objects to get accelarations and the resulting
       change of velocity */
    for (i=0; i<N-1; i++) {
      for (j=i+1; j<N; j++) {

	   /* calculate separation of i & j */
	   r1 = 0;
	   for (k=0; k<NDIM; k++) {
	     rij[k] = r[i][k] - r[j][k];
	     r1 += rij[k]*rij[k];
	   }
	   r1 = sqrt(r1);

	   /* no force if they are very close */
	   if (r1 < RMIN) {
	     for (k=0; k<NDIM; k++) {
	       a[k] = 0;
	     }
	   }
	   else {
	     r3 = pow(r1, 3.0);
	     for (k=0; k<NDIM; k++) {
	       a[k] = -G *rij[k] / r3;
	     }
	   }

	   /* change in velocity from the acceleration, making use of  Newton's 3rd law */
	   for (k=0; k<NDIM; k++) {
	     deltav[i][k] += mass[j]*a[k]*dt;
	     deltav[j][k] -= mass[i]*a[k]*dt;
        } 
      }
    }
    
    /* update the positions and velocities to the midpoint*/
    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	   rmid[i][k] = r[i][k] + 0.5*v[i][k]*dt;
	   vmid[i][k] = v[i][k] + 0.5*deltav[i][k];
      }
    }

    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	   deltav[i][k] = 0;
      }
    }

    /* double loops over objects to get accelarations and the resulting change of velocity */
    for (i=0; i<N-1; i++) {
      for (j=i+1; j<N; j++) {

	   /* calculate separation of i & j */
	   r1 = 0;
	   for (k=0; k<NDIM; k++) {
	     rij[k] = rmid[i][k] - rmid[j][k];
	     r1 += rij[k]*rij[k];
	   }
	   r1 = sqrt(r1);

	   /* no force if they are very close */
	   if (r1 < RMIN) {
	     for (k=0; k<NDIM; k++) {
	       a[k] = 0;
	     }
	   }
	   else {
	     r3 = pow(r1, 3.0);
	     for (k=0; k<NDIM; k++) {
	       a[k] = -G *rij[k] / r3;
	     }
	   }

	   /* change in velocity from the acceleration, making use of  Newton's 3rd law */
	   for (k=0; k<NDIM; k++) {
	     deltav[i][k] += mass[j]*a[k]*dt;
	     deltav[j][k] -= mass[i]*a[k]*dt;
	   }
      }
    }
    
    /* finally update the positions and velocities */
    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	   r[i][k] += vmid[i][k]*dt;
	   v[i][k] += deltav[i][k];
      }
    }
  }
}



void move_velocity_verlet(int N, double mass[NMAX], double r[NMAX][NDIM], 
			  double v[NMAX][NDIM], int nsteps, double dt)
{
  int i, j, k, n;
  double rij[NDIM], r1, r3;
  double f[NMAX][NDIM], dt2;

  dt2 = dt*dt;

  for (n=0; n<nsteps; n++) {	

    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	f[i][k] = 0;
      }
    }

    /* double loops over objects to get accelarations and the resulting
       change of velocity */
    for (i=0; i<N-1; i++) {
      for (j=i+1; j<N; j++) {

	/* calculate separation of i & j */
	r1 = 0;
	for (k=0; k<NDIM; k++) {
	  rij[k] = r[i][k] - r[j][k];
	  r1 += rij[k]*rij[k];
	}
	r1 = sqrt(r1);
	
	/* no force if they are very close */
	if (r1 > RMIN) {
	  r3 = pow(r1, 3.0);
	  for (k=0; k<NDIM; k++) {
	    f[i][k] += -G * mass[i]*mass[j]*rij[k] / r3;
	    f[j][k] +=  G * mass[i]*mass[j]*rij[k] / r3;
	  }
	}
      }
    }

    /* update the positions and velocities */
    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	r[i][k] += v[i][k]*dt + 0.5/mass[i]*f[i][k]*dt2;
	v[i][k] += 0.5/mass[i]*f[i][k]*dt;
      }
    }

    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	f[i][k] = 0;
      }
    }

    /* double loops over objects to get accelarations and the resulting
       change of velocity */
    for (i=0; i<N-1; i++) {
      for (j=i+1; j<N; j++) {

	/* calculate separation of i & j */
	r1 = 0;
	for (k=0; k<NDIM; k++) {
	  rij[k] = r[i][k] - r[j][k];
	  r1 += rij[k]*rij[k];
	}
	r1 = sqrt(r1);

	/* no force if they are very close */
	if (r1 > RMIN) {
	  r3 = pow(r1, 3.0);
	  for (k=0; k<NDIM; k++) {
	    f[i][k] += -G * mass[i]*mass[j]*rij[k] / r3;
	    f[j][k] +=  G * mass[i]*mass[j]*rij[k] / r3;
	  }
	}
      }
    }
    /* finally update the velocities */
    for (i=0; i<N; i++) {
      for (k=0; k<NDIM; k++) {
	v[i][k] += 0.5/mass[i]*f[i][k]*dt;
      }
    }
    
  }
}


int nbody_rk(double *r_arr, double *v_arr, double *mass, double dt, int N, int Nsteps, int downsample, int which) {
  int i, k, t1, Nps;
  double KE, PE;
  double r[NMAX][NDIM], v[NMAX][NDIM];
  double rcm[NDIM], vcm[NDIM];
  Nps = Nsteps/downsample;
  printf("\nRescaling positions and velocities so that COM is fixed at origin\n");
  for (i=0;i<N;i++) {
    for (k=0;k<NDIM;k++) {
      r[i][k] = r_arr[i*NDIM*Nps + k*Nps];
      v[i][k] = v_arr[i*NDIM*Nps + k*Nps];
    }
  }
  center_of_mass(N, mass, r, rcm);
  center_of_mass(N, mass, v, vcm);
  for (i=0;i<N;i++) {
    for (k=0;k<NDIM;k++) {
      printf("%d %lf %lf", k, rcm[k], vcm[k]);
      r[i][k] -= rcm[k];
      v[i][k] -= vcm[k];
    }
  }
  printf("\nINITIAL CONDITIONS:\n");
  
  center_of_mass(N, mass, r, rcm);
  center_of_mass(N, mass, v, vcm);

  printf("\tCOM position: %lf %lf %lf \n", rcm[0], rcm[1], rcm[2]);
  printf("\tCOM velocity: %lf %lf %lf \n", vcm[0], vcm[1], vcm[2]);
  
  KE = kinetic_energy(N, mass, v);
  PE = potential(N, mass, r);
  printf("\tEnergies:\n\tKinetic: %lf\n\tPotential: %lf\n\tTotal: %lf\n",
	 KE, PE, KE+PE); 
  for (t1=0; t1<Nps; t1++) {
      for (i=0; i<N; i++) {
        for (k=0;k<NDIM;k++) {
          r_arr[i*NDIM*Nps + k*Nps + t1] = r[i][k];
          v_arr[i*NDIM*Nps + k*Nps + t1] = v[i][k];
        }
      }
      if (which == 0) {
        move_runge_kutta(N, mass, r, v, downsample, dt);
      }
      else {
        move_velocity_verlet(N, mass, r, v, downsample, dt);
      }
  }

  printf("\nFINAL STATE:\n");
  center_of_mass(N, mass, r, rcm);
  center_of_mass(N, mass, v, vcm);

  printf("\tCOM position: %lf %lf %lf\n", rcm[0], rcm[1], rcm[2]);
  printf("\tCOM velocity: %lf %lf %lf\n", vcm[0], vcm[1], vcm[2]);
  KE = kinetic_energy(N, mass, v);
  PE = potential(N, mass, r);
  printf("\tEnergies:\n\tKinetic: %lf\n\tPotential: %lf\n\tTotal: %lf\n",
	 KE, PE, KE+PE);

  /*free(rcm);
  free(vcm);*/
  return 0;
}

