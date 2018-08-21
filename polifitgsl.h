#ifndef _POLIFITGSL_H
#define _POLIFITGSL_H
#include <gsl/gsl_multifit.h>
#include <stdbool.h>
#include <math.h>
bool polynomialfit(int obs, int degree, 
		   double *dx, double *dy,
                   double *store);

bool polynomialfit_w(int obs, int degree, 
		   double *dx, double *dy, double *derr,
                   double *store); /* n, p */
#endif
