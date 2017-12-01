#pragma once

double* calcNewVector(double ** A, double * x, double * x_n, int n);
double norm(double * x, double * x_n, int n);
double* jacobi(double ** A, double * x, double toler, int niter, int n);
