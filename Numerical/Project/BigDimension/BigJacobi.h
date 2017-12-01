#pragma once

void initialize_bigjacobi(int n, int filed, int pagesz, long int mapcnt,
                          long int memsz, long int size);

void acum_partial(double * Ab, double * x, int n, int st, int pos, int row, int sz);
int acum_row(double * Ab, double * x, int n, int row_st, int row_rd, int row_cnt, int part);

double* calcNewVector(double * x_n, int n);
void bigjacobi(double * Ab, double * x, int n, double toler, int niter);
