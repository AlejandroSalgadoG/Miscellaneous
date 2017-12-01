#pragma once

double** allocMatrix(int n);
double** allocMatrix(int row, int col);
double* allocVector(int n);
double* readVector(int n, char * file);
double** readMatrix(int n, char * file);
double* readMatrixAsVector(int n, char * file);
void printMatrix(double ** A, int n);
void printMatrix(double ** A, int row, int col);
void printMatrix(int * C, int n);
void printVector(double * A, int n);
