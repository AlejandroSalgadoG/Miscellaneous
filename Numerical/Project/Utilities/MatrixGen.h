#pragma once

double ** fillMatrixStatic(int n, double val);
double ** fillMatrixRandom(int n, double seed);
double ** fillMatrixStaticVec(int n, double val, char * vect);
double ** fillMatrixRandomVec(int n, double seed, char * vect);
void writeMatrix(double ** A, int n, char * file);
