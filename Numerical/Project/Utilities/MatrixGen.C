#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "MatrixGen.h"
#include "Reader.h"

using namespace std;

double ** fillMatrixStatic(int n, double val){

    double ** A = allocMatrix(n);

    cout << "Starting the matrix fill...";

    double acum = 0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n+1;j++)
            acum += A[i][j] = (i+j)*val;

	    A[i][i] = acum;
        acum = 0;
    }

    cout << "Done." << endl;

    return A;
}

double ** fillMatrixRandom(int n, double seed){

    double ** A = allocMatrix(n);

    cout << endl << "Setting random seed to " << seed << "...";
    srand(seed);
    cout << "done" << endl;

    cout << "Starting the matrix fill...";

    double acum = 0;
    for(int i=0;i<n;i++){
       for(int j=0;j<n+1;j++){
            acum += A[i][j] = (rand()%100);
            if(rand()%2 == 0) A[i][j] *= -1;
       }

	    A[i][i] = acum;
        acum = 0;
    }

    cout << "Done." << endl;

    return A;
}

double ** fillMatrixStaticVec(int n, double val, char * vect){

    double ** A = allocMatrix(n);
    double * x = readVector(n, vect);

    cout << "Starting the matrix fill...";

    double diag_acum = 0;
    double b_acum = 0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            diag_acum += A[i][j] = (i+j)*val;
	    A[i][i] = diag_acum;

        for(int j=0;j<n;j++)
            b_acum += A[i][j] * x[j];
        A[i][n] = b_acum;

        diag_acum = 0;
        b_acum = 0;
    }

    cout << "Done." << endl;

    return A;

}

double ** fillMatrixRandomVec(int n, double seed, char * vect){

    double ** A = allocMatrix(n);
    double * x = readVector(n, vect);

    cout << endl << "Setting random seed to " << seed << "...";
    srand(seed);
    cout << "done" << endl;

    cout << "Starting the matrix fill...";

    double diag_acum = 0;
    double b_acum = 0;
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            diag_acum += A[i][j] = (i+j)*(rand()%10);
            if(rand()%2 == 0) A[i][j] *= -1;
        }
	    A[i][i] = diag_acum;

        for(int j=0;j<n;j++)
            b_acum += A[i][j] * x[j];
        A[i][n] = b_acum;

        diag_acum = 0;
        b_acum = 0;
    }

    cout << "Done." << endl;

    return A;

}

void writeMatrix(double ** A, int n, char * file){

    ofstream matrix;
    matrix.open(file, ios::out);

    cout << "Starting to write the file...";

    for(int i=0;i<n;i++){
        for(int j=0;j<n+1;j++){
            if(j==n)
                matrix << A[i][j] << endl;
            else
                matrix << A[i][j] << " ";
        }
    }

    matrix.close();

    cout << "Done." << endl;
}
