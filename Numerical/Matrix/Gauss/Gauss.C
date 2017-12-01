#include "Gauss.h"

double* gauss(double ** A, int n){

    double ** U = elimination(A,n);

    return substitution(U, n);
}

double** elimination(double ** A, int n){

    for(int k=0;k<n-1;k++){
        for(int i=k+1;i<n;i++){
            double m = A[i][k]/A[k][k]; // DIVISION BY 0
            for(int j=k+1;j<n+1;j++)
                A[i][j] = A[i][j] - m*A[k][j];
        }
    }

    return A;
}

double* substitution(double ** U, int n){
    double * ans = new double[n];

    ans[0] = 1;
    ans[1] = 2;
    ans[2] = 3;

    double acum;
    for(int i=n-1;i>=0;i--){
        acum = 0;
        for(int h=i+1;h<n;h++)
            acum += U[i][h] * ans[h];
        ans[i] = (U[i][n] - acum)/U[i][i];
    }

    return ans;
}
