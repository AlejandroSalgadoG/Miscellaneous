#include <iostream>
#include <iomanip>

#include "Jacobi.h"

#include "Reader.h"
#include "Norm.h"

using namespace std;

double* calcNewVector(double ** A, double * x, double * x_n, int n){
    double cnt;
    for(int i=0;i<n;i++){
        cnt = 0;
        for(int j=0;j<n;j++){
            if(j == i) continue;
            else cnt += A[i][j] * x[j];
        }
        x_n[i] = (A[i][n] - cnt) / A[i][i];
    }

    return x_n;
}

double* jacobi(double ** A, double * x, double toler, int niter, int n){

    cout << "Initial x = ";
    printVector(x, n);
    cout << endl;

    cout << setprecision(14) <<
            setw(4) << "iter" <<
            setw(23) << "x" <<
            setw(23) << "error" << endl;

    int cnt = 0;
    double error = toler + 1;
    double * x_n = new double[n];

    while(error > toler && cnt++ < niter){
        x_n = calcNewVector(A, x, x_n, n);
        error = maxNorm(x, x_n, n);

        cout << setprecision(14) <<
                setw(4) << cnt <<
                setw(23) << x_n[0] << endl;

        for(int i=1;i<n-1;i++)
            cout << setprecision(14) << setw(27) << x_n[i] << endl;

        cout << setprecision(14) <<
                setw(27) << x_n[n-1] <<
                setw(23) << error << endl;

        cout << "--------------------------------------------------" << endl;

        for(int i=0;i<n;i++)
            x[i] = x_n[i];
    }

    if(error < toler){
        cout << endl << "x = ";
        printVector(x_n, n);
        cout << "is a root" <<
                ", error = " << toler << endl;
    }
    else{
        cerr << endl <<
                "Can't find any solution with " << niter << " iterations" << endl;
    }

    delete x_n;

}
