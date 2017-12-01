#include <iostream>
#include <string.h>

#include "Reader.h"

#include "Jacobi.h"

using namespace std;

int main(int argc, char *argv[]){
    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1],"-h");

    if(help == 0){
        cout << "MainJacobi -h | n toler niter mfile vfile" << endl;
        cout << "\tn = size of the matrix (int)" << endl;
        cout << "\ttoler = method tolerance (double)" << endl;
        cout << "\tniter = number of iterations (int)" << endl;
        cout << "\tmfile = name of the matrix file (string)" << endl;
        cout << "\tvfile = name of the initial gues file (string)" << endl;

        return 0;
    }

    cout << endl << "Matrix size is going to be set to " << argv[1] << endl;
    cout << "Method tolerance is going to be set to " << argv[2] << endl;
    cout << "Method iterations is going to be set to " << argv[3] << endl;

    int n = atoi(argv[1]);
    double toler = atof(argv[2]);
    int niter = atoi(argv[3]);

    cout << "done" << endl;

    double ** A = readMatrix(n, argv[4]);
    double * x = readVector(n, argv[5]);

    jacobi(A, x, toler, niter, n);

    return 0;
}
