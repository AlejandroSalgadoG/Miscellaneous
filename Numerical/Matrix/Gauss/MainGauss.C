#include <iostream>
#include <string.h>

#include "Reader.h"

#include "Gauss.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1],"-h");

    if(help == 0){
        cout << "MainGauss -h | n file" << endl;
        cout << "\tn = size of the matrix (int)" << endl;
        cout << "\tfile = name of the matrix file (string)" << endl;

        return 0;
    }

    cout << endl << "Size of the A matrix has been set to " << argv[1] << endl;
    int n = atoi(argv[1]);

    double ** A = readMatrix(n, argv[2]);

    double * ans = gauss(A,n);

    cout << "The answers to the system are" << endl << endl;
    for(int i=0;i<n;i++)
        cout << "\tx_" << i << " = " << ans[i] << endl;

    return 0;
}
