#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "MatrixGen.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1], "-h");

    if(help == 0){
        cout << "MainMatrixGen -h | file n mode val [vec]" << endl;
        cout << "\tfile = name of the matrix file (string)" << endl;
        cout << "\tn = size of the matrix (int)" << endl;
        cout << "\tmode = either stat, rand (string)" << endl;
        cout << "\tval = key value to create the matrix (double)" << endl;
        cout << "\tvec = vector to create the matrix (double)" << endl;

        return 0;
    }

    cout << endl << "Name of the file has been set to " << argv[1] << endl;

    cout << "Size of the A matrix has been set to " << argv[2] << endl;
    int n = atoi(argv[2]);

    cout << "Mode has been set to " << argv[3] << endl << endl;

    cout << "Main value has been set to " << argv[4] << endl << endl;
    double val = atof(argv[4]);

    bool vec;
    double vector;

    if(argv[5]) vec = true;
    else vec = false;

    double ** A;

    if(vec){
        if(strcmp(argv[3], "stat") == 0) A = fillMatrixStaticVec(n, val, argv[5]);
        else if(strcmp(argv[3], "rand") == 0) A = fillMatrixRandomVec(n, val, argv[5]);
        else{ cerr << "ERROR: mode not recognized" << endl; return 1; }
    }
    else{
        if(strcmp(argv[3], "stat") == 0) A = fillMatrixStatic(n, val);
        else if(strcmp(argv[3], "rand") == 0) A = fillMatrixRandom(n, val);
        else{ cerr << "ERROR: mode not recognized" << endl; return 1; }
    }

    writeMatrix(A,n,argv[1]);

    delete A;

    return 0;
}
