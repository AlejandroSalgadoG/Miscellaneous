#include <iostream>
#include <stdlib.h>
#include <string.h>

#include "VectorGen.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1], "-h");

    if(help == 0){
        cout << "MainVectorGen -h | file n mode val" << endl;
        cout << "\tfile = name of the vector file (string)" << endl;
        cout << "\tn = size of the vector (int)" << endl;
        cout << "\tmode = either stat, rand (string)" << endl;
        cout << "\tval = key value to create the vector (double)" << endl;

        return 0;
    }

    cout << endl << "Name of the file has been set to " << argv[1] << endl;

    cout << "Size of the x vector has been set to " << argv[2] << endl;
    int n = atoi(argv[2]);

    cout << "Mode has been set to " << argv[3] << endl << endl;

    cout << "Main value has been set to " << argv[4] << endl << endl;
    double val = atof(argv[4]);

    bool vec;
    double vector;

    double * x;

    if(strcmp(argv[3], "stat") == 0) x = fillVectorStatic(n, val);
    else if(strcmp(argv[3], "rand") == 0) x = fillVectorRandom(n, val);
    else{ cerr << "ERROR: mode not recognized" << endl; return 1; }

    writeVector(x,n,argv[1]);

    delete x;

    return 0;
}
