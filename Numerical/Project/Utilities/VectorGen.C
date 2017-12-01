#include <iostream>
#include <fstream>
#include <stdlib.h>

#include "VectorGen.h"
#include "Reader.h"

using namespace std;

double* fillVectorStatic(int n, double val){

    double * x = allocVector(n);

    cout << "Starting the vector fill...";

    for(int i=0;i<n;i++)
        x[i] = val;

    cout << "Done." << endl;

    return x;
}

double* fillVectorRandom(int n, double seed){

    double * x = allocVector(n);

    cout << endl << "Setting random seed to " << seed << "...";
    srand(seed);
    cout << "done" << endl;

    cout << "Starting the vector fill...";

    for(int i=0;i<n;i++)
        x[i] = rand()%10;

    cout << "Done." << endl;

    return x;
}

void writeVector(double * x, int n, char * file){

    ofstream matrix;
    matrix.open(file, ios::out);

    cout << "Starting to write the file...";

    for(int i=0;i<n;i++)
        matrix << x[i] << endl;

    matrix.close();

    cout << "Done." << endl;
}
