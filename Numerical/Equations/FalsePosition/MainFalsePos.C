#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "FalsePos.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1],"-h");

    if(help == 0){
        cout << "MainFalsePos -h | x_i x_s toler niter err_type" << endl;
        cout << "    x_i = punto menor del intervalo (double)" << endl;
        cout << "    x_s = punto mayor del intervalo (double)" << endl;
        cout << "    toler = tolerancia (double)" << endl;
        cout << "    niter = numero de iteraciones (int)" << endl;
	cout << "    err_type = type of the error: abs | rel (string)" << endl;
        return 0;
    }

    double x_i = atof(argv[1]);
    double x_s = atof(argv[2]);
    double toler = atof(argv[3]);
    int niter = atoi(argv[4]);
    bool err_type;
    if(strcmp(argv[5], "abs") == 0) err_type = true;
    else if(strcmp(argv[5], "rel") == 0) err_type = false;
    else cout << "ERROR: Error type argument must be either: abs or rel" << endl;

    if(toler < 0){
        cerr << "ERROR: the tolerance can't be negative" << endl;
        return 2;
    }
    else if(niter < 1){
        cerr << "ERROR: the iterations must be greater than 0" << endl;
        return 3;
    }

    falsePos(x_i, x_s, toler, niter, err_type);

    return 0;
}
