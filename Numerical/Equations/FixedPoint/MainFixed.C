#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "Fixed.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1], "-h");

    if(help == 0){
        cout << "MainFixed -h | x_0 toler niter err_type" << endl;
        cout << "    x_0 = first root guess (double)" << endl;
        cout << "    toler = tolerance (double)" << endl;
        cout << "    niter = number of iterations (int)" << endl;
	cout << "    err_type = type of the error: abs | rel (string)" << endl;
        return 0;
    }

    double x_0 = atof(argv[1]);
    double toler = atof(argv[2]);
    int niter = atoi(argv[3]);
    bool err_type;
    if(strcmp(argv[4], "abs") == 0) err_type = true;
    else if(strcmp(argv[4], "rel") == 0) err_type = false;
    else cout << "ERROR: Error type argument must be either: abs or rel" << endl;

    if(toler < 0){
        cerr << "ERROR: the tolerance can't be negative" << endl;
        return 2;
    }
    else if(niter < 1){
        cerr << "ERROR: the iterations must be greater than 0" << endl;
        return 3;
    }

    fixedPoint(x_0, toler, niter, err_type);

    return 0;
}
