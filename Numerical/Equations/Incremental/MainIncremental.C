#include <iostream>
#include <string.h>
#include <stdlib.h>

#include "Incremental.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1],"-h");

    if(help == 0){
        cout << "MainIncremental -h | x_i delta niter" << endl;
        cout << "    x_i = initial point (double)" << endl;
        cout << "    delta = method increment value (double)" << endl;
        cout << "    niter = number of iterations (int)" << endl;
        return 0;
    }

    double x_i = atof(argv[1]);
    double delta = atof(argv[2]);
    int niter = atoi(argv[3]);

    if(niter < 1){
        cerr << "ERROR: the iterations must be greater than 0" << endl;
        return 3;
    }

    incremental(x_i, delta, niter);

    return 0;
}
