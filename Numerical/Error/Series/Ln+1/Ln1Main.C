#include <iostream>
#include <iomanip>
#include "Ln1.h"

using namespace std;

/*
    Arguments:
        1) Number of iterations (int)
        2) Number to evaluate (float)
*/

int main(int argc, char *argv[]){

     if(argc < 3){
        cerr << "Number of iterations or value missing" << endl;
        return 1;
    }

    int iterations = atoi(argv[1]);
    float x = atof(argv[2]);

    cout << "Starting taylor serie for LN(x+1) in " << x << endl;

    double result;

    for(int n=1;n<iterations;n++){
        result += ln1(n,x);
        cout << setprecision(14) << result << endl;
    }

    return 0;
}
