#include <iostream>
#include <iomanip>
#include "Expo.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc < 3){
        cerr << "Number of iterations or value missing" << endl;
        return 1;
    }

    int iterations = atoi(argv[1]);
    float x = atof(argv[2]);

    cout << "Starting taylor serie for EXPO in " << x << endl;

    double result;

    for(int n=0;n<iterations;n++){
        result += expo(n,x);
        cout << setprecision(14) << result << endl;
    }

    return 0;
}
