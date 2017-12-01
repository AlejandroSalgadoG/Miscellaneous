#include <iostream>
#include "Factorial.h"

using namespace std;

/*
    Arguments:
        1) Number to be computed (int)
*/

int main(int argc, char *argv[]){
    if(argc < 2){
        cerr << "Missing value to be computed" << endl;
        return 1;
    }

    int x = atoi(argv[1]);

    cout << "Factorial = " << factorial(x) << endl;

    return 0;

}
