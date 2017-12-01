#include <math.h>
#include "Factorial.h"

/*
    Formula:

        x^n
        ---
        n!

    Taken from http://mathworld.wolfram.com/MaclaurinSeries.html
*/

double expo(int n, float x){

    float numerator = pow(x,n);
    int denominator = factorial(n);

    return numerator / denominator;
}
