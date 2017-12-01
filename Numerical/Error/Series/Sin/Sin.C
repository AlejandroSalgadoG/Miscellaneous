#include <math.h>
#include "Factorial.h"
#include "Sin.h"

/*
    Formula:

        -1^n * x^2n+1
               ------
               (2n+1)!

    Taken from http://mathworld.wolfram.com/MaclaurinSeries.html
*/

double sin(int n, float x){

    int factor = pow(-1,n);
    float numerator = pow(x,2*n+1);
    int denominator = factorial(2*n+1);

    return factor * (numerator / denominator);
}
