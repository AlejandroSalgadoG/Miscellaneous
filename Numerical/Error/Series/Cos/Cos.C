#include <math.h>
#include "Factorial.h"
#include "Cos.h"

/*
    Formula:

    (-1)^n * x^2n
             ----
             (2n)!

    Taken from http://mathworld.wolfram.com/MaclaurinSeries.html
*/

double cos(int n, float x){

    int factor = pow(-1,n);
    float numerator = pow(x,2*n);
    int denominator = factorial(2*n);

    return factor * (numerator / denominator);
}
