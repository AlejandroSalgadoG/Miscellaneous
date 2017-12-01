#include <math.h>
#include "Tan_1.h"

/*
    Formula:

        -1^n+1 * x^2n-1
                 ------
                  2n-1
    Taken from http://mathworld.wolfram.com/MaclaurinSeries.html
*/

double tan_1(int n, float x){

    int factor = pow(-1,n+1);
    float numerator = pow(x,2*n-1);
    int denominator = 2*n-1;

    return factor * (numerator / denominator);
}
