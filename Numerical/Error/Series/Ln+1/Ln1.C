#include <math.h>
#include "Ln1.h"

/*
    formula:

        -1^n+1 * (x)^n
                 -------
                    n
    Taken from http://mathworld.wolfram.com/MaclaurinSeries.html
*/

double ln1(int n, float x){

    int factor = pow(-1,n+1);
    float numerator = pow(x,n);
    int denominator = n;

    return factor * (numerator / denominator);
}
