#include <math.h>
#include "Ln.h"

/*
    formula:

    -1^n+1 * (x-1)^n
             -------
                n

    Taken from http://math.feld.cvut.cz/mt/txte/3/txe3ea3e.htm
*/

double ln(int n, float x){

    int factor = pow(-1,n+1);
    float numerator = pow(x-1,n);
    int denominator = n;

    return factor * (numerator / denominator);
}
