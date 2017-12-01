#include <cmath>

#include "Gfunction.h"

double gFunction(double x){
    return log(pow(x,2) + 1) + x * cos(6*x +3) - 2*x - 10;
}
