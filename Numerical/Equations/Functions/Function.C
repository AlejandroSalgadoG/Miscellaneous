#include <cmath>

#include "Function.h"

double function(double x){
    return log(pow(x,2) + 1) + x*cos(6*x + 3) - 3*x - 10;
}

double derivative(double x){
    return ((2*x) / (1 + pow(x,2))) + cos(3 + 6*x) - 6*x*sin(3 + 6*x)  - 3;
}

double secDerivative(double x){
    double div = (2 / (1 + pow(x,2))) - ((4*pow(x,2)) / pow((1 + pow(x,2)), 2));
    return div - 36*x*cos(3 + 6*x) - 12*sin(3 + 6*x);
}
