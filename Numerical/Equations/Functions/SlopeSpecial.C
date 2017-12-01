#include <cmath>

#include "SlopeSpecial.h"

double calcPoint(double x_i, double x_s, double fx_i, double fx_s){
    double div = fx_s - fx_i;
    if(div == 0) return 0; // division by 0
    else return x_i-((fx_i * (x_s - x_i)) / div);
}

double calcPointMul(double x, double fx, double d_fx, double dd_fx){
    double div = pow(x,2) - fx * dd_fx;
    if(div == 0) return 0; // division by 0
    else return x - ((fx* d_fx) / div);
}
