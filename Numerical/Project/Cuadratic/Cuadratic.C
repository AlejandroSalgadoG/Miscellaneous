#include "Cuadratic.h"

solution cuadratic_pos(solution sol, variables vars){

    double a = vars.a;
    double b = vars.b;
    double c = vars.c;

    double internal = pow(b,2) - 4*a*c;
    sol.pos = (-b + sqrt(internal))/(2*a);

    return sol;
}

solution cuadratic_neg(solution sol, variables vars){

    double a = vars.a;
    double b = vars.b;
    double c = vars.c;

    double internal = pow(b,2) - 4*a*c;
    sol.neg = (-b - sqrt(internal))/(2*a);

    return sol;
}
