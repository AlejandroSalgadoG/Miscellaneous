#include <iostream>
#include <iomanip>
#include <cmath>

#include "Function.h"
#include "Gfunction.h"
#include "Fixed.h"

using namespace std;

void fixedPoint(double x, double toler, int niter, bool err_type){

    cout << endl << "Initial x = " << x << endl << endl;

    cout << setprecision(14) <<
            setw(4) << "iter" <<
            setw(23) << "g(x)" <<
            setw(23) << "f(x)" <<
            setw(23) << "error" << endl;

    double fx, gx, error;

    fx = function(x);
    gx = gFunction(x);
    error = toler+1;

    cout << setprecision(14) <<
            setw(4) << 1 <<
            setw(23) << gx <<
            setw(23) << fx <<
            setw(23) << error << endl;

    int cnt = 1;

    while(gx != x && error > toler && cnt++ < niter){

        x = gx;
        fx = function(x);
        gx = gFunction(x);
        if(err_type == true) error = abs(gx - x);
	    else error = abs((gx - x)/gx);

        cout << setprecision(14) <<
                setw(4) << cnt <<
                setw(23) << gx <<
                setw(23) << fx <<
                setw(23) << error << endl;
    }

    if(gx == x){
        cout << endl <<
                setprecision(14) <<
                "x = " << gx << " is a root" << endl;
    }
    else if(error < toler){
        cout << endl <<
                setprecision(14) <<
                "x = " << gx << " is a root" <<
                ", error = " << toler << endl;
    }
    else{
       cerr << endl <<
               "Can't find any root with " << niter << " iterations" << endl;
    }
}
