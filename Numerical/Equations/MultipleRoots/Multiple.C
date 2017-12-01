#include <iostream>
#include <iomanip>
#include <cmath>

#include "Function.h"
#include "SlopeSpecial.h"
#include "Multiple.h"

using namespace std;

void multiple(double x, double toler, int niter, bool err_type){

    cout << setprecision(14) <<
                setw(4) << "iter" <<
                setw(23) << "x" <<
                setw(23) << "fx" <<
                setw(23) << "d_fx" <<
                setw(23) << "dd_fx" <<
                setw(23) << "error" << endl;

    double x_n, fx, d_fx, dd_fx, error;

    fx = function(x);
    d_fx = derivative(x);
    dd_fx = secDerivative(x);
    error = toler+1;

    cout << setprecision(14) <<
            setw(4) << 1 <<
            setw(23) << x <<
            setw(23) << fx <<
            setw(23) << d_fx <<
            setw(23) << dd_fx <<
            setw(23) << "N/A" << endl;

    int cnt = 1;

    if(fx == 0) cout << endl << "x = " << x << "is a root" << endl;
    else{
        while(error > toler && fx != 0 && cnt++ < niter){

            x_n = calcPointMul(x, fx, d_fx, dd_fx);
            fx = function(x_n);
            d_fx = derivative(x_n);
            dd_fx = secDerivative(x_n);
            if(err_type == true) error = abs(x_n - x);
	    else error = abs((x_n - x)/x_n);
            x = x_n;

            cout << setprecision(14) <<
                    setw(4) << cnt <<
                    setw(23) << x <<
                    setw(23) << fx <<
                    setw(23) << d_fx <<
                    setw(23) << dd_fx <<
                    setw(23) << error << endl;
        }

        if(fx == 0){
            cout << endl <<
                    setprecision(14) <<
                    "x = " << x << " is a root" << endl;
        }
        else if(error < toler){
            cout << endl <<
                    setprecision(14) <<
                    "x = " << x << " is a root" <<
                    ", error = " << toler << endl;
        }
        else{
           cout << endl <<
                   "Can't find any root with " << niter << " iterations" << endl;
        }
    }
}
