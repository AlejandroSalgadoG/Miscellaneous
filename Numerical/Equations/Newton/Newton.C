#include <iostream>
#include <iomanip>
#include <cmath>

#include "Newton.h"
#include "Function.h"

using namespace std;

void newton(double x, double toler, int niter, bool err_type){

    cout << setprecision(14) <<
                setw(4) << "iter" <<
                setw(23) << "x" <<
                setw(23) << "fx" <<
                setw(23) << "d_fx" <<
                setw(23) << "error" << endl;

    double x_n, fx, d_fx, error;

    fx = function(x);
    d_fx = derivative(x);
    error = toler+1;

    cout << setprecision(14) <<
            setw(4) << 1 <<
            setw(23) << x <<
            setw(23) << fx <<
            setw(23) << d_fx <<
            setw(23) << "N/A" << endl;

    int cnt = 1;

    if(fx == 0) cout << "x is a root with value = " << x << endl;
    else{
        while(error > toler && fx != 0 && d_fx != 0 && cnt++ < niter){

            x_n = x - (fx / d_fx);
            fx = function(x_n);
	    d_fx = derivative(x_n);

            if(err_type == true) error = abs(x_n - x);
	    else error = abs((x_n - x)/x_n);

	    x = x_n;

            cout << setprecision(14) <<
                    setw(4) << cnt <<
                    setw(23) << x <<
                    setw(23) << fx <<
                    setw(23) << d_fx <<
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
        else if(d_fx == 0){
            cerr << endl <<
                    "its possible that x is a multiple root" << endl;
        }
        else{
           cerr << endl <<
                   "Can't find any root with " << niter << " iterations" << endl;
        }
    }
}
