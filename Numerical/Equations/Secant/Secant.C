#include <iostream>
#include <iomanip>
#include <cmath>

#include "Function.h"
#include "SlopeSpecial.h"
#include "Secant.h"

using namespace std;

void secant(double x_i, double x_s, double toler, int niter, bool err_type){

    cout << setprecision(14) <<
            setw(4) << "iter" <<
            setw(23) << "x" <<
            setw(23) << "fx" <<
            setw(23) << "error" << endl;

    double fx_i, fx_s;

    fx_i = function(x_i);
    fx_s = function(x_s);

    if(fx_i == 0) cout << "x_i is a root with value = " << x_i << endl;
    else if(fx_s == 0) cout << "x_s is a root with value = " << x_s << endl;
    else{

        double x_n, fx_n, error;

        x_n = calcPoint(x_i, x_s, fx_i, fx_s);
        fx_n = function(x_n);
        error = toler+1;

        cout << setprecision(14) <<
                setw(4) << 1 <<
                setw(23) << x_i <<
                setw(23) << fx_i <<
                setw(23) << "N/A" << endl;

        cout << setprecision(14) <<
                setw(4) << 1 <<
                setw(23) << x_s <<
                setw(23) << fx_s <<
                setw(23) << "N/A" << endl;

        int cnt = 1;

        while(error > toler && fx_n != 0 && cnt++ < niter){

            x_i = x_s;
            fx_i = fx_s;

            x_s = x_n;
            fx_s = fx_n;

            x_n = calcPoint(x_i, x_s, fx_i, fx_s);
            fx_n = function(x_n);

            if(err_type == true) error = abs(x_n - x_s);
	    else error = abs((x_n - x_s)/x_n);

            cout << setprecision(14) <<
                    setw(4) << cnt <<
                    setw(23) << x_n <<
                    setw(23) << fx_n <<
                    setw(23) << error << endl;

        }

        if(fx_n == 0){
            cout << endl <<
                    setprecision(14) <<
                    "x = " << x_n << " is a root" << endl;
        }
        else if(error < toler){
            cout << endl <<
                    setprecision(14) <<
                    "x = " << x_n << "is a root" <<
                    ", error = " << toler << endl;
        }
        else{
           cout << endl <<
                   "Can't find any root with " << niter << " iterations" << endl;
        }
    }
}
