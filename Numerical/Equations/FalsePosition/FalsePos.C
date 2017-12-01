#include <iostream>
#include <iomanip>
#include <cmath>

#include "Function.h"
#include "SlopeSpecial.h"
#include "FalsePos.h"

using namespace std;

void falsePos(double x_i, double x_s, double toler, int niter, bool err_type){

    cout << setprecision(14) <<
            setw(4) << "iter" <<
            setw(23) << "x_i" <<
            setw(23) << "x_s" <<
            setw(23) << "x_m" <<
            setw(23) << "error" << endl;

    double fx_i, fx_s;

    fx_i = function(x_i);
    fx_s = function(x_s);

    if(fx_i == 0) cout << endl << "x_i is a root with value = " << x_i << endl;
    else if(fx_s == 0) cout << endl << "x_s is a root with value = " << x_s << endl;
    else if(fx_i * fx_s < 0){

        int cnt;
        double x_m, fx_m, aux, error;

        x_m = calcPoint(x_i, x_s, fx_i, fx_s);
        fx_m = function(x_m);
        error = toler+1;

        cnt = 1;

        cout << setprecision(14) <<
                setw(4) << 1 <<
                setw(23) << x_i <<
                setw(23) << x_s <<
                setw(23) << x_m <<
                setw(23) << "N/A" << endl;

        while(error > toler && fx_m != 0 && cnt++ < niter){
            if(fx_i * fx_m < 0){
                x_s = x_m;
                fx_s = fx_m;
            }
            else{
                x_i = x_m;
                fx_i = fx_m;
            }

            aux = x_m;
            x_m = calcPoint(x_i, x_s, fx_i, fx_s);
            fx_m = function(x_m);
    	    if(err_type == true) error = abs(x_m - aux);
    	    else error = abs((x_m - aux)/x_m);

            cout << setprecision(14) <<
                    setw(4) << cnt <<
                    setw(23) << x_i <<
                    setw(23) << x_s <<
                    setw(23) << x_m <<
                    setw(23) << error << endl;

        }
        if(fx_m == 0){
           cout << endl <<
                   setprecision(14) <<
                   "x_m = " << x_m << " is a root" << endl;
        }
        else if(error < toler){
           cout << endl <<
                   setprecision(14) <<
                   "x_m = " << x_m << " is a root" <<
                   ", error = " << toler << endl;
        }
        else{
           cerr << endl <<
                   "Can't find any root with " << niter << " iterations" << endl;
        }
    }
    else{
        cerr << endl << "Wrong interval" << endl;
    }
}
