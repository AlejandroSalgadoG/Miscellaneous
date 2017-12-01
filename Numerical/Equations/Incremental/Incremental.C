#include <iostream>
#include <iomanip>

#include "Function.h"
#include "Incremental.h"

using namespace std;

void incremental(double x_i, double delta, int niter){

    cout << endl << "Delta = " << delta << endl << endl;

    cout << setprecision(14) <<
            setw(4) << "iter" <<
            setw(23) << "x_i" <<
            setw(23) << "x_s" <<
            setw(23) << "fx_i" <<
            setw(23) << "fx_s" << endl;

    double x_s, fx_i, fx_s;

    x_s = x_i + delta;
    fx_i = function(x_i);
    fx_s = function(x_s);

    if(fx_i == 0) cout << endl << "x is a root with value = " << x_i << endl;
    else if(fx_s == 0) cout << endl << "x is a root with value = " << x_s << endl;
    else{

        cout << setprecision(14) <<
                setw(4) << 1 <<
                setw(23) << x_i <<
                setw(23) << x_s <<
                setw(23) << fx_i <<
                setw(23) << fx_s << endl;

        int cnt = 1;

        while(fx_i * fx_s >= 0 && cnt++ < niter){

            x_i = x_s;
            fx_i = fx_s;
            x_s = x_i + delta;
            fx_s = function(x_s);

            cout << setprecision(14) <<
                    setw(4) << cnt <<
                    setw(23) << x_i <<
                    setw(23) << x_s <<
                    setw(23) << fx_i <<
                    setw(23) << fx_s << endl;
        }

        if(fx_i == 0){
            cout << endl <<
                    setprecision(14) <<
                    "x = " << x_i << " is a root" << endl;
        }
        if(fx_s == 0){
            cout << endl <<
                    setprecision(14) <<
                    "x = " << x_s << " is a root" << endl;
        }
        else if(fx_i * fx_s < 0){
            cout << endl <<
                    setprecision(14) <<
                    "At least one root exists between "<< x_i <<
                    " and " << x_s << endl;
        }
        else{
           cerr << endl <<
                   "Can't find any root with " << niter << " iterations" << endl;

        }
    }
}
