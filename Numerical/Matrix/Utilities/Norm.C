#include <cmath>
#include <iostream>

using namespace std;

double distNorm(double * x, double * x_n, int n){

    double cnt = 0;
    for(int i=0;i<n;i++)
        cnt += pow(abs(x[i]-x_n[i]), 2);

    return sqrt(cnt);
}

double sumNorm(double * x, double * x_n, int n){
    double cnt = 0;
    for(int i=0;i<n;i++)
        cnt += abs(x[i]-x_n[i]);

    return cnt;
}

double maxNorm(double * x, double * x_n, int n){
    double tmp, max;
    max = abs(x[0] - x_n[0]);

    for(int i=1;i<n;i++){
        tmp = abs(x[i] - x_n[i]);
        if(tmp > max) max = tmp;
    }

    return max;
}
