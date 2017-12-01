#include "Initializer.h"

void initializeMatrix(int * A, int * B , int n){
    for(int cnt=0, i=0;i<n;i++)
        for(int j=0;j<n;j++)
            B[n*i+j] = A[n*i+j] = cnt++;
}
