#include "Factorial.h"

int factorial(int x){
    int ans=x;
    x--;
    for(;x>1;x--) ans = x * ans;

    if(x < 1) return 1;
    else return ans;
}
