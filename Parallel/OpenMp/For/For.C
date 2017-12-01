#include "stdio.h"
#include "stdlib.h"
#include "omp.h"

using namespace std;

int main(int argc, char *argv[]){

    char * numLetter = getenv("OMP_NUM_THREADS");
    int num = atoi(numLetter);

    #pragma omp parallel for num_threads(num)

        for(int x=0;x<400;x++){
            printf("x = %d Thread = %d\n", x, omp_get_thread_num());
        }
}
