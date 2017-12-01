#include "stdio.h"
#include "omp.h"

using namespace std;

int main(int argc, char *argv[]){

    #pragma omp parallel
    {
        printf("Thread %d\n", omp_get_thread_num() );
    }

}
