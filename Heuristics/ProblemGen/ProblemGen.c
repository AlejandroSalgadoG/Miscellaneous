#include <stdio.h>
#include <stdlib.h>

//This code is based on
//Taillard E. Benchmarks for basic scheduling problems. European Journal of
//Operational Research 1993;64:278--85.

int unif(int* seed, int low, int high){
    const int m = 2147483647;
    const int a = 16807;
    const int b = 127773;
    const int c = 2836;

    int k = *seed / b;
    *seed = a*(*seed % b) - k*c;

    if(*seed < 0) *seed += m;

    double value_0_1 = (double) *seed / m;
    return low + value_0_1 * (high - low + 1);
}

int* generate_flow_shop(int* time_seed, int nb_jobs, int nb_machines, int* d){
    for(int i=0;i<nb_machines;i++)
        for(int j=0;j<nb_jobs;j++)
            d[i*nb_jobs + j] = unif(time_seed, 1, 99);
    return d;
}

void print(int* d, int nb_jobs, int nb_machines){
    for(int i=0;i<nb_machines;i++){
        for(int j=0;j<nb_jobs;j++)
            printf("%d ", d[i*nb_jobs + j]);
        printf("\n");
    }
}

int main(int argc, char *argv[]){
    int time_seed = atoi(argv[1]);
    int nb_jobs = atoi(argv[2]);
    int nb_machines = atoi(argv[3]);

    int* d = malloc(nb_machines * nb_jobs * sizeof(int));
    d = generate_flow_shop(&time_seed, nb_jobs, nb_machines, d);
    print(d, nb_jobs, nb_machines);

    return 0;
}
