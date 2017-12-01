#include <iostream>
#include <mpi.h>
#include "Cuadratic.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc < 4){ 
        cout << "Not enough arguments" << endl;
        exit(1);
    }

    solution sol;
    variables vars;

    vars.a = atof(argv[1]);
    vars.b = atof(argv[2]);
    vars.c = atof(argv[3]);

    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if(rank == 0){
            cout << "Starting process 0... ";
            sol = cuadratic_pos(sol,vars); 
            cout << "Process 0 return x = " << sol.pos << endl;
        }
        else{
            cout << "Starting process 1... ";
            sol = cuadratic_neg(sol,vars); 
            cout << "Process 1 return x = " << sol.neg << endl;
        }

    MPI_Finalize();

}
