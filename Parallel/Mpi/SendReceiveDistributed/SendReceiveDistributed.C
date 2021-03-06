#include "stdlib.h"
#include "stdio.h"
#include "mpi.h"

int main(int argc, char *argv[]){

    int rank;

    int data[] = {1,2,3};
    int dataSize = 3;

    int src = 0;
    int dest = 1;
    int tag = 0;

    MPI_Status status;

    MPI_Init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        if(rank == 0){
            MPI_Send(data, dataSize, MPI_INT, dest, tag, MPI_COMM_WORLD);
            system("hostname");
        }

        if(rank == 1){
            MPI_Recv(data, dataSize, MPI_INT, src, tag, MPI_COMM_WORLD, &status);

            system("hostname");
            printf("Data[0] = %d\n", data[0]);
            printf("Data[1] = %d\n", data[1]);
            printf("Data[2] = %d\n", data[2]);
        }


    MPI_Finalize();

}
