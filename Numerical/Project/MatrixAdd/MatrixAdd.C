#include <iostream>
#include <mpi.h>

#include "Initializer.h"
#include "Distribute.h"
#include "Reader.h"

using namespace std;

int main(int argc, char *argv[]){

    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if(rank == 0){

            int n = 5;
            int * A = new int[n*n];
            int * B = new int[n*n];

            initializeMatrix(A, B, n);
            rnk_info * info = distributeMatrix(size, n*n);

            for(int arr_st, arr_sz, i=1;i<size;i++){
                arr_st = info[i].start;
                arr_sz = info[i].size;

                MPI_Send(&arr_sz, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(A+arr_st, arr_sz, MPI_INT, i, 0, MPI_COMM_WORLD);
                MPI_Send(B+arr_st, arr_sz, MPI_INT, i, 0, MPI_COMM_WORLD);
            }

            int * C = new int[n*n];
            for(int i=0; i<info[0].size; i++) C[i] = A[i] + B[i];

            for(int arr_st, arr_sz, i=1;i<size;i++){
                arr_st = info[i].start;
                arr_sz = info[i].size;

                int * c = new int[arr_sz];
                MPI_Recv(c, arr_sz, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

                for(int j=0;j<arr_sz;j++) C[arr_st++] = c[j];

                delete c;
            }

            printMatrix(C, n);

            delete info;
            delete A;
            delete B;
            delete C;
        }
        else{
            int arr_size;
            MPI_Recv(&arr_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            int * A = new int[arr_size];
            int * B = new int[arr_size];

            MPI_Recv(A, arr_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            MPI_Recv(B, arr_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            int * C = new int[arr_size];

            for(int i=0;i<arr_size;i++) C[i] = A[i] + B[i];

            MPI_Send(C, arr_size, MPI_INT, 0, 0, MPI_COMM_WORLD);

            delete A;
            delete B;
            delete C;
        }

    MPI_Finalize();
}
