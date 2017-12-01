#include <iostream>
#include <iomanip>
#include <mpi.h>

#include "Reader.h"
#include "Distribute.h"
#include "Norm.h"

using namespace std;

int main(int argc, char *argv[]){

    int rnk, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);

        MPI_Comm_rank(MPI_COMM_WORLD, &rnk);
        MPI_Comm_size(MPI_COMM_WORLD, &size);


    if(argc == 2 && strcmp(argv[1],"-h") == 0){
        if(rnk == 0){
            cout << "MainJacobi -h | n toler niter mfile vfile" << endl;
            cout << "\tn = size of the matrix (int)" << endl;
            cout << "\ttoler = method tolerance (double)" << endl;
            cout << "\tniter = number of iterations (int)" << endl;
            cout << "\tmfile = name of the matrix file (string)" << endl;
            cout << "\tvfile = name of the initial gues file (string)" << endl;
        }

        MPI_Finalize();
        return 0;
    }
    else if(argc != 6){
        if(rnk == 0)
            cerr << "ERROR: bad arguments, use -h to display help" << endl;

        MPI_Finalize();
        return 1;
    }

        if(rnk == 0){


            cout << "Size of the A matrix has been set to " << argv[1] << endl;
            int n = atoi(argv[1]);

            cout << "Tolerance has been set to " << argv[2] << endl;
            double toler = atof(argv[2]);

            cout << "Method iterations is going to be set to " << argv[3] << endl;
            int niter = atoi(argv[3]);

            double * Ab = readMatrixAsVector(n, argv[4]);
            double * x = readVector(n, argv[5]);

            rnk_info * info = distributeMatrix(size-1, n);


            for(int row_st, row_cnt, pos_st, arr_sz, i=0;i<size-1;i++){
                row_st = info[i].start;
                row_cnt = info[i].size;
                arr_sz = (n+1)*row_cnt;
                pos_st = row_st*(n+1);

                MPI_Send(&n, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                MPI_Send(&row_st, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                MPI_Send(&row_cnt, 1, MPI_INT, i+1, 0, MPI_COMM_WORLD);
                MPI_Send(Ab+pos_st, arr_sz, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD);
            }

            cout << "Initial x = ";
            printVector(x, n);
            cout << endl;

            cout << setprecision(14) <<
                    setw(4) << "iter" <<
                    setw(23) << "x" <<
                    setw(23) << "error" << endl;

            int cnt = 0;
            int x_ncnt = 0;
            double error = toler + 1;
            double * x_n = new double[n];

            int keep_going = 1;

            while(error > toler && cnt++ < niter){

                for(int i=1;i<size;i++){
                    MPI_Send(&keep_going, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
                    MPI_Send(x, n, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
                }

                x_ncnt = 0;

                for(int row_st, row_cnt, i=0;i<size-1;i++){
                    row_st = info[i].start;
                    row_cnt = info[i].size;

                    double * x_par = new double[row_cnt];
                    MPI_Recv(x_par, row_cnt, MPI_DOUBLE, i+1, 0, MPI_COMM_WORLD, &status);

                    for(int j=0;j<row_cnt;j++) // If process exced size skip
                        x_n[x_ncnt++] = x_par[j];

                }

                error = maxNorm(x, x_n, n);

                cout << setprecision(14) <<
                        setw(4) << cnt <<
                        setw(23) << x_n[0] << endl;

                for(int i=1;i<n-1;i++)
                    cout << setprecision(14) << setw(27) << x_n[i] << endl;

                cout << setprecision(14) <<
                        setw(27) << x_n[n-1] <<
                        setw(23) << error << endl;

                cout << "--------------------------------------------------" << endl;

                for(int i=0;i<n;i++)
                    x[i] = x_n[i];
            }

            keep_going = 0;
            for(int i=1;i<size;i++)
                MPI_Send(&keep_going, 1, MPI_INT, i, 0, MPI_COMM_WORLD);

            if(error < toler){
                cout << endl << "x = ";
                printVector(x_n, n);
                cout << "is a root" <<
                        ", error = " << toler << endl;
            }
            else{
                cerr << endl <<
                        "Can't find any solution with " << niter << " iterations" << endl;
            }

            delete x_n;
            delete info;
            delete Ab;
            delete x;
        }
        else{

            int row_sz;
            MPI_Recv(&row_sz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            int row_st;
            MPI_Recv(&row_st, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            int row_cnt;
            MPI_Recv(&row_cnt, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            int arr_sz = (row_sz+1) * row_cnt;
            double * A = new double[arr_sz];
            MPI_Recv(A, arr_sz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

            double ** Ab = new double* [row_cnt];
            for(int i=0;i<row_cnt;i++)
                Ab[i] = A+(row_sz+1)*i;

            double * x = new double[row_sz];
            double * x_n = new double[row_cnt];

            int keep_going;
            MPI_Recv(&keep_going, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

            double calc_acum;

            while(keep_going){

                MPI_Recv(x, row_sz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);

                for(int i=0;i<row_cnt;i++){
                    calc_acum = 0;
                    for(int j=0;j<row_sz;j++){
                        if(j == i+row_st) continue;
                        else calc_acum += Ab[i][j] * x[j];
                    }
                    x_n[i] = (Ab[i][row_sz] - calc_acum) / Ab[i][i+row_st];
                }

                MPI_Send(x_n, row_cnt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);

                MPI_Recv(&keep_going, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }

            //cout << "Process " << rnk << " done" << endl;

            delete Ab;
            delete A;
            delete x;
            delete x_n;
        }

    MPI_Finalize();
    return 0;
}
