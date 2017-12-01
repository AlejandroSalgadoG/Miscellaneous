#include <iostream>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>

#include "Reader.h"
#include "Info.h"
#include "BigJacobi.h"

using namespace std;

int main(int argc, char *argv[]){

    if(argc == 1){
        cerr << "ERROR: missing arguments, use -h to display help" << endl;
        return 1;
    }

    int help = strcmp(argv[1],"-h");

    if(help == 0){
        cout << "MainBigJacobi -h | mfile n per toler niter vfile" << endl;
        cout << "\tmfile = name of the matrix file (string)" << endl;
        cout << "\tn = size of the matrix (int)" << endl;
        cout << "\tper = percentage of free memory to be used (double)" << endl;
        cout << "\ttoler = method tolerance (double)" << endl;
        cout << "\tniter = number of iterations (int)" << endl;
        cout << "\tvfile = name of the initial gues file (string)" << endl;

        return 0;
    }
    else if(!argv[2]){
        cerr << "No size specified" << endl;
        return 1;
    }
    else if(!argv[3]){
        cerr << "No memory rate specified (70% Recommended)" << endl;
        return 1;
    }
    else if(!argv[4]){
        cerr << "No tolerance specified" << endl;
        return 1;
    }
    else if(!argv[5]){
        cerr << "No iterations specified" << endl;
        return 1;
    }
    else if(!argv[6]){
        cerr << "No initial guess specified" << endl;
        return 1;
    }

    int n = atoi(argv[2]);
    double rate = atof(argv[3]);
    double toler = atof(argv[4]);
    int niter = atoi(argv[5]);

    if(rate <= 0 || rate > 1){
        cerr << "Bad rate (70% Recommended)" << endl;
        return 1;
    }

    int fd, page_sz;
    long int mem_sz, file_sz;

    page_sz = getpagesize();
    mem_sz = getMem(rate);
    file_sz = getFileSize(argv[1]);

    //Debug
    //cout << "Page size = " << page_sz << " bytes" << endl;
    //cout << "Memory specified = " << mem_sz << " bytes" << endl;
    //cout << "File size = " << file_sz << " bytes" << endl << endl;

    if(file_sz <= mem_sz){
        cerr << "Your matrix can be loaded using " << rate * 100
             << "% of free memory, you should use "
             << "normal or parallel jacobi instead." << endl << endl;

        cerr << "Mem_sz = " << mem_sz << " bytes" << endl;
        cerr << "File_sz = " << file_sz << " bytes" << endl;

        return 1;
    }
    else if(mem_sz < page_sz){
        cerr << "This program needs at least " << page_sz << " bytes of free "
             << "memory to run." << endl;
        cerr << rate * 100 << "% of free memory is not enough." << endl;
        cerr << "please increase the rate or the amount of free memory." << endl;

        return 1;
    }

    cout << "Looking for file " << argv[1] << "...";
    fd = open(argv[1], 0006);

    if(fd == -1){
        cerr << "Error opening the file " << argv[1]
             << ", are you sure the path is ok?" << endl;
        return 1;
    }

    cout << "File founded" << endl;

    long int map_cnt, sz, page_cnt;
    page_cnt = mem_sz / page_sz;
    sz = page_sz * page_cnt;
    map_cnt = file_sz / sz;

    if( (file_sz % sz) != 0) map_cnt++;

    double * Ab = allocVector(sz);
    double * x = readVector(n, argv[6]);

    initialize_bigjacobi(n, fd, page_sz, map_cnt, mem_sz, sz);
    bigjacobi(Ab, x, n, toler, niter); 

    return 0;
}
