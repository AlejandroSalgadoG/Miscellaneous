#include <iostream>
#include <unistd.h>

#include "BigJacobi.h"
#include "ReadBig.h"

#include "Reader.h"
#include "Norm.h"

int fd;
int page_sz;
long int map_cnt;
long int mem_sz;
long int sz;

long int frag;
bool frag_indic = false;

double * acum;
double * fact;
double * b;

using namespace std;

void initialize_bigjacobi(int n, int filed, int pagesz, long int mapcnt,
                          long int memsz, long int size){
    fd = filed;
    page_sz = pagesz;
    map_cnt = mapcnt;
    mem_sz = memsz;
    sz = size;

    acum = allocVector(n);
    for(int i=0;i<n;i++) acum[i] = 0;

    fact = allocVector(n);
    b = allocVector(n);
}

void acum_partial(double * Ab, double * x, int n, int st, int pos, int row, int sz){
    n++; // take into account b

    //take into account that Ab cant have the same index as x
    for(int i=pos,j=st;i<(pos+sz);i++,j++){
        if(j == row) fact[row] = Ab[i];
        else if(j == n-1) b[row] = Ab[i];
        else acum[row] += Ab[i]*x[j];
    }
}

int acum_row(double * Ab, double * x, int n, int row_st, int row_rd, int row_cnt, int part){
    n++; // take into account b

    //take into account partial reading (row_st+row_rd)
    for(long int i=row_st;i<row_st+row_rd;i++,row_cnt++){ //dont forget row_cnt

        //take into account partial reading (-part)
        for(long int j=0;j<n-1;j++){
            if(j == row_cnt) fact[j] = Ab[i*n+j - part];
            else acum[i] += Ab[i*n+j - part] * x[j];
        }
        b[i] = Ab[i*n+(n-1) - part];
    }

    return row_cnt;
}

double* calcNewVector(double * x_n, int n){
    for(int i=0;i<n;i++){
        x_n[i] = (b[i] - acum[i]) / fact[i];
    }

    return x_n;
}

void bigjacobi(double * Ab, double * x, int n, double toler, int niter){

    int cnt = 0;
    double error = toler +1;

    long int readed, page_cnt;
    int row_st, row_rd, row_cnt = 0, part = 0, new_part;

    cout << endl;
    double * x_n = allocVector(n);
    cout << endl;

    while(error > toler && cnt++ < niter){

        page_cnt = 0;
        frag = 0;
        for(int i=0;i<map_cnt;i++){
            readed = readBig(Ab, fd, sz, page_cnt, page_sz);

            frag += readed;

	        page_cnt += mem_sz / page_sz;

            readed += part; // take into account partial reading

            row_rd = readed / (n+1); //take b into account

            if(part != 0){
                //Read last part of partial reading
                acum_partial(Ab, x, n, part, 0, row_cnt, n+1-part);

                row_st = 1; //Notify about partial reading
                row_cnt++; //First row already readed
                row_rd--; //No need to read fist row
            }
            else row_st = 0; //No partial reading

            row_cnt = acum_row(Ab, x, n, row_st, row_rd, row_cnt, part);

            new_part = readed % (n+1); //calculate amount of partial reading

            //Read first part of partial reading
            acum_partial(Ab,x, n, 0, (row_st+row_rd)*(n+1)-part, row_cnt, new_part);

            //update partial count
            part = new_part;
        }

        if(frag != (n+1)*n && !frag_indic){
            cerr << "WARNING: fragmentation detected, maps = "
                 << map_cnt << " fragmentation = "
                 << frag - (n+1)*n << endl << endl;
            frag_indic = true;
        }

        x_n = calcNewVector(x_n, n);
        error = maxNorm(x, x_n, n);

        cout << "Iteration " << cnt << ", error = " << error << endl;

        //Debug
        //printVector(x, n);
        //printVector(x_n, n);
        //cout << endl << endl;

        for(int i=0;i<n;i++)
            x[i] = x_n[i];
    }

    if(error < toler){
        cout << endl << "x = ";
        printVector(x_n, n);
        cout << "is a root, "
             << "error = " << toler << endl;
    }
    else{
        cerr << endl
             << "Can't find any solution with "
             << niter << " iterations" << endl;
    }

    delete Ab;
    delete x;

    close(fd);
}
