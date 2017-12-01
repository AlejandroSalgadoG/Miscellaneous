#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>

#include "Reader.h"

using namespace std;

double** allocMatrix(int n){

    cout << "Reserving memory for a matrix of size " << n << "x" << n << "...";
    double ** A = new double* [n];

    for(int i=0;i<n;i++)
        A[i] = new double[n+1];

    cout << "done" << endl;

    return A;
}

double** allocMatrix(int row, int col){

    cout << "Reserving memory for a matrix of size " << row << "x" << col << "...";
    double ** A = new double*[row];

    for(int i=0;i<row;i++)
        A[i] = new double[col];

    cout << "done" << endl;

    return A;
}

double* allocVector(int n){

    cout << "Reserving memory for a vector of size " << n << "...";
    double * x = new double[n];
    cout << "done" << endl;

    return x;
}

double* readVector(int n, char * file){

    cout << "Reserving memory for a vector of size " << n << "...";
    double * A = new double[n];
    cout << "done" << endl;

    cout << "Looking for file " << file << "...";
    ifstream matrix(file);

    if(matrix) cout << "File founded" << endl;
    else{
        cout << "Error reading the file, are you sure the path is ok?" << endl;
        exit(1);
    }

    string line;
    int i = 0;

    cout << "Starting reading the file...";

    while(getline(matrix,line)){
        istringstream reader(line);
        reader >> A[i];
        i++;
    }

    cout << "done" << endl << endl;

    return A;
}

double** readMatrix(int n, char * file){

    double ** A = allocMatrix(n);

    cout << "Looking for file " << file << "...";
    ifstream matrix(file);

    if(matrix) cout << "File founded" << endl;
    else{
        cout << "Error reading the file, are you sure the path is ok?" << endl;
        exit(1);
    }

    string line;
    int i=0,j;

    cout << "Starting reading the file...";

    while(getline(matrix,line)){
        istringstream reader(line);

        j=0;
        while(reader >> A[i][j]) j++;
        i++;
    }

    cout << "done" << endl << endl;

    return A;
}

double* readMatrixAsVector(int n, char * file){

    double * A = new double[n*(n+1)];

    cout << "Looking for file " << file << "...";
    ifstream matrix(file);

    if(matrix) cout << "File founded" << endl;
    else{
        cout << "Error reading the file, are you sure the path is ok?" << endl;
        exit(1);
    }

    string line;
    int i=0;

    cout << "Starting reading the file...";

    double a;

    while(getline(matrix,line)){
        istringstream reader(line);

        while(reader >> A[i]) i++;
    }

    cout << "done" << endl << endl;

    return A;

}

void printMatrix(double ** A, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n+1;j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}

void printMatrix(double ** A, int row, int col){
    for(int i=0;i<row;i++){
        for(int j=0;j<col+1;j++)
            cout << A[i][j] << " ";
        cout << endl;
    }
}

void printMatrix(int * A, int n){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++)
            cout << A[n*i+j] << " ";
        cout << endl;
    }
}

void printVector(double * A, int n){
    for(int i=0;i<n;i++)
        cout << A[i] << " ";
    cout << endl;
}
