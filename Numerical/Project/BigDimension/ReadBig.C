#include <sys/mman.h>
#include <streambuf>
#include <iostream>
#include <sstream>

#include "ReadBig.h"

void * mapped_file;
char * lett;
std::string line;

long int y = 0;

using namespace std;

struct membuf : streambuf{
    membuf(char * start, char * end){
        this->setg(start,start,end);
    }
};

long int readBig(double * Ab, int fd, long int sz, long int page_cnt, int page_sz){

    long int indx = 0;

    //Debug
    //cout << "Starting the mapping of " << sz << " bytes of the matrix...";

    mapped_file = mmap(0, sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, page_sz*page_cnt);

    //Debug
    //cout << "done" << endl;

    lett = (char*) mapped_file;

    membuf sbuf(lett, lett + sz);
    istream in(&sbuf);

    //Debug
    //cout << "Starting vector read...";
    while (getline(in, line)) {
        istringstream reader(line);
        while(reader >> Ab[indx]){
            //Debug
            //cout << Ab[indx] << " ";
            indx++;
        }
        //Debug
        //cout << endl;

    }
    //Debug
    //cout << "done" << endl;

    //Debug
    //cout << "Unmaping the matrix...";
    munmap(mapped_file, page_sz);
    //Debug
    //cout << "done" << endl << endl;


    return indx;
}
