#include <iostream>
#include <unistd.h>

using namespace std;

int global = 0;

int main(int argc, char *argv[]){

    int local1 = 0;

    pid_t id = fork();

    int local2 = 0;

    if(id == 0){

        global = 1;
        local1 = 2;
        local2 = 3;

        cout << "Child: global = " << global << endl;
        cout << "Child: local1 = " << local1 << endl;
        cout << "Child: local2 = " << local2 << endl;
    }
    else{

        global = 4;
        local1 = 5;
        local2 = 6;

        cout << "Parent: global = " << global << endl;
        cout << "Parent: local1 = " << local1 << endl;
        cout << "Parent: local2 = " << local2 << endl;
    }
}
