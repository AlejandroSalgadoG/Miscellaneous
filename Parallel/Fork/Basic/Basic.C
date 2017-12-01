#include <iostream>
#include <unistd.h>

using namespace std;

int main(int argc, char *argv[]){

    pid_t id = fork();

    if(id == 0)
        cout << "Hello from child" << endl;
    else
        cout << "Hello from parent" << endl;
}
