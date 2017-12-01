#include <iostream>
#include <unistd.h>

using namespace std;

int main(int argc, char *argv[]){

    pid_t id = fork();

    if(id == 0){
        cout << "Child: I'm going to be replaced with echo" << endl;

        char * arg0 = (char*) "echo";
        char * arg1 = (char*) "message";
        char * arg2 = NULL;

        char * args[] = {arg0, arg1, arg2};

        execv("/usr/bin/echo",args);
    }
    else
        cout << "Parent: Hello from parent code" << endl;
}
