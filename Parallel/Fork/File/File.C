#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/wait.h>

using namespace std;

int main(int argc, char *argv[]){

    int fd = open("file.txt", O_CREAT|O_WRONLY, 0600);

    pid_t id = fork();

    if(id == 0){
      write(fd, "first", 5);
        _exit(0);
    }
    else{
        int chld_stat;
        wait(&chld_stat);

        write(fd, " second", 7);
    }
}
