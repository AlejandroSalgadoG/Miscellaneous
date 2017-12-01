#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>

#define NUM_THREADS 2
#define STDIN 1

void* Print(void * arg){
    int * num = (int*) arg;

    char msg[21] = "Hello from thread  \n";
    msg[18] = *num + '0';

    write(STDIN, msg, 21);

    pthread_exit(NULL);
}

int main(int argc, char *argv[]){

    pthread_t thread1, thread2;

    int param1 = 1;
    printf("creating thread number 1\n");
    pthread_create(&thread1, NULL, Print, &param1);

    int param2 = 2;
    printf("creating thread number 2\n");
    pthread_create(&thread2, NULL, Print, &param2);

    pthread_exit(0);
}
