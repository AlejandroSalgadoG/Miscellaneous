#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include "Mutex.h"

int global = 0;

using namespace std;

void * update1(void * t){
    int * val = (int*) t;

    pthread_mutex_lock(&mutex);

        printf("Thread 1: Get global = %d, global must be incremented %d\n", global,*val);
        sleep(2); //Simulate the worst case, evil scheduler
        printf("Thread 1: Changing global value from %d to %d\n",global,global+*val);

        global += *val;

    pthread_mutex_unlock(&mutex);
}

void * update2(void * t){
    int * val = (int*) t;

    pthread_mutex_lock(&mutex);

        printf("Thread 2: Something happened, global has to be %d now\n", *val);
        global += *val;

    pthread_mutex_unlock(&mutex);
}

int main(int argc, char *argv[]){

    pthread_mutex_init(&mutex, NULL);

    printf("Main: creating thread 1\n");
    pthread_create(&thread1, NULL, update1, &t1);

    printf("Main: creating thread 2\n");
    pthread_create(&thread2, NULL, update2, &t2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&mutex);

    pthread_exit(0);
}
