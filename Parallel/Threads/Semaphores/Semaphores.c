#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#include "Semaphores.h"

int global = 0;

void* update1(void * t){
    int * val = (int*) t;

    sem_wait(&mutex);

        printf("Thread 1: Get global = %d, global must be incremented %d\n", global,*val);
        sleep(2); //Simulate the worst case, evil scheduler
        printf("Thread 1: Changing global value from %d to %d\n",global,global+*val);

        global += *val;

    sem_post(&mutex);
}

void* update2(void * t){
    int * val = (int*) t;

    sem_wait(&mutex);
        printf("Thread 2: Something happened, global has to be %d now\n", *val);
        global += *val;
    sem_post(&mutex);
}

int main(int argc, char *argv[]){
    sem_init(&mutex, 0, 1);

    printf("Main: creating thread 1\n");
    pthread_create(&thread1, NULL, update1, &t1);

    printf("Main: creating thread 2\n");
    pthread_create(&thread2, NULL, update2, &t2);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    sem_destroy(&mutex);

    pthread_exit(0);
}
