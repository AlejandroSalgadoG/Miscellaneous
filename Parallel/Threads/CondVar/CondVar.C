#include <stdio.h>
#include <pthread.h>
#include <unistd.h>

#include "CondVar.h"

int global = 0;
int cond_val;
bool cond = false;

void* wait(void* t){

    int * val = (int*) t;
    int id = *val;

    pthread_mutex_lock(&mutex);

        cond_val = 10;

        printf("Thread %d: doing something...\n",id);
        printf("Thread %d: global = %d, I've to wait util global = %d\n", id, global, cond_val);
        pthread_cond_wait(&condVar, &mutex);
        printf("Thread %d: Signal recived, now I can finish my job\n", id);

    pthread_mutex_unlock(&mutex);
}

void* increment(void* t){

    int * val = (int*) t;
    int id = *val;

    while(!cond){
        pthread_mutex_lock(&mutex);

            if(global == cond_val){
                printf("Thread %d: Condition reached\n",id);
                printf("Thread %d: Checking if the signal was already sent...",id);

                if(!cond){
                    printf("no, sending signal\n");
                    pthread_cond_signal(&condVar);
                    cond = true;
                }
                else printf("yes\n");
            }
            else printf("Thread %d: Incrementing global to %d\n", id, ++global);

        pthread_mutex_unlock(&mutex);
        sleep(1);
    }
}

int main(int argc, char *argv[]){
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&condVar, NULL);

    printf("Main: creating thread 1\n");
    pthread_create(&thread1, NULL, wait, &t1);

    printf("Main: creating thread 2\n");
    pthread_create(&thread2, NULL, increment, &t2);

    printf("Main: creating thread 3\n");
    pthread_create(&thread3, NULL, increment, &t3);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    pthread_mutex_destroy(&mutex);
    pthread_cond_destroy(&condVar);

    pthread_exit(0);
}
