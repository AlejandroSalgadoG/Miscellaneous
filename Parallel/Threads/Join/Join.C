#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

void *function1(void *t){

    int * tid = (int*) t;
    printf("Thread %d starting...\n",*tid);

    sleep(5);

    printf("Thread %d done.\n",*tid);

    pthread_exit(t);
}

void *function2(void *t){

    int * tid = (int*) t;
    printf("Thread %d starting...\n",*tid);

    sleep(3);

    printf("Thread %d done.\n",*tid);

    pthread_exit(t);
}

int main (int argc, char *argv[]){

    pthread_t thread1;
    pthread_t thread2;

    int tid1 = 1;
    int tid2 = 2;

    printf("Main: creating thread 1\n");
    pthread_create(&thread1, NULL, function1, &tid1);

    printf("Main: creating thread 2\n");
    pthread_create(&thread2, NULL, function2, &tid2);

    void *status1;
    void *status2;

    pthread_join(thread1, &status1);
    int * ret1 = (int*) status1;
    printf("Main: completed join of thread 1 having a status of %d\n", *ret1);

    pthread_join(thread2, &status2);
    int * ret2 = (int*) status2;
    printf("Main: completed join of thread 2 having a status of %d\n", *ret2);

    printf("Main: program completed. Exiting.\n");

    pthread_exit(0);
}
