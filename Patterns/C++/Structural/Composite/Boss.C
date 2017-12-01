#include "Boss.h"

Boss::Boss(const char* name): Worker(name) {}

Boss::~Boss(){
    for(Worker * worker: workers)
        delete worker;
}

void Boss::add(Worker * worker){
    workers.push_back(worker);
}

void Boss::remove(Worker * worker){
    workers.remove(worker);
    delete worker;
}

void Boss::displayStructure(){
    cout << name << endl;

    for(Worker * worker: workers)
        worker->displayStructure();
}
