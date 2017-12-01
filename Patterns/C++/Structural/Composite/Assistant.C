#include "Assistant.h"

Assistant::Assistant(const char* name): Worker(name) {}

void Assistant::displayStructure(){
    cout << name << endl;
}

void Assistant::add(Worker * worker){}
void Assistant::remove(Worker * worker){}
