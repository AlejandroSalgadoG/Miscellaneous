#include "Employee.h"

Employee::Employee(const char* name): Worker(name) {}

void Employee::displayStructure(){
    cout << name << endl;
}

void Employee::add(Worker * worker){}
void Employee::remove(Worker * worker){}
