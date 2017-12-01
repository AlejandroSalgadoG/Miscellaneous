#pragma once

#include "Io.h"
#include "Worker.h"

class Employee : public Worker{

    public:
        Employee(const char* name);

        void add(Worker * worker);
        void remove(Worker * worker);
        void displayStructure();
};
