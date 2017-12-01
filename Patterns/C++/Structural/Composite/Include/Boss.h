#pragma once

#include "Io.h"
#include "Worker.h"
#include <list>

class Boss : public Worker{

    list<Worker*> workers;

    public:
        Boss(const char* name);
        ~Boss();

        void add(Worker * worker);
        void remove(Worker * worker);
        void displayStructure();
};
