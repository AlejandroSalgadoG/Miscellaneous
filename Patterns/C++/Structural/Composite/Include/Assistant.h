#pragma once

#include "Io.h"
#include "Worker.h"

class Assistant : public Worker{

    public:
        Assistant(const char* name);

        void add(Worker * worker);
        void remove(Worker * worker);
        void displayStructure();
};
