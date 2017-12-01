#pragma once

#include "Io.h"
#include "Clothe.h"
#include <list>

class Outfit{

    list<Clothe*> shoped;

    public:
        ~Outfit();
        void addClothe(Clothe * clothe);
        void listClothe();
};
