#pragma once

#include "AbstractFactory.h"
#include "FactoryWhite.h"
#include "FactoryBlack.h"

class FactoryCreator{
    public:
        static AbstractFactory* createFactoryWhite();
        static AbstractFactory* createFactoryBlack();
};
