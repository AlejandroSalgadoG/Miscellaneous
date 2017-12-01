#include "FactoryCreator.h"

AbstractFactory* FactoryCreator::createFactoryWhite(){
    return new FactoryWhite();
}

AbstractFactory* FactoryCreator::createFactoryBlack(){
    return new FactoryBlack();
}
