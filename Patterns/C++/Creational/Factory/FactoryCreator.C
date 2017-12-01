#include "FactoryCreator.h"

InstrumentFactory* FactoryCreator::createTrumpetFactory(){
    return new TrumpetFactory();
}

InstrumentFactory* FactoryCreator::createDrumFactory(){
    return new DrumFactory();
}

InstrumentFactory* FactoryCreator::createGuitarFactory(){
    return new GuitarFactory();
}
