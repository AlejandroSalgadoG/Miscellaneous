#pragma once

#include "TrumpetFactory.h"
#include "DrumFactory.h"
#include "GuitarFactory.h"

class FactoryCreator{

    public:
        static InstrumentFactory* createTrumpetFactory();
        static InstrumentFactory* createDrumFactory();
        static InstrumentFactory* createGuitarFactory();
};
