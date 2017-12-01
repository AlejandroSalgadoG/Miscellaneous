#pragma once

#include "InstrumentFactory.h"
#include "Trumpet.h"

class TrumpetFactory : public InstrumentFactory{

    public:
        Instrument* createInstrument();
};
