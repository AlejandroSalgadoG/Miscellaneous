#pragma once

#include "InstrumentFactory.h"
#include "Drum.h"

class DrumFactory : public InstrumentFactory{

    public:
        Instrument * createInstrument();
};
