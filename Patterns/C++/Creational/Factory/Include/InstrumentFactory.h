#pragma once

#include "Instrument.h"

class InstrumentFactory{

    public:
        virtual Instrument* createInstrument() = 0;
};
