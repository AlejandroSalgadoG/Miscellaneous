#pragma once

#include "InstrumentFactory.h"
#include "Guitar.h"

class GuitarFactory : public InstrumentFactory{

    public:
        Instrument * createInstrument();
};
