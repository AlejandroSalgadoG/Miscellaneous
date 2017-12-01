#include "TrumpetFactory.h"

Instrument* TrumpetFactory::createInstrument(){
    return new Trumpet();
}
