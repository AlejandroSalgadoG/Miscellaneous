#include "GuitarFactory.h"

Instrument* GuitarFactory::createInstrument(){
    return new Guitar();
}
