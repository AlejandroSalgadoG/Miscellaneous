#include "DrumFactory.h"

Instrument* DrumFactory::createInstrument(){
    return new Drum();
}
