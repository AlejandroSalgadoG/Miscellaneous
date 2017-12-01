#include "FactoryWhite.h"

Apartment* FactoryWhite::createApartment(){
    return new WhiteApartment();
}

House* FactoryWhite::createHouse(){
    return new WhiteHouse();
}
