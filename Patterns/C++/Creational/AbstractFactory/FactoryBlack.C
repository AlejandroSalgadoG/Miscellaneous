#include "FactoryBlack.h"

Apartment* FactoryBlack::createApartment(){
    return new BlackApartment();
}

House* FactoryBlack::createHouse(){
    return new BlackHouse();
}
