#pragma once

#include "AbstractFactory.h"
#include "Apartment.h"
#include "WhiteApartment.h"
#include "House.h"
#include "WhiteHouse.h"

class FactoryWhite : public AbstractFactory{

    public:
        Apartment* createApartment();
        House* createHouse();

};
