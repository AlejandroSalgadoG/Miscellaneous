#pragma once

#include "AbstractFactory.h"
#include "Apartment.h"
#include "BlackApartment.h"
#include "House.h"
#include "BlackHouse.h"

class FactoryBlack : public AbstractFactory{

    public:
        Apartment* createApartment();
        House* createHouse();

};
