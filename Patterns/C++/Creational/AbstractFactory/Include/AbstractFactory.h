#pragma once

#include "Apartment.h"
#include "House.h"

class AbstractFactory{
	public:
		virtual Apartment* createApartment() = 0;
		virtual House* createHouse() = 0;
};
