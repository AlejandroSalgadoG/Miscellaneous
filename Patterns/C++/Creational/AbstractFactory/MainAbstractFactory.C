#include "FactoryCreator.h"

int main(){

    AbstractFactory * factoryWhite = FactoryCreator::createFactoryWhite();
    AbstractFactory * factoryBlack = FactoryCreator::createFactoryBlack();

    Apartment * whiteApartment = factoryWhite->createApartment();
    Apartment * blackApartment = factoryBlack->createApartment();

    House * whiteHouse = factoryWhite->createHouse();
    House * blackHouse = factoryBlack->createHouse();

    whiteApartment->getApartmentColor();
    blackApartment->getApartmentColor();

    whiteHouse->getHouseColor();
    blackHouse->getHouseColor();

    delete factoryWhite;
    delete factoryBlack;
    delete whiteApartment;
    delete blackApartment;
    delete whiteHouse;
    delete blackHouse;
}
