#include "BuilderVestCap.h"

BuilderVestCap::BuilderVestCap(){
    outfit = new Outfit();
}

void BuilderVestCap::buildCoat(){
    outfit->addClothe( new Vest() );
}

void BuilderVestCap::buildHat(){
    outfit->addClothe( new Cap() );
}

Outfit* BuilderVestCap::getOutfit(){
    return outfit;
}
